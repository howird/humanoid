import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn

from scenedetect import AdaptiveDetector, detect

from humanoid_vision.datasets.utils import expand_bbox_to_aspect_ratio

from humanoid_vision.configs.base import BaseConfig
from humanoid_vision.deep_sort.tracker import Tracker
from humanoid_vision.models.predictors.pose_transformer_v2 import PoseTransformerV2
from humanoid_vision.models.hmar.hmr2 import HMAROutput, HMR2023TextureSampler
from humanoid_vision.common.detection import Detection

from humanoid_vision.visualize.postprocessor import Postprocessor
from humanoid_vision.visualize.visualizer import Visualizer

from humanoid_vision.utils.utils import progress_bar, smpl_to_pose_camera_vector
from humanoid_vision.utils.utils_detectron2_phalp import DefaultPredictor_Lazy
from humanoid_vision.utils.video_writer import VideoWriter
from humanoid_vision.utils.bbox import get_cropped_image
from humanoid_vision.utils.pylogger_phalp import get_pylogger

log = get_pylogger(__name__)


class PHALP(nn.Module):
    def __init__(
        self,
        cfg: BaseConfig,
        hmr_model: HMR2023TextureSampler,
        pose_predictor: PoseTransformerV2,
        detector: DefaultPredictor_Lazy,
    ):
        super(PHALP, self).__init__()

        self.cfg = cfg
        self.device = torch.device(self.cfg.device)

        # Store models
        self.HMAR = hmr_model
        self.pose_predictor = pose_predictor
        self.detector = detector

        self.visualizer = Visualizer(self.cfg, self.HMAR)
        self.tracker = Tracker(
            self.cfg,
            self.HMAR,
            self.pose_predictor,
            max_age=self.cfg.phalp.max_age_track,
            n_init=self.cfg.phalp.n_init,
            dims=(4096, 4096, 99),
        )

        # by default this will not be initialized
        # TODO(howird): add a flag to enable it
        if False:
            self.postprocessor = Postprocessor(self.cfg, self)

        self.to(self.device)
        self.train() if (self.cfg.train) else self.eval()

    def track(self, video_name: str, list_of_frames: List[Path]):
        eval_keys = ["tracked_ids", "tracked_bbox", "tid", "bbox", "tracked_time"]
        history_keys = ["appe", "loca", "pose", "uv"] if self.cfg.render.enable else []
        prediction_keys = ["prediction_uv", "prediction_pose", "prediction_loca"] if self.cfg.render.enable else []
        extra_keys_1 = ["center", "scale", "size", "img_path", "img_name", "class_name", "conf", "annotations"]
        extra_keys_2 = ["smpl", "camera", "camera_bbox", "3d_joints", "2d_joints", "mask"]
        history_keys = history_keys + extra_keys_1 + extra_keys_2
        visual_store_ = eval_keys + history_keys + prediction_keys
        tmp_keys_ = ["uv", "prediction_uv", "prediction_pose", "prediction_loca"]

        (self.cfg.video.output_dir / video_name).mkdir(parents=True, exist_ok=True)

        pkl_path = self.cfg.video.output_dir / f"{self.cfg.track_dataset}_{video_name}.pkl"
        video_path = self.cfg.video.output_dir / f"{self.cfg.base_tracker}_{video_name}.mp4"

        # check if the video is already processed
        if not (self.cfg.overwrite) and pkl_path.is_file():
            return 0

        # eval mode
        self.eval()

        log.info(f"Saving tracks at : {self.cfg.video.output_dir}")

        list_of_frames = (
            list_of_frames
            if self.cfg.phalp.start_frame == -1
            else list_of_frames[self.cfg.phalp.start_frame : self.cfg.phalp.end_frame]
        )
        list_of_shots = self.get_list_of_shots(video_name, list_of_frames)

        tracked_frames = []
        final_visuals_dic = {}

        with VideoWriter(self.cfg.video, self.cfg.render.fps) as vwriter:
            for t, frame_name in progress_bar(
                enumerate(list_of_frames),
                description=f"Tracking : {video_name}",
                total=len(list_of_frames),
                disable=False,
            ):
                image_frame = cv2.imread(str(frame_name))
                img_height, img_width, _ = image_frame.shape
                new_image_size = max(img_height, img_width)
                top, left = (
                    (new_image_size - img_height) // 2,
                    (new_image_size - img_width) // 2,
                )
                measurments = [img_height, img_width, new_image_size, left, top]
                self.cfg.phalp.shot = 1 if t in list_of_shots else 0

                if self.cfg.render.enable:
                    # reset the renderer
                    # TODO: add a flag for full resolution rendering
                    self.cfg.render.up_scale = int(self.cfg.render.output_resolution / self.cfg.render.res)
                    self.visualizer.reset_render(self.cfg.render.res * self.cfg.render.up_scale)

                ############ detection ##############
                pred_bbox, pred_bbox_pad, pred_masks, pred_scores, pred_classes, gt_tids, gt_annots = (
                    self.get_detections(image_frame, frame_name, t)
                )

                ############ HMAR ##############
                detections = self.get_human_features(
                    image_frame,
                    pred_masks,
                    pred_bbox,
                    pred_bbox_pad,
                    pred_scores,
                    frame_name,
                    pred_classes,
                    t,
                    measurments,
                    gt_tids,
                    gt_annots,
                )

                ############ tracking ##############
                self.tracker.predict()
                self.tracker.update(detections, t, frame_name, self.cfg.phalp.shot)

                ############ record the results ##############
                final_visuals_dic.setdefault(
                    frame_name, {"time": t, "shot": self.cfg.phalp.shot, "frame_path": frame_name}
                )

                if self.cfg.render.enable:
                    final_visuals_dic[frame_name]["frame"] = image_frame

                for key_ in visual_store_:
                    final_visuals_dic[frame_name][key_] = []

                ############ record the track states (history and predictions) ##############
                for tracks_ in self.tracker.tracks:
                    if frame_name not in tracked_frames:
                        tracked_frames.append(frame_name)
                    if not (tracks_.is_confirmed()):
                        continue

                    track_id = tracks_.track_id
                    track_data_hist = tracks_.track_data["history"][-1]
                    track_data_pred = tracks_.track_data["prediction"]

                    final_visuals_dic[frame_name]["tid"].append(track_id)
                    final_visuals_dic[frame_name]["bbox"].append(track_data_hist["bbox"])
                    final_visuals_dic[frame_name]["tracked_time"].append(tracks_.time_since_update)

                    for hkey_ in history_keys:
                        final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
                    for pkey_ in prediction_keys:
                        final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split("_")[1]][-1])

                    if tracks_.time_since_update == 0:
                        final_visuals_dic[frame_name]["tracked_ids"].append(track_id)
                        final_visuals_dic[frame_name]["tracked_bbox"].append(track_data_hist["bbox"])

                        if tracks_.hits == self.cfg.phalp.n_init:
                            for pt in range(self.cfg.phalp.n_init - 1):
                                track_data_hist_ = tracks_.track_data["history"][-2 - pt]
                                track_data_pred_ = tracks_.track_data["prediction"]
                                frame_name_ = tracked_frames[-2 - pt]
                                final_visuals_dic[frame_name_]["tid"].append(track_id)
                                final_visuals_dic[frame_name_]["bbox"].append(track_data_hist_["bbox"])
                                final_visuals_dic[frame_name_]["tracked_ids"].append(track_id)
                                final_visuals_dic[frame_name_]["tracked_bbox"].append(track_data_hist_["bbox"])
                                final_visuals_dic[frame_name_]["tracked_time"].append(0)

                                for hkey_ in history_keys:
                                    final_visuals_dic[frame_name_][hkey_].append(track_data_hist_[hkey_])
                                for pkey_ in prediction_keys:
                                    final_visuals_dic[frame_name_][pkey_].append(
                                        track_data_pred_[pkey_.split("_")[1]][-1]
                                    )

                ############ save the video ##############
                if self.cfg.render.enable and t >= self.cfg.phalp.n_init:
                    d_ = self.cfg.phalp.n_init + 1 if (t + 1 == len(list_of_frames)) else 1
                    for t_ in range(t, t + d_):
                        frame_key = list_of_frames[t_ - self.cfg.phalp.n_init]
                        rendered, f_size = self.visualizer.render_video(final_visuals_dic[frame_key])

                        # save the rendered frame
                        vwriter.save_video(video_path, rendered, f_size, t=t_ - self.cfg.phalp.n_init)

                        # delete the frame after rendering it
                        del final_visuals_dic[frame_key]["frame"]

                        # delete unnecessary keys
                        for tkey_ in tmp_keys_:
                            del final_visuals_dic[frame_key][tkey_]

            joblib.dump(final_visuals_dic, pkl_path, compress=3)

        if self.cfg.use_gt:
            joblib.dump(
                self.tracker.tracked_cost,
                self.cfg.video.output_dir / f"{video_name}_{self.cfg.phalp.start_frame}_distance.pkl",
            )

        return final_visuals_dic, pkl_path

    def get_detections(self, image, frame_name, t, additional_data={}, measurments=None):
        outputs = self.detector(image)
        instances = outputs["instances"]
        instances = instances[instances.pred_classes == 0]
        instances = instances[instances.scores > self.cfg.phalp.low_th_c]

        pred_bbox = instances.pred_boxes.tensor.cpu().numpy()
        pred_masks = instances.pred_masks.cpu().numpy()
        pred_scores = instances.scores.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()

        ground_truth_track_id = [1 for _ in range(len(pred_scores))]
        ground_truth_annotations = [[] for _ in range(len(pred_scores))]

        # Pad bounding boxes
        pred_bbox_padded = (
            pred_bbox
            if self.cfg.expand_bbox_shape is None
            else expand_bbox_to_aspect_ratio(pred_bbox, self.cfg.expand_bbox_shape)
        )

        return (
            pred_bbox,
            pred_bbox_padded,
            pred_masks,
            pred_scores,
            pred_classes,
            ground_truth_track_id,
            ground_truth_annotations,
        )

    def get_human_features(
        self,
        image,
        seg_mask,
        bbox,
        bbox_pad,
        score,
        frame_name,
        cls_id,
        t,
        measurments,
        gt,
        ann,
    ) -> List[Detection]:
        num_detected_persons = len(score)
        if num_detected_persons == 0:
            log.warning(f"No people found in {frame_name}.")
            return []

        img_height, img_width, new_image_size, left, top = measurments
        ratio = 1.0 / int(new_image_size) * self.cfg.render.res
        masked_image_list = []
        center_list = []
        scale_list = []
        rles_list = []
        selected_ids = []

        for p_ in range(num_detected_persons):
            if bbox[p_][2] - bbox[p_][0] < self.cfg.phalp.small_w or bbox[p_][3] - bbox[p_][1] < self.cfg.phalp.small_h:
                continue
            masked_image, _center, _scale, rles, center_pad, scale_pad = get_cropped_image(
                image, bbox[p_], bbox_pad[p_], seg_mask[p_]
            )
            masked_image_list.append(masked_image)
            center_list.append(center_pad)
            scale_list.append(scale_pad)
            rles_list.append(rles)
            selected_ids.append(p_)

        num_valid_persons = len(masked_image_list)
        if num_valid_persons == 0:
            log.warning("No eligible bounding boxes found, phalp.{small_w, small_h} may be set too high.")
            return []

        masked_image_list = torch.stack(masked_image_list, dim=0)

        with torch.no_grad():
            hmar_out: HMAROutput = self.HMAR(masked_image_list.cuda())
            uv_vector = hmar_out.uv_vector
            appe_embedding = self.HMAR.autoencoder_hmar(uv_vector, en=True)
            appe_embedding = appe_embedding.view(appe_embedding.shape[0], -1)

            pred_smpl_params, pred_joints_2d, pred_joints, pred_cam = self.HMAR.get_3d_parameters(
                # HACK(howird)
                dict(body_pose=hmar_out.body_pose, betas=hmar_out.betas, global_orient=hmar_out.global_orient),
                hmar_out.pred_cam,
                center=(np.array(center_list) + np.array([left, top])) * ratio,
                img_size=self.cfg.render.res,
                scale=np.max(np.array(scale_list), axis=1, keepdims=True) * ratio,
            )
            pred_smpl_params = [
                {k: v[i].cpu().numpy() for k, v in pred_smpl_params.items()} for i in range(num_valid_persons)
            ]

            if self.cfg.phalp.pose_distance == "joints":
                pose_embedding = pred_joints.cpu().view(num_valid_persons, -1)
            elif self.cfg.phalp.pose_distance == "smpl":
                pose_embedding = []
                for i in range(num_valid_persons):
                    pose_embedding_ = smpl_to_pose_camera_vector(pred_smpl_params[i], pred_cam[i])
                    pose_embedding.append(torch.from_numpy(pose_embedding_[0]))
                pose_embedding = torch.stack(pose_embedding, dim=0)
            else:
                raise ValueError("Unknown pose distance")
            pred_joints_2d_ = pred_joints_2d.reshape(num_valid_persons, -1) / self.cfg.render.res
            pred_cam_ = pred_cam.view(num_valid_persons, -1)
            pred_joints_2d_.contiguous()
            pred_cam_.contiguous()

            loca_embedding = torch.cat((pred_joints_2d_, pred_cam_, pred_cam_, pred_cam_), 1)

        # keeping it here for legacy reasons (T3DP), but it is not used.
        full_embedding = torch.cat((appe_embedding.cpu(), pose_embedding, loca_embedding.cpu()), 1)

        detection_data_list = []
        for i, p_ in enumerate(selected_ids):
            detection_data = {
                "bbox": np.array([bbox[p_][0], bbox[p_][1], (bbox[p_][2] - bbox[p_][0]), (bbox[p_][3] - bbox[p_][1])]),
                "mask": rles_list[i],
                "conf": score[p_],
                "appe": appe_embedding[i].cpu().numpy(),
                "pose": pose_embedding[i].numpy(),
                "loca": loca_embedding[i].cpu().numpy(),
                "uv": uv_vector[i].cpu().numpy(),
                "embedding": full_embedding[i],
                "center": center_list[i],
                "scale": scale_list[i],
                "smpl": pred_smpl_params[i],
                "camera": pred_cam_[i].cpu().numpy(),
                "camera_bbox": hmar_out.pred_cam[i].cpu().numpy(),
                "3d_joints": pred_joints[i].cpu().numpy(),
                "2d_joints": pred_joints_2d_[i].cpu().numpy(),
                "size": [img_height, img_width],
                "img_path": frame_name,
                "img_name": frame_name.split("/")[-1] if isinstance(frame_name, str) else None,
                "class_name": cls_id[p_],
                "time": t,
                "ground_truth": gt[p_],
                "annotations": ann[p_],
                "hmar_out_cam": hmar_out.pred_cam.view(num_valid_persons, -1).cpu().numpy(),
                "hmar_out_cam_t": hmar_out.pred_cam_t.view(num_valid_persons, -1).cpu().numpy(),
                "hmar_out_focal_length": hmar_out.focal_length.view(num_valid_persons, -1).cpu().numpy(),
            }
            detection_data_list.append(Detection(detection_data))

        return detection_data_list

    def get_list_of_shots(self, video_name, list_of_frames):
        # https://github.com/Breakthrough/PySceneDetect
        list_of_shots = []
        remove_tmp_video = False
        if self.cfg.detect_shots:
            if isinstance(list_of_frames[0], str):
                # make a video if list_of_frames is frames
                video_tmp_name = self.cfg.video.output_dir / "_TMP" / f"{video_name}.mp4"
                for ft_, fname_ in enumerate(list_of_frames):
                    im_ = cv2.imread(fname_)
                    if ft_ == 0:
                        video_file = cv2.VideoWriter(
                            str(video_tmp_name),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            24,
                            frameSize=(im_.shape[1], im_.shape[0]),
                        )
                    video_file.write(im_)
                video_file.release()
                remove_tmp_video = True
            elif isinstance(list_of_frames[0], tuple):
                video_tmp_name = list_of_frames[0][0]
            else:
                raise Exception("Unknown type of list_of_frames")

            # Detect scenes in a video using PySceneDetect.
            scene_list = detect(str(video_tmp_name), AdaptiveDetector())

            if remove_tmp_video:
                video_tmp_name.unlink()  # Use pathlib to remove file

            for scene in scene_list:
                list_of_shots.append(scene[0].get_frames())
                list_of_shots.append(scene[1].get_frames())
            list_of_shots = np.unique(list_of_shots)
            list_of_shots = list_of_shots[1:-1]
            log.info(f"Detected shot change at frames: {list_of_shots}.")

        return list_of_shots
