import warnings

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances
from pycocotools import mask as mask_utils
from scenedetect import AdaptiveDetector, detect
from sklearn.linear_model import Ridge

from phalp.configs.base import CACHE_DIR, BaseConfig

from phalp.external.deep_sort_ import nn_matching
from phalp.external.deep_sort_.detection import Detection
from phalp.external.deep_sort_.tracker import Tracker
from phalp.external.deep_sort_.track import Track

from phalp.models.predictor import Pose_transformer_v2

from phalp.utils import get_pylogger

from phalp.utils.io_manager import IOManager
from phalp.utils.utils import get_prediction_interval, progress_bar, smpl_to_pose_camera_vector
from phalp.utils.utils_dataset import process_image, process_mask
from phalp.utils.utils_detectron2 import DefaultPredictor_Lazy, DefaultPredictor_with_RPN
from phalp.utils.utils_download import cache_url

from phalp.visualize.postprocessor import Postprocessor
from phalp.visualize.visualizer import Visualizer

log = get_pylogger(__name__)


@dataclass
class FrameKeys:
    """Keys for storing different types of data in the tracking results"""

    eval_keys: List[str]
    history_keys: List[str]
    prediction_keys: List[str]
    temporary_keys: List[str]


class PHALP(nn.Module):
    def __init__(self, cfg: BaseConfig):
        super(PHALP, self).__init__()

        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        self.io_manager = IOManager(self.cfg.video, self.cfg.render.fps)

        # download wights and configs from Google Drive
        self.cached_download_from_drive()

        # setup HMR, and pose_predictor. Override this function to use your own model
        self.setup_hmr()

        # setup temporal pose predictor
        self.setup_predictor()

        # setup Detectron2, override this function to use your own model
        self.setup_detectron2()

        # create a visualizer
        self.setup_visualizer()

        # move to device
        self.to(self.device)

        # train or eval
        self.train() if (self.cfg.train) else self.eval()

    def setup_hmr(self):
        from phalp.models.hmar import HMAR

        log.info("Loading HMAR model...")
        self.HMAR = HMAR(self.cfg)
        self.HMAR.load_weights(self.cfg.hmr.hmar_path)

    def setup_predictor(self):
        log.info("Loading Predictor model...")
        self.pose_predictor = Pose_transformer_v2(self.cfg, self)
        self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)

    def setup_detectron2(self):
        log.info("Loading Detection model...")
        if self.cfg.phalp.detector == "maskrcnn":
            self.detectron2_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
            )
            self.detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            self.detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(self.detectron2_cfg)
            self.class_names = self.detector.metadata.get("thing_classes")
        elif self.cfg.phalp.detector == "vitdet":
            from detectron2.config import LazyConfig
            import phalp

            cfg_path = Path(phalp.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
            self.detectron2_cfg = LazyConfig.load(str(cfg_path))
            self.detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                self.detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
            self.detector = DefaultPredictor_Lazy(self.detectron2_cfg)
        else:
            raise ValueError(f"Detector {self.cfg.phalp.detector} not supported")

        # for predicting masks with only bounding boxes, e.g. for running on ground truth tracks
        self.setup_detectron2_with_RPN()
        # TODO: make this work with DefaultPredictor_Lazy

    def setup_detectron2_with_RPN(self):
        self.detectron2_cfg = get_cfg()
        self.detectron2_cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        )
        self.detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.detectron2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
        self.detectron2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
        self.detectron2_cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN_with_proposals"
        self.detector_x = DefaultPredictor_with_RPN(self.detectron2_cfg)

    def setup_deepsort(self):
        log.info("Setting up DeepSort...")
        metric = nn_matching.NearestNeighborDistanceMetric(
            self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback
        )
        self.tracker = Tracker(
            self.cfg,
            metric,
            max_age=self.cfg.phalp.max_age_track,
            n_init=self.cfg.phalp.n_init,
            phalp_tracker=self,
            dims=[4096, 4096, 99],
        )

    def setup_visualizer(self):
        log.info("Setting up Visualizer...")
        self.visualizer = Visualizer(self.cfg, self.HMAR)

    def setup_postprocessor(self):
        # by default this will not be initialized
        self.postprocessor = Postprocessor(self.cfg, self)

    def default_setup(self, video_name):
        tracking_output_path = self.cfg.video.output_dir / video_name
        tracking_output_path.mkdir(parents=True, exist_ok=True)
        # (tracking_output_path / "results_tracks").mkdir(parents=True, exist_ok=True)
        # (tracking_output_path / "_TMP").mkdir(parents=True, exist_ok=True)
        # (tracking_output_path / "_DEMO").mkdir(parents=True, exist_ok=True)

    def track(self):
        """
        Main tracking pipeline that processes a video and tracks people.

        Returns:
            Tuple[Dict, Path]: Tracking results dictionary and path to the saved pickle file
            or 0 if the video was already processed
        """

        # Define keys for storing different types of data
        eval_keys = ["tracked_ids", "tracked_bbox", "tid", "bbox", "tracked_time"]
        history_keys = ["appe", "loca", "pose", "uv"] if self.cfg.render.enable else []
        prediction_keys = ["prediction_uv", "prediction_pose", "prediction_loca"] if self.cfg.render.enable else []
        extra_keys_1 = ["center", "scale", "size", "img_path", "img_name", "class_name", "conf", "annotations"]
        extra_keys_2 = ["smpl", "camera", "camera_bbox", "3d_joints", "2d_joints", "mask", "extra_data"]
        history_keys = history_keys + extra_keys_1 + extra_keys_2
        visual_store_ = eval_keys + history_keys + prediction_keys
        tmp_keys_ = ["uv", "prediction_uv", "prediction_pose", "prediction_loca"]

        keys = FrameKeys(
            eval_keys=eval_keys, history_keys=history_keys, prediction_keys=prediction_keys, temporary_keys=tmp_keys_
        )

        # Process the source video and return a list of frames
        # Source can be a video file, a YouTube link or an image folder
        video_name, list_of_frames, additional_data = self.io_manager.get_frames_from_source()

        pkl_path = Path(self.cfg.video.output_dir) / f"{self.cfg.track_dataset}_{video_name}.pkl"
        video_path = Path(self.cfg.video.output_dir) / f"{self.cfg.base_tracker}_{video_name}.mp4"

        # Check if the video is already processed
        if not (self.cfg.overwrite) and pkl_path.is_file():
            return 0

        # Set to eval mode
        self.eval()

        # Setup rendering, deep sort and directory structure
        self.setup_deepsort()
        self.default_setup(video_name)

        log.info(f"Saving tracks at : {self.cfg.video.output_dir}")

        # Apply frame range limits if specified
        list_of_frames = (
            list_of_frames
            if self.cfg.phalp.start_frame == -1
            else list_of_frames[self.cfg.phalp.start_frame : self.cfg.phalp.end_frame]
        )

        # Get list of shot changes in the video
        list_of_shots = self.get_list_of_shots(video_name, list_of_frames)

        tracked_frames: List[str] = []
        final_visuals_dic: Dict[str, Dict[str, Any]] = {}

        # Process each frame
        for t_, frame_name in progress_bar(
            enumerate(list_of_frames),
            description=f"Tracking : {video_name}",
            total=len(list_of_frames),
            disable=False,
        ):
            # Read the frame and get its dimensions
            image_frame = self.io_manager.read_frame(frame_name)
            img_height, img_width, _ = image_frame.shape
            new_image_size = max(img_height, img_width)
            top, left = (
                (new_image_size - img_height) // 2,
                (new_image_size - img_width) // 2,
            )
            measurements = [img_height, img_width, new_image_size, left, top]
            self.cfg.phalp.shot = 1 if t_ in list_of_shots else 0

            # Reset renderer if rendering is enabled
            if self.cfg.render.enable:
                self.cfg.render.up_scale = int(self.cfg.render.output_resolution / self.cfg.render.res)
                self.visualizer.reset_render(self.cfg.render.res * self.cfg.render.up_scale)

            ############ Detection Stage ##############
            # Get detections from the current frame
            pred_bbox, pred_bbox_pad, pred_masks, pred_scores, pred_classes, gt_tids, gt_annots = self.get_detections(
                image_frame, frame_name, t_, additional_data, measurements
            )

            ############ Feature Extraction Stage ##############
            # Extract human features using HMAR
            detections = self.get_human_features(
                image_frame,
                pred_masks,
                pred_bbox,
                pred_bbox_pad,
                pred_scores,
                frame_name,
                pred_classes,
                t_,
                measurements,
                gt_tids,
                gt_annots,
            )

            ############ Tracking Stage ##############
            tracking_result = self.tracker.process_frame(detections, t_, frame_name, self.cfg.phalp.shot)

            ############ Results Recording Stage ##############
            # Initialize the frame entry in the results dictionary
            final_visuals_dic.setdefault(
                frame_name, {"time": t_, "shot": self.cfg.phalp.shot, "frame_path": frame_name}
            )

            # Store the frame if rendering is enabled
            if self.cfg.render.enable:
                final_visuals_dic[frame_name]["frame"] = image_frame

            # Initialize empty lists for all data keys
            for key_ in visual_store_:
                final_visuals_dic[frame_name][key_] = []

            ############ Track State Recording Stage ##############
            # Record the state of each confirmed track
            self._record_track_states(
                tracking_result.active_tracks, frame_name, tracked_frames, final_visuals_dic, keys
            )

            ############ Video Saving Stage ##############
            # Render and save video frames if enabled
            if self.cfg.render.enable and t_ >= self.cfg.phalp.n_init:
                self._save_rendered_frames(t_, list_of_frames, final_visuals_dic, video_path, keys.temporary_keys)

        # Save the final results
        joblib.dump(final_visuals_dic, pkl_path, compress=3)
        self.io_manager.close_video()

        # Save tracking costs if using ground truth
        if self.cfg.use_gt:
            joblib.dump(
                self.tracker.tracked_cost,
                Path(self.cfg.video.output_dir) / f"{video_name}_{self.cfg.phalp.start_frame}_distance.pkl",
            )

        return final_visuals_dic, pkl_path

    def _record_track_states(
        self,
        active_tracks: List[Track],
        frame_name: str,
        tracked_frames: List[str],
        final_visuals_dic: Dict[str, Dict[str, Any]],
        keys,
    ) -> None:
        """
        Record the states of active tracks in the results dictionary.

        Args:
            active_tracks: List of active tracks
            frame_name: Name of the current frame
            tracked_frames: List of frames that have been tracked
            final_visuals_dic: Dictionary to store results
            keys: FrameKeys object containing key definitions
        """
        if frame_name not in tracked_frames:
            tracked_frames.append(frame_name)

        for track in active_tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            track_data_hist = track.track_data["history"][-1]
            track_data_pred = track.track_data["prediction"]

            # Record basic track information
            final_visuals_dic[frame_name]["tid"].append(track_id)
            final_visuals_dic[frame_name]["bbox"].append(track_data_hist["bbox"])
            final_visuals_dic[frame_name]["tracked_time"].append(track.time_since_update)

            # Record history and prediction data
            for hkey_ in keys.history_keys:
                final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
            for pkey_ in keys.prediction_keys:
                final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split("_")[1]][-1])

            # For tracks that were just updated (not missed)
            if track.time_since_update == 0:
                final_visuals_dic[frame_name]["tracked_ids"].append(track_id)
                final_visuals_dic[frame_name]["tracked_bbox"].append(track_data_hist["bbox"])

                # For newly confirmed tracks, backfill previous frames
                if track.hits == self.cfg.phalp.n_init:
                    self._backfill_track_history(track, tracked_frames, final_visuals_dic, keys)

    def _backfill_track_history(
        self, track: Track, tracked_frames: List[str], final_visuals_dic: Dict[str, Dict[str, Any]], keys
    ) -> None:
        """
        Backfill track history for newly confirmed tracks.

        Args:
            track: The track to backfill
            tracked_frames: List of frames that have been tracked
            final_visuals_dic: Dictionary to store results
            keys: FrameKeys object containing key definitions
        """
        track_id = track.track_id
        track_data_pred = track.track_data["prediction"]

        for pt in range(self.cfg.phalp.n_init - 1):
            track_data_hist_ = track.track_data["history"][-2 - pt]
            frame_name_ = tracked_frames[-2 - pt]

            # Add track information to previous frames
            final_visuals_dic[frame_name_]["tid"].append(track_id)
            final_visuals_dic[frame_name_]["bbox"].append(track_data_hist_["bbox"])
            final_visuals_dic[frame_name_]["tracked_ids"].append(track_id)
            final_visuals_dic[frame_name_]["tracked_bbox"].append(track_data_hist_["bbox"])
            final_visuals_dic[frame_name_]["tracked_time"].append(0)

            # Add history and prediction data
            for hkey_ in keys.history_keys:
                final_visuals_dic[frame_name_][hkey_].append(track_data_hist_[hkey_])
            for pkey_ in keys.prediction_keys:
                final_visuals_dic[frame_name_][pkey_].append(track_data_pred[pkey_.split("_")[1]][-1])

    def _save_rendered_frames(
        self,
        t_: int,
        list_of_frames: List[str],
        final_visuals_dic: Dict[str, Dict[str, Any]],
        video_path: Path,
        tmp_keys: List[str],
    ) -> None:
        """
        Render and save video frames.

        Args:
            t_: Current frame index
            list_of_frames: List of all frames
            final_visuals_dic: Dictionary containing tracking results
            video_path: Path to save the video
            tmp_keys: List of temporary keys to delete after rendering
        """
        # Determine how many frames to render
        d_ = self.cfg.phalp.n_init + 1 if (t_ + 1 == len(list_of_frames)) else 1

        for t__ in range(t_, t_ + d_):
            frame_key = list_of_frames[t__ - self.cfg.phalp.n_init]
            rendered_, f_size = self.visualizer.render_video(final_visuals_dic[frame_key])

            # Save the rendered frame
            self.io_manager.save_video(video_path, rendered_, f_size, t=t__ - self.cfg.phalp.n_init)

            # Delete the frame after rendering it
            del final_visuals_dic[frame_key]["frame"]

            # Delete unnecessary keys
            for tkey_ in tmp_keys:
                if tkey_ in final_visuals_dic[frame_key]:
                    del final_visuals_dic[frame_key][tkey_]

    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        if frame_name in additional_data.keys():
            img_height, img_width, new_image_size, left, top = measurments

            gt_bbox = additional_data[frame_name]["gt_bbox"]
            if len(additional_data[frame_name]["extra_data"]["gt_track_id"]) > 0:
                ground_truth_track_id = additional_data[frame_name]["extra_data"]["gt_track_id"]
            else:
                ground_truth_track_id = [-1 for i in range(len(gt_bbox))]

            if len(additional_data[frame_name]["extra_data"]["gt_class"]) > 0:
                ground_truth_annotations = additional_data[frame_name]["extra_data"]["gt_class"]
            else:
                ground_truth_annotations = [[] for i in range(len(gt_bbox))]

            inst = Instances((img_height, img_width))
            bbox_array = []
            class_array = []
            scores_array = []

            # for ava bbox format
            # for bbox_ in gt_bbox:
            #     x1 = bbox_[0] * img_width
            #     y1 = bbox_[1] * img_height
            #     x2 = bbox_[2] * img_width
            #     y2 = bbox_[3] * img_height

            # for posetrack bbox format
            for bbox_ in gt_bbox:
                x1 = bbox_[0]
                y1 = bbox_[1]
                x2 = bbox_[2] + x1
                y2 = bbox_[3] + y1

                bbox_array.append([x1, y1, x2, y2])
                class_array.append(0)
                scores_array.append(1)

            bbox_array = np.array(bbox_array)
            class_array = np.array(class_array)
            box = Boxes(torch.as_tensor(bbox_array))
            inst.pred_boxes = box
            inst.pred_classes = torch.as_tensor(class_array)
            inst.scores = torch.as_tensor(scores_array)

            outputs_x = self.detector_x.predict_with_bbox(image, inst)
            instances_x = outputs_x["instances"]
            instances_people = instances_x[instances_x.pred_classes == 0]

            pred_bbox = instances_people.pred_boxes.tensor.cpu().numpy()
            pred_masks = instances_people.pred_masks.cpu().numpy()
            pred_scores = instances_people.scores.cpu().numpy()
            pred_classes = instances_people.pred_classes.cpu().numpy()

        else:
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

        return (
            pred_bbox,
            pred_bbox,
            pred_masks,
            pred_scores,
            pred_classes,
            ground_truth_track_id,
            ground_truth_annotations,
        )

    def get_croped_image(self, image, bbox, bbox_pad, seg_mask):
        # Encode the mask for storing, borrowed from tao dataset
        # https://github.com/TAO-Dataset/tao/blob/master/scripts/detectors/detectron2_infer.py
        masks_decoded = np.array(np.expand_dims(seg_mask, 2), order="F", dtype=np.uint8)
        rles = mask_utils.encode(masks_decoded)
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        seg_mask = seg_mask.astype(int) * 255
        if len(seg_mask.shape) == 2:
            seg_mask = np.expand_dims(seg_mask, 2)
            seg_mask = np.repeat(seg_mask, 3, 2)

        center_ = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2])
        scale_ = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])

        center_pad = np.array([(bbox_pad[2] + bbox_pad[0]) / 2, (bbox_pad[3] + bbox_pad[1]) / 2])
        scale_pad = np.array([(bbox_pad[2] - bbox_pad[0]), (bbox_pad[3] - bbox_pad[1])])
        mask_tmp = process_mask(seg_mask.astype(np.uint8), center_pad, 1.0 * np.max(scale_pad))
        image_tmp = process_image(image, center_pad, 1.0 * np.max(scale_pad))

        # bbox_        = expand_bbox_to_aspect_ratio(bbox, target_aspect_ratio=(192,256))
        # center_x     = np.array([(bbox_[2] + bbox_[0])/2, (bbox_[3] + bbox_[1])/2])
        # scale_x      = np.array([(bbox_[2] - bbox_[0]), (bbox_[3] - bbox_[1])])
        # mask_tmp     = process_mask(seg_mask.astype(np.uint8), center_x, 1.0*np.max(scale_x))
        # image_tmp    = process_image(image, center_x, 1.0*np.max(scale_x))

        masked_image = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)

        return masked_image, center_, scale_, rles, center_pad, scale_pad

    def get_human_features(
        self,
        image,
        seg_mask,
        bbox,
        bbox_pad,
        score,
        frame_name,
        cls_id,
        t_,
        measurments,
        gt=1,
        ann=None,
    ):
        NPEOPLE = len(score)

        if NPEOPLE == 0:
            log.warning(f"No people found in {frame_name}.")
            return []

        img_height, img_width, new_image_size, left, top = measurments
        ratio = 1.0 / int(new_image_size) * self.cfg.render.res
        masked_image_list = []
        center_list = []
        scale_list = []
        rles_list = []
        selected_ids = []
        for p_ in range(NPEOPLE):
            if bbox[p_][2] - bbox[p_][0] < self.cfg.phalp.small_w or bbox[p_][3] - bbox[p_][1] < self.cfg.phalp.small_h:
                continue
            masked_image, _center, _scale, rles, center_pad, scale_pad = self.get_croped_image(
                image, bbox[p_], bbox_pad[p_], seg_mask[p_]
            )
            masked_image_list.append(masked_image)
            center_list.append(center_pad)
            scale_list.append(scale_pad)
            rles_list.append(rles)
            selected_ids.append(p_)

        if len(masked_image_list) == 0:
            log.warning("No eligible bounding boxes found, phalp..{small_w, small_h} may be set too high.")
            return []

        masked_image_list = torch.stack(masked_image_list, dim=0)
        BS = masked_image_list.size(0)

        with torch.no_grad():
            extra_args = {}
            hmar_out = self.HMAR(masked_image_list.cuda(), **extra_args)
            uv_vector = hmar_out["uv_vector"]
            appe_embedding = self.HMAR.autoencoder_hmar(uv_vector, en=True)
            appe_embedding = appe_embedding.view(appe_embedding.shape[0], -1)
            pred_smpl_params, pred_joints_2d, pred_joints, pred_cam = self.HMAR.get_3d_parameters(
                hmar_out["pose_smpl"],
                hmar_out["pred_cam"],
                center=(np.array(center_list) + np.array([left, top])) * ratio,
                img_size=self.cfg.render.res,
                scale=np.max(np.array(scale_list), axis=1, keepdims=True) * ratio,
            )
            pred_smpl_params = [{k: v[i].cpu().numpy() for k, v in pred_smpl_params.items()} for i in range(BS)]

            if self.cfg.phalp.pose_distance == "joints":
                pose_embedding = pred_joints.cpu().view(BS, -1)
            elif self.cfg.phalp.pose_distance == "smpl":
                pose_embedding = []
                for i in range(BS):
                    pose_embedding_ = smpl_to_pose_camera_vector(pred_smpl_params[i], pred_cam[i])
                    pose_embedding.append(torch.from_numpy(pose_embedding_[0]))
                pose_embedding = torch.stack(pose_embedding, dim=0)
            else:
                raise ValueError("Unknown pose distance")
            pred_joints_2d_ = pred_joints_2d.reshape(BS, -1) / self.cfg.render.res
            pred_cam_ = pred_cam.view(BS, -1)
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
                "camera_bbox": hmar_out["pred_cam"][i].cpu().numpy(),
                "3d_joints": pred_joints[i].cpu().numpy(),
                "2d_joints": pred_joints_2d_[i].cpu().numpy(),
                "size": [img_height, img_width],
                "img_path": frame_name,
                "img_name": frame_name.split("/")[-1] if isinstance(frame_name, str) else None,
                "class_name": cls_id[p_],
                "time": t_,
                "ground_truth": gt[p_],
                "annotations": ann[p_],
                "hmar_out_cam": hmar_out["pred_cam"].view(BS, -1).cpu().numpy(),
                "hmar_out_cam_t": hmar_out["_pred_cam_t"].view(BS, -1).cpu().numpy(),
                "hmar_out_focal_length": hmar_out["_focal_length"].view(BS, -1).cpu().numpy(),
            }
            detection_data_list.append(Detection(detection_data))

        return detection_data_list

    def forward_for_tracking(self, vectors, attibute="A", time=1):
        """
        Predict future states of tracks based on their history.

        Args:
            vectors: List of feature vectors to use for prediction
                    For pose: [pose_vectors, position_data, time_vectors]
                    For location: [location_vectors, time_vectors, confidence_vectors]
            attibute: Type of prediction to perform
                     "P" for pose prediction using the pose transformer
                     "L" for location prediction using linear regression
            time: Time steps to predict into the future

        Returns:
            Predicted feature vectors for the specified attribute
        """
        if attibute == "P":
            vectors_pose = vectors[0]
            vectors_data = vectors[1]
            vectors_time = vectors[2]

            en_pose = torch.from_numpy(vectors_pose)
            en_data = torch.from_numpy(vectors_data)
            en_time = torch.from_numpy(vectors_time)

            if len(en_pose.shape) != 3:
                en_pose = en_pose.unsqueeze(0)  # (BS, 7, pose_dim)
                en_time = en_time.unsqueeze(0)  # (BS, 7)
                en_data = en_data.unsqueeze(0)  # (BS, 7, 6)

            with torch.no_grad():
                pose_pred = self.pose_predictor.predict_next(en_pose, en_data, en_time, time)

            return pose_pred.cpu()

        if attibute == "L":
            vectors_loca = vectors[0]
            vectors_time = vectors[1]
            vectors_conf = vectors[2]

            en_loca = torch.from_numpy(vectors_loca)
            en_time = torch.from_numpy(vectors_time)
            en_conf = torch.from_numpy(vectors_conf)
            time = torch.from_numpy(time)

            if len(en_loca.shape) != 3:
                en_loca = en_loca.unsqueeze(0)
                en_time = en_time.unsqueeze(0)
            else:
                en_loca = en_loca.permute(0, 1, 2)

            BS = en_loca.size(0)
            t_ = en_loca.size(1)

            en_loca_xy = en_loca[:, :, :90]
            en_loca_xy = en_loca_xy.view(BS, t_, 45, 2)
            en_loca_n = en_loca[:, :, 90:]
            en_loca_n = en_loca_n.view(BS, t_, 3, 3)

            new_en_loca_n = []
            for bs in range(BS):
                x0_ = np.array(en_loca_xy[bs, :, 44, 0])
                y0_ = np.array(en_loca_xy[bs, :, 44, 1])
                n_ = np.log(np.array(en_loca_n[bs, :, 0, 2]))
                t_ = np.array(en_time[bs, :])

                loc_ = torch.diff(en_time[bs, :], dim=0) != 0
                if self.cfg.phalp.distance_type == "EQ_020" or self.cfg.phalp.distance_type == "EQ_021":
                    loc_ = 1
                else:
                    loc_ = loc_.shape[0] - torch.sum(loc_) + 1

                M = t_[:, np.newaxis] ** [0, 1]
                time_ = 48 if time[bs] > 48 else time[bs]

                clf = Ridge(alpha=5.0)
                clf.fit(M, n_)
                n_p = clf.predict(np.array([1, time_ + 1 + t_[-1]]).reshape(1, -1))
                n_p = n_p[0]
                n_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                n_pi = get_prediction_interval(n_, n_hat, t_, time_ + 1 + t_[-1])

                clf = Ridge(alpha=1.2)
                clf.fit(M, x0_)
                x_p = clf.predict(np.array([1, time_ + 1 + t_[-1]]).reshape(1, -1))
                x_p = x_p[0]
                x_p_ = (x_p - 0.5) * np.exp(n_p) / 5000.0 * 256.0
                x_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                x_pi = get_prediction_interval(x0_, x_hat, t_, time_ + 1 + t_[-1])

                clf = Ridge(alpha=2.0)
                clf.fit(M, y0_)
                y_p = clf.predict(np.array([1, time_ + 1 + t_[-1]]).reshape(1, -1))
                y_p = y_p[0]
                y_p_ = (y_p - 0.5) * np.exp(n_p) / 5000.0 * 256.0
                y_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                y_pi = get_prediction_interval(y0_, y_hat, t_, time_ + 1 + t_[-1])

                new_en_loca_n.append([x_p_, y_p_, np.exp(n_p), x_pi / loc_, y_pi / loc_, np.exp(n_pi) / loc_, 1, 1, 0])
                en_loca_xy[bs, -1, 44, 0] = x_p
                en_loca_xy[bs, -1, 44, 1] = y_p

            new_en_loca_n = torch.from_numpy(np.array(new_en_loca_n))
            xt = torch.cat((en_loca_xy[:, -1, :, :].view(BS, 90), (new_en_loca_n.float()).view(BS, 9)), 1)

        return xt

    def get_uv_distance(self, t_uv, d_uv):
        t_uv = torch.from_numpy(t_uv).cuda().float()
        d_uv = torch.from_numpy(d_uv).cuda().float()
        d_mask = d_uv[3:, :, :] > 0.5
        t_mask = t_uv[3:, :, :] > 0.5

        mask_dt = torch.logical_and(d_mask, t_mask)
        mask_dt = mask_dt.repeat(4, 1, 1)
        mask_ = torch.logical_not(mask_dt)

        t_uv[mask_] = 0.0
        d_uv[mask_] = 0.0

        with torch.no_grad():
            t_emb = self.HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
            d_emb = self.HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
        t_emb = t_emb.view(-1) / 10**3
        d_emb = d_emb.view(-1) / 10**3
        return t_emb.cpu().numpy(), d_emb.cpu().numpy(), torch.sum(mask_dt).cpu().numpy() / 4 / 256 / 256 / 2

    def get_pose_distance(self, track_pose, detect_pose):
        """Compute pair-wise squared l2 distances between points in `track_pose` and `detect_pose`."""
        track_pose, detect_pose = np.asarray(track_pose), np.asarray(detect_pose)

        if self.cfg.phalp.pose_distance == "smpl":
            # remove additional dimension used for encoding location (last 3 elements)
            track_pose = track_pose[:, :-3]
            detect_pose = detect_pose[:, :-3]

        if len(track_pose) == 0 or len(detect_pose) == 0:
            return np.zeros((len(track_pose), len(detect_pose)))
        track_pose2, detect_pose2 = np.square(track_pose).sum(axis=1), np.square(detect_pose).sum(axis=1)
        r2 = -2.0 * np.dot(track_pose, detect_pose.T) + track_pose2[:, None] + detect_pose2[None, :]
        r2 = np.clip(r2, 0.0, float(np.inf))

        return r2

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

    def cached_download_from_drive(self, additional_urls=None):
        """Download a file from Google Drive if it doesn't exist yet.
        :param url: the URL of the file to download
        :param path: the path to save the file to
        """
        # Create necessary directories
        (CACHE_DIR / "phalp").mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "phalp/3D").mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "phalp/weights").mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "phalp/ava").mkdir(parents=True, exist_ok=True)

        additional_urls = additional_urls if additional_urls is not None else {}
        download_files = {
            "head_faces.npy": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/head_faces.npy",
                str(CACHE_DIR / "phalp/3D"),
            ],
            "mean_std.npy": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/mean_std.npy",
                str(CACHE_DIR / "phalp/3D"),
            ],
            "smpl_mean_params.npz": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/smpl_mean_params.npz",
                str(CACHE_DIR / "phalp/3D"),
            ],
            "SMPL_to_J19.pkl": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/SMPL_to_J19.pkl",
                str(CACHE_DIR / "phalp/3D"),
            ],
            "texture.npz": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/texture.npz",
                str(CACHE_DIR / "phalp/3D"),
            ],
            "bmap_256.npy": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/bmap_256.npy",
                str(CACHE_DIR / "phalp/3D"),
            ],
            "fmap_256.npy": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/fmap_256.npy",
                str(CACHE_DIR / "phalp/3D"),
            ],
            "hmar_v2_weights.pth": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/hmar_v2_weights.pth",
                str(CACHE_DIR / "phalp/weights"),
            ],
            "pose_predictor.pth": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/pose_predictor_40006.ckpt",
                str(CACHE_DIR / "phalp/weights"),
            ],
            "pose_predictor.yaml": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/config_40006.yaml",
                str(CACHE_DIR / "phalp/weights"),
            ],
            # data for ava dataset
            "ava_labels.pkl": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_labels.pkl",
                str(CACHE_DIR / "phalp/ava"),
            ],
            "ava_class_mapping.pkl": [
                "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_class_mappping.pkl",
                str(CACHE_DIR / "phalp/ava"),
            ],
        }
        download_files.update(additional_urls)

        for file_name, url in download_files.items():
            file_path = Path(url[1]) / file_name
            if not file_path.exists():
                print(f"Downloading file: {file_name}")
                # output = gdown.cached_download(url[0], str(file_path), fuzzy=True)
                output = cache_url(url[0], str(file_path))
                assert Path(output).exists(), f"{output} does not exist"
