import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances

from humanoid_vision.utils.utils_detectron2_phalp import DefaultPredictor_with_RPN
from humanoid_vision.trackers.phalp import PHALP
from humanoid_vision.utils.pylogger_phalp import get_pylogger

log = get_pylogger(__name__)


class PHALPGT(PHALP):
    """Extended PHALP tracker with HMR2023 texture sampling capabilities."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        from humanoid_vision.models.hmar.hmar import HMAR

        log.info("Loading HMAR model...")
        self.HMAR = HMAR(self.cfg)
        self.HMAR.load_weights(self.cfg.hmr.hmar_path)

    def setup_detectron2(self):
        super().setup_detectron2()
        self.setup_detectron2_with_RPN()

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

    def get_detections(self, image, frame_name, t_, additional_data={}, measurments=None):
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

            return (
                pred_bbox,
                pred_bbox,
                pred_masks,
                pred_scores,
                pred_classes,
                ground_truth_track_id,
                ground_truth_annotations,
            )
        else:
            return super().get_detections(image, frame_name, t_)
