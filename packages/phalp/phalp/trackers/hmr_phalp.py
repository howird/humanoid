from phalp.trackers.phalp_base import PHALPBase
from phalp.models.hmar.hmr2 import HMR2023TextureSampler
from phalp.utils import get_pylogger

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio

log = get_pylogger(__name__)


class HMR2PHALP(PHALPBase):
    """Extended PHALP tracker with HMR2023 texture sampling capabilities."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2023TextureSampler(self.cfg)

    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        (
            pred_bbox,
            pred_bbox,
            pred_masks,
            pred_scores,
            pred_classes,
            ground_truth_track_id,
            ground_truth_annotations,
        ) = super().get_detections(image, frame_name, t_, additional_data, measurments)

        # Pad bounding boxes
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox, self.cfg.expand_bbox_shape)

        return (
            pred_bbox,
            pred_bbox_padded,
            pred_masks,
            pred_scores,
            pred_classes,
            ground_truth_track_id,
            ground_truth_annotations,
        )
