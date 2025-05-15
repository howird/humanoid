from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import LazyConfig

from phalp.configs.base import BaseConfig
from phalp.models.predictor.pose_transformer_v2 import Pose_transformer_v2
from phalp.utils.utils_detectron2 import DefaultPredictor_Lazy
from phalp.utils import get_pylogger

log = get_pylogger(__name__)


def setup_predictor(cfg: BaseConfig, smpl_model) -> Pose_transformer_v2:
    """Initialize the pose predictor model."""
    log.info("Loading Predictor model...")
    pose_predictor = Pose_transformer_v2(cfg, smpl_model)
    pose_predictor.load_weights(cfg.pose_predictor.weights_path)
    return pose_predictor


def setup_detectron2(cfg: BaseConfig) -> DefaultPredictor_Lazy:
    """Initialize the object detector model."""
    log.info("Loading Detection model...")
    if cfg.phalp.detector == "maskrcnn":
        detectron2_cfg = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5  # type: ignore
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4  # type: ignore
    elif cfg.phalp.detector == "vitdet":
        import phalp

        cfg_path = Path(phalp.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"  # type: ignore
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5  # type: ignore
    else:
        raise ValueError(f"Detector {cfg.phalp.detector} not supported")

    return DefaultPredictor_Lazy(detectron2_cfg)
