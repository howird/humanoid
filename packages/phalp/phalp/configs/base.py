from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, List

CACHE_DIR = Path.home() / ".cache"


@dataclass
class VideoIOConfig:
    output_dir: Path = Path("outputs") / "tracking"

    extract_video: bool = True
    delete_frame_dir: bool = False
    base_path: Optional[Path] = None

    start_frame: int = 0
    end_frame: int = 1300

    useffmpeg: bool = False

    # this will be used if extract_video=False
    start_time: str = "0s"
    end_time: str = "10s"

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if self.base_path and isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)

        if self.output_dir.is_file():
            raise ValueError(f"Output path, {self.output_dir}, must be a directory.")


@dataclass
class PHALPConfig:
    # Predictions methods: T: UV image, P: pose, L: location
    predict: Tuple[Literal["T", "P", "L"], ...] = ("T", "P", "L")
    # Distance metric for poses
    pose_distance: Literal["smpl", "joints"] = "smpl"
    distance_type: Literal["EQ_019"] = "EQ_019"
    alpha: float = 0.1
    low_th_c: float = 0.8
    hungarian_th: float = 100.0
    track_history: int = 7
    max_age_track: int = 50
    n_init: int = 5
    encode_type: str = "4c"
    past_lookback: int = 1
    detector: str = "vitdet"

    shot: int = 0
    start_frame: int = -1
    end_frame: int = 10

    small_w: int = 0
    small_h: int = 0


@dataclass
class PosePredictorConfig:
    config_path: Path = CACHE_DIR / "phalp/weights/pose_predictor.yaml"
    weights_path: Path = CACHE_DIR / "phalp/weights/pose_predictor.pth"
    mean_std: Path = CACHE_DIR / "phalp/3D/mean_std.npy"


@dataclass
class AVAConfig:
    ava_labels_path: Path = CACHE_DIR / "phalp/ava/ava_labels.pkl"
    ava_class_mappping_path: Path = CACHE_DIR / "phalp/ava/ava_class_mapping.pkl"


@dataclass
class HMRConfig:
    hmar_path: Path = CACHE_DIR / "phalp/weights/hmar_v2_weights.pth"


@dataclass
class RenderConfig:
    enable: bool = True
    # rendering type
    type: Literal["HUMAN_MESH", "HUMAN_MASK", "HUMAN_BBOX"] = "HUMAN_MESH"
    up_scale: int = 2
    res: int = 256
    side_view_each: bool = False
    metallicfactor: float = 0.0
    roughnessfactor: float = 0.7
    colors: str = "phalp"
    head_mask: bool = False
    head_mask_path: Path = CACHE_DIR / "phalp/3D/head_faces.npy"
    output_resolution: int = 1440
    fps: int = 30
    blur_faces: bool = False
    show_keypoints: bool = False


@dataclass
class PostProcessConfig:
    apply_smoothing: bool = True
    phalp_pkl_path: Path = Path("_OUT/videos_v0")
    save_fast_tracks: bool = False


@dataclass
class SMPLConfig:
    MODEL_PATH: Path = Path("data/smpl")
    GENDER: str = "neutral"
    MODEL_TYPE: str = "smpl"
    NUM_BODY_JOINTS: int = 23
    JOINT_REGRESSOR_EXTRA: Path = CACHE_DIR / "phalp/3D/SMPL_to_J19.pkl"
    TEXTURE: Path = CACHE_DIR / "phalp/3D/texture.npz"
    MEAN_PARAMS: Path = CACHE_DIR / "phalp/3D/smpl_mean_params.npz"


@dataclass
class TransformerDecoderConfig:
    """Configuration for the transformer decoder."""

    depth: int = 6
    heads: int = 8
    mlp_dim: int = 1024
    dim_head: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0
    norm: str = "layer"
    context_dim: int = 1280


@dataclass
class SMPLHeadConfig:
    TYPE: str = "transformer_decoder"
    POOL: str = "max"
    IN_CHANNELS: int = 2048
    JOINT_REP: Literal["6d", "aa"] = "6d"
    TRANSFORMER_INPUT: Literal["zero", "mean_shape"] = "zero"
    INIT_DECODER_XAVIER: bool = False
    IEF_ITERS: int = 1
    TRANSFORMER_DECODER: TransformerDecoderConfig = field(default_factory=TransformerDecoderConfig)


@dataclass
class BackboneConfig:
    TYPE: str = "vit"
    NUM_LAYERS: int = 50
    OUT_CHANNELS: int = 2048
    PRETRAINED_WEIGHTS: Optional[Path] = None


@dataclass
class ModelConfig:
    IMAGE_SIZE: int = 256
    IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    SMPL_HEAD: SMPLHeadConfig = field(default_factory=SMPLHeadConfig)
    BACKBONE: BackboneConfig = field(default_factory=BackboneConfig)
    pose_transformer_size: int = 2048


@dataclass
class ExtraConfig:
    FOCAL_LENGTH: int = 5000
    NUM_LOG_IMAGES: int = 4
    NUM_LOG_SAMPLES_PER_IMAGE: int = 8
    PELVIS_IND: int = 39


@dataclass
class BaseConfig:
    """Base configuration for PHALP tracking system."""

    seed: int = 42
    track_dataset: str = "demo"
    device: str = "cuda"
    base_tracker: str = "PHALP"
    train: bool = False
    use_gt: bool = False
    overwrite: bool = True
    task_id: int = -1
    num_tasks: int = 100
    verbose: bool = False
    detect_shots: bool = False
    video_seq: Optional[str] = None

    # Target aspect ratio for bounding boxes (width, height)
    expand_bbox_shape: Optional[Tuple[int, int]] = (192, 256)

    # Fields
    video: VideoIOConfig = field(default_factory=VideoIOConfig)
    phalp: PHALPConfig = field(default_factory=PHALPConfig)
    pose_predictor: PosePredictorConfig = field(default_factory=PosePredictorConfig)
    ava_config: AVAConfig = field(default_factory=AVAConfig)
    hmr: HMRConfig = field(default_factory=HMRConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)
    SMPL: SMPLConfig = field(default_factory=SMPLConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    EXTRA: ExtraConfig = field(default_factory=ExtraConfig)

    # tmp configs
    hmr_type: str = "hmr2018"


@dataclass
class DatasetTrainConfig:
    """Configuration for training datasets."""

    H36M_TRAIN_WMASK: Dict[str, float] = field(default_factory=lambda: {"WEIGHT": 0.1})
    MPII_TRAIN_WMASK: Dict[str, float] = field(default_factory=lambda: {"WEIGHT": 0.1})
    COCO_TRAIN_2014_WMASK_PRUNED: Dict[str, float] = field(default_factory=lambda: {"WEIGHT": 0.1})
    COCO_TRAIN_2014_VITPOSE_REPLICATE_PRUNED12: Dict[str, float] = field(default_factory=lambda: {"WEIGHT": 0.1})
    MPI_INF_TRAIN_PRUNED: Dict[str, float] = field(default_factory=lambda: {"WEIGHT": 0.02})
    AVA_TRAIN_MIDFRAMES_1FPS_WMASK: Dict[str, float] = field(default_factory=lambda: {"WEIGHT": 0.19})
    AIC_TRAIN_WMASK: Dict[str, float] = field(default_factory=lambda: {"WEIGHT": 0.19})
    INSTA_TRAIN_WMASK: Dict[str, float] = field(default_factory=lambda: {"WEIGHT": 0.2})


@dataclass
class DatasetValConfig:
    """Configuration for validation datasets."""

    COCO_VAL: Dict[str, float] = field(default_factory=lambda: {"WEIGHT": 1.0})


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""

    SCALE_FACTOR: float = 0.3
    ROT_FACTOR: int = 30
    TRANS_FACTOR: float = 0.02
    COLOR_SCALE: float = 0.2
    ROT_AUG_RATE: float = 0.6
    TRANS_AUG_RATE: float = 0.5
    DO_FLIP: bool = True
    FLIP_AUG_RATE: float = 0.5
    EXTREME_CROP_AUG_RATE: float = 0.1
    EXTREME_CROP_AUG_LEVEL: int = 1


@dataclass
class DatasetsConfig:
    """Configuration for all datasets."""

    SUPPRESS_KP_CONF_THRESH: float = 0.3
    FILTER_NUM_KP: int = 4
    FILTER_NUM_KP_THRESH: float = 0.0
    FILTER_REPROJ_THRESH: int = 31000
    SUPPRESS_BETAS_THRESH: float = 3.0
    SUPPRESS_BAD_POSES: bool = True
    POSES_BETAS_SIMULTANEOUS: bool = True
    FILTER_NO_POSES: bool = False
    TRAIN: DatasetTrainConfig = field(default_factory=DatasetTrainConfig)
    VAL: DatasetValConfig = field(default_factory=DatasetValConfig)
    MOCAP: str = "CMU-MOCAP"
    CONFIG: DatasetConfig = field(default_factory=DatasetConfig)
    BETAS_REG: bool = True


@dataclass
class LossWeightsConfig:
    """Configuration for loss weights."""

    KEYPOINTS_3D: float = 0.05
    KEYPOINTS_2D: float = 0.01
    GLOBAL_ORIENT: float = 0.001
    BODY_POSE: float = 0.001
    BETAS: float = 0.0005
    ADVERSARIAL: float = 0.0005


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    _target_: str = "pytorch_lightning.Trainer"
    default_root_dir: str = "${paths.output_dir}"
    accelerator: str = "gpu"
    devices: int = 8
    deterministic: bool = False
    num_sanity_val_steps: int = 0
    log_every_n_steps: int = "${GENERAL.LOG_STEPS}"
    val_check_interval: int = "${GENERAL.VAL_STEPS}"
    precision: int = 16
    max_steps: int = "${GENERAL.TOTAL_STEPS}"
    move_metrics_to_cpu: bool = True
    limit_val_batches: int = 1
    track_grad_norm: int = -1
    strategy: str = "ddp"
    num_nodes: int = 1
    sync_batchnorm: bool = True


@dataclass
class PathsConfig:
    """Configuration for paths."""

    root_dir: str = "${oc.env:PROJECT_ROOT}"
    data_dir: str = "${paths.root_dir}/data/"
    log_dir: str = "/fsx/shubham/code/hmr2023/logs_hydra/"
    output_dir: str = "${hydra:runtime.output_dir}"
    work_dir: str = "${hydra:runtime.cwd}"


@dataclass
class GeneralConfig:
    """Configuration for general training parameters."""

    TOTAL_STEPS: int = 1000000
    LOG_STEPS: int = 1000
    VAL_STEPS: int = 1000
    CHECKPOINT_STEPS: int = 10000
    CHECKPOINT_SAVE_TOP_K: int = 1
    NUM_WORKERS: int = 6
    PREFETCH_FACTOR: int = 2


@dataclass
class TrainingConfig:
    LR: float = 1.0e-5
    WEIGHT_DECAY: float = 0.0001
    BATCH_SIZE: int = 48
    LOSS_REDUCTION: str = "mean"
    NUM_TRAIN_SAMPLES: int = 2
    NUM_TEST_SAMPLES: int = 64
    POSE_2D_NOISE_RATIO: float = 0.01
    SMPL_PARAM_NOISE_RATIO: float = 0.005


@dataclass
class TrainConfig(BaseConfig):
    """Configuration for training the model."""

    task_name: str = "train"
    tags: List[str] = field(default_factory=lambda: ["dev"])
    train: bool = True
    test: bool = False
    ckpt_path: Optional[str] = None
    seed: Optional[int] = None

    # Nested configurations
    TRAINING: TrainingConfig = field(default_factory=TrainingConfig)
    DATASETS: DatasetsConfig = field(default_factory=DatasetsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    GENERAL: GeneralConfig = field(default_factory=GeneralConfig)
    LOSS_WEIGHTS: LossWeightsConfig = field(default_factory=LossWeightsConfig)

    exp_name: str = "hmr2"
