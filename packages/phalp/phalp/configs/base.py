from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Union

CACHE_DIR = Path.home() / ".cache"


@dataclass
class VideoConfig:
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
    predict: Literal["TPL"] = "TPL"  # Prediction method
    pose_distance: Literal["smpl"] = "smpl"  # Distance metric for poses
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
    MODEL_PATH: Path = Path("data/smpl/")
    GENDER: str = "neutral"
    MODEL_TYPE: str = "smpl"
    NUM_BODY_JOINTS: int = 23
    JOINT_REGRESSOR_EXTRA: Path = CACHE_DIR / "phalp/3D/SMPL_to_J19.pkl"
    TEXTURE: Path = CACHE_DIR / "phalp/3D/texture.npz"


# Config for HMAR
@dataclass
class SMPLHeadConfig:
    TYPE: str = "basic"
    POOL: str = "max"
    SMPL_MEAN_PARAMS: Path = CACHE_DIR / "phalp/3D/smpl_mean_params.npz"
    IN_CHANNELS: int = 2048


@dataclass
class BackboneConfig:
    TYPE: str = "resnet"
    NUM_LAYERS: int = 50
    MASK_TYPE: str = "feat"


@dataclass
class TransformerConfig:
    HEADS: int = 1
    LAYERS: int = 1
    BOX_FEATS: int = 6


@dataclass
class ModelConfig:
    IMAGE_SIZE: int = 256
    SMPL_HEAD: SMPLHeadConfig = field(default_factory=SMPLHeadConfig)
    BACKBONE: BackboneConfig = field(default_factory=BackboneConfig)
    TRANSFORMER: TransformerConfig = field(default_factory=TransformerConfig)
    pose_transformer_size: int = 2048


@dataclass
class ExtraConfig:
    FOCAL_LENGTH: int = 5000


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

    # Fields
    video: VideoConfig = field(default_factory=VideoConfig)
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
