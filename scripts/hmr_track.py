"""
HMR Tracking Script - Tracks humans in video and extracts 3D mesh information.
"""

import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass, field

import tyro

from puffer_phc.config import DebugConfig

from phalp.configs.base import BaseConfig
from phalp.utils.video_io_manager import VideoIOManager
from phalp.models.hmar.hmr2 import HMR2023TextureSampler
from phalp.trackers.phalp import PHALP
from phalp.trackers import setup_predictor, setup_detectron2


@dataclass
class Human4DTrackingConfig(BaseConfig):
    """Configuration for Human4D tracking."""

    video_source: tyro.conf.Positional[str] = tyro.MISSING
    debug: DebugConfig = field(default_factory=DebugConfig)


def main():
    cfg = tyro.cli(Human4DTrackingConfig)
    cfg.debug()

    hmr_model = HMR2023TextureSampler(cfg)
    pose_predictor = setup_predictor(cfg, hmr_model.smpl)
    detector = setup_detectron2(cfg)

    # Create PHALP instance with all required models
    phalp_tracker = PHALP(cfg, hmr_model, pose_predictor, detector)

    with VideoIOManager(cfg.video, cfg.render.fps) as video_io:
        video_name, list_of_frames = video_io.get_frames_from_source(cfg.video_source)
        phalp_tracker.track(video_name, list_of_frames)


if __name__ == "__main__":
    main()
