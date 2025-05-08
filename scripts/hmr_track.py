"""
HMR Tracking Script - Tracks humans in video and extracts 3D mesh information.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import tyro

from puffer_phc.config import DebugConfig

from phalp.configs.base import BaseConfig
from phalp.utils.io_manager import IOManager
from phalp.trackers.hmr_phalp import HMR2PHALP
from phalp.utils import get_pylogger


warnings.filterwarnings("ignore")

log = get_pylogger(__name__)


@dataclass
class Human4DTrackingConfig(BaseConfig):
    """Configuration for Human4D tracking."""

    video_source: tyro.conf.Positional[str] = tyro.MISSING

    # Target aspect ratio for bounding boxes (width, height)
    expand_bbox_shape: Optional[Tuple[int, int]] = (192, 256)
    debug: DebugConfig = field(default_factory=DebugConfig)


def main():
    cfg = tyro.cli(Human4DTrackingConfig)
    cfg.debug()

    io_mgr = IOManager(cfg.video, cfg.render.fps)
    video_name, list_of_frames = io_mgr.get_frames_from_source(cfg.video_source)

    phalp_tracker = HMR2PHALP(cfg)
    phalp_tracker.track(video_name, list_of_frames)


if __name__ == "__main__":
    main()
