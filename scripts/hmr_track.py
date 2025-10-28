"""
HMR Tracking Script - Tracks humans in video and extracts 3D mesh information.
"""

import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from pathlib import Path

import tyro
import cv2
import joblib

from puffer_phc.config import DebugConfig

from humanoid_vision.utils.video_writer import VideoWriter
from humanoid_vision.visualize.visualizer import Visualizer
from humanoid_vision.configs.base import BaseConfig
from humanoid_vision.utils.video_io_manager import VideoIOManager
from humanoid_vision.models.hmar.hmr2 import HMR2023TextureSampler
from humanoid_vision.trackers.phalp import PHALP
from humanoid_vision.trackers import setup_predictor, setup_detectron2


@dataclass
class Human4DTrackingConfig(BaseConfig):
    """Configuration for Human4D tracking."""

    video_source: tyro.conf.Positional[str] = tyro.MISSING
    debug: DebugConfig = field(default_factory=DebugConfig)


def main():
    cfg = tyro.cli(Human4DTrackingConfig)
    cfg.debug()

    # set up models
    hmr_model = HMR2023TextureSampler(cfg)
    pose_predictor = setup_predictor(cfg, hmr_model.smpl)
    detector = setup_detectron2(cfg)
    phalp_tracker = PHALP(cfg, hmr_model, pose_predictor, detector)

    # do tracking
    with VideoIOManager(cfg.video_io, cfg.render.fps) as video_io:
        video_name, list_of_frames = video_io.get_frames_from_source(cfg.video_source)
        tracklets_data = phalp_tracker.track(video_name, list_of_frames)

    # save data to joblib pickle
    pkl_path = cfg.video_io.output_dir / f"{cfg.track_dataset}_{video_name}.pkl"
    if not (cfg.overwrite) and pkl_path.is_file():
        raise ValueError(f"{pkl_path} already exists, video has likely already been processed.")
    joblib.dump(tracklets_data, pkl_path, compress=3)

    # do rendering
    if cfg.render.enable:
        video_path = cfg.video_io.output_dir / f"{cfg.base_tracker}_{video_name}.mp4"
        visualizer = Visualizer(cfg.render, hmr_model.smpl, cfg.EXTRA.FOCAL_LENGTH, cfg.SMPL.TEXTURE)
        visualizer.reset_render(cfg.render.res * cfg.render.up_scale)
        with VideoWriter(cfg.video_io, cfg.render.fps) as vwriter:
            for i, frame_key in enumerate(sorted(tracklets_data.keys())):
                frame_path = tracklets_data[frame_key]["frame_path"]
                image_frame = cv2.imread(str(frame_path))
                rendered, f_size = visualizer.render_video(image_frame, tracklets_data[frame_key])

                # save the rendered frame
                vwriter.save_video(video_path, rendered, f_size, t=i)


if __name__ == "__main__":
    main()
