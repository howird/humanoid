import os

import cv2

from phalp.configs.base import VideoConfig
from phalp.utils import get_pylogger

log = get_pylogger(__name__)


# TODO(howird): this code is terrible
class VideoWriter:
    """
    Class used for loading and saving videos.
    """

    def __init__(self, cfg: VideoConfig, fps: int):
        self.cfg = cfg
        self.output_fps = fps
        self.video = None

    def reset(self):
        self.video = None

    def save_video(self, video_path, rendered_, f_size, t=0):
        if t == 0:
            video_path = str(video_path)
            self.video = {
                "video": cv2.VideoWriter(
                    video_path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=self.output_fps, frameSize=f_size
                ),
                "path": video_path,
            }

        if self.video is None:
            raise Exception("Video is not initialized")

        self.video["video"].write(rendered_)

    def close_video(self):
        if self.video is not None:
            self.video["video"].release()
            if self.cfg.useffmpeg:
                ret = os.system(
                    "ffmpeg -hide_banner -loglevel error -y -i {} {}".format(
                        self.video["path"], self.video["path"].replace(".mp4", "_compressed.mp4")
                    )
                )
                # Delete if successful
                if ret == 0:
                    os.system("rm {}".format(self.video["path"]))
            self.video = None
