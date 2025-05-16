import os
from pathlib import Path
from typing import Optional, Tuple, Any

import cv2

from phalp.configs.base import VideoIOConfig
from phalp.utils import get_pylogger

log = get_pylogger(__name__)


class VideoWriter:
    """
    A context manager class for writing video files.

    This class handles the creation and management of video files, supporting both
    direct OpenCV writing and optional FFmpeg compression.

    Attributes:
        cfg (VideoIOConfig): Configuration for video I/O operations
        output_fps (int): Frames per second for the output video
        video (Optional[Tuple[cv2.VideoWriter, str]]): Tuple containing video writer and path
    """

    def __init__(self, cfg: VideoIOConfig, fps: int):
        """
        Initialize the VideoWriter.

        Args:
            cfg (VideoIOConfig): Configuration for video I/O operations
            fps (int): Frames per second for the output video
        """
        self.cfg = cfg
        self.output_fps = fps
        self.video: Optional[Tuple[cv2.VideoWriter, str]] = None
        self._video_path: Optional[Path] = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_video()

    def save_video(self, video_path: Path, rendered_frame: Any, frame_size: Tuple[int, int], t: int = 0) -> None:
        """
        Save a frame to the video file.

        Args:
            video_path (Path): Path where the video will be saved
            rendered_frame: The frame to be written to the video
            frame_size (Tuple[int, int]): Size of the frame (width, height)
            t (int, optional): Frame index. Defaults to 0.

        Raises:
            RuntimeError: If video writer is not initialized
        """
        if t == 0:
            self._video_path = Path(video_path)
            writer = cv2.VideoWriter(
                str(self._video_path), fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=self.output_fps, frameSize=frame_size
            )
            self.video = (writer, str(self._video_path))

        if self.video is None:
            raise RuntimeError("Video writer is not initialized. Call save_video with t=0 first.")

        self.video[0].write(rendered_frame)

    def close_video(self) -> None:
        """
        Close the video writer and optionally compress the video using FFmpeg.
        """
        if self.video is not None:
            self.video[0].release()

            if self.cfg.useffmpeg and self._video_path is not None:
                compressed_path = self._video_path.with_name(
                    f"{self._video_path.stem}_compressed{self._video_path.suffix}"
                )
                ret = os.system(f"ffmpeg -hide_banner -loglevel error -y -i {self._video_path} {compressed_path}")
                # Delete original if compression was successful
                if ret == 0:
                    self._video_path.unlink()
                    log.info(f"Compressed video saved to {compressed_path}")
                else:
                    log.warning("FFmpeg compression failed, keeping original video")

            self.video = None
            self._video_path = None
