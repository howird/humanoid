import os
import shutil
import itertools

from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import joblib
import torchvision
from pytube import YouTube

from phalp.utils import get_pylogger
from phalp.configs.base import VideoConfig
from phalp.utils.frame_extractor import FrameExtractor

log = get_pylogger(__name__)


class IOManager:
    """
    Class used for loading and saving videos.
    """

    def __init__(self, cfg: VideoConfig, fps: int):
        self.cfg = cfg
        self.output_fps = fps
        self.video = None
        self.frames_dir = None

    def get_frames_from_source(self) -> Tuple[str, Sequence[Path], Dict]:
        # {key: frame name, value: {"gt_bbox": None, "extra data": None}}
        additional_data = {}

        # check for youtube video, str implies url, see `VideoConfig.__post_init__`
        if isinstance(self.cfg.source, str):
            video_name = self.cfg.source[-11:]  # youtube video id

            youtube_dir = self.cfg.output_dir / "youtube_downloads"
            youtube_dir.mkdir(parents=True, exist_ok=True)

            yt_video_filename = f"{video_name}.mp4"
            self.cfg.source = youtube_dir / yt_video_filename

            youtube_video = YouTube(str(self.cfg.source))
            youtube_video.streams.get_highest_resolution().download(
                output_path=str(youtube_dir), filename=yt_video_filename
            )

        if self.cfg.source.is_file() and self.cfg.source.suffix == ".mp4":
            video_name = self.cfg.source.stem
            self.frames_dir = self.cfg.source.parent / video_name

            if self.cfg.extract_video:
                if self.frames_dir.is_file():
                    raise ValueError(f"Directory for frames, {self.frames_dir}, is a file.")

                self.frames_dir.mkdir(parents=True, exist_ok=True)

                if not any(self.frames_dir.glob("*.jpg")):
                    fe = FrameExtractor(str(self.cfg.source))
                    log.info(f"Extracting video at {self.cfg.source} to {self.frames_dir}")
                    log.info(f"Number of frames: {fe.n_frames}")
                    list_of_frames = fe.extract_frames(
                        dest_path=self.frames_dir,
                        every_x_frame=1,
                        start_frame=self.cfg.start_frame,
                        end_frame=self.cfg.end_frame,
                    )
                else:
                    list_of_frames = sorted(self.frames_dir.glob("*.jpg"))
                    log.warning(
                        f"Found {len(list_of_frames)} frames for video at {self.cfg.source} in {self.frames_dir}"
                    )
            else:
                raise NotImplementedError("TODO(howird)")
                start_time, end_time = int(self.cfg.start_time[:-1]), int(self.cfg.end_time[:-1])
                try:
                    # TODO: check if torchvision is compiled from source
                    raise Exception("torchvision error")
                    # https://github.com/pytorch/vision/issues/3188
                    reader = torchvision.io.VideoReader(self.cfg.source, "video")
                    list_of_frames = []
                    for frame in itertools.takewhile(lambda x: x["pts"] <= end_time, reader.seek(start_time)):
                        list_of_frames.append(frame["data"])
                except:
                    log.warning("torchvision is NOT compliled from source!!!")

                    stamps_PTS = torchvision.io.read_video_timestamps(str(self.cfg.source), pts_unit="pts")
                    stamps_SEC = torchvision.io.read_video_timestamps(str(self.cfg.source), pts_unit="sec")

                    index_start = min(range(len(stamps_SEC[0])), key=lambda i: abs(stamps_SEC[0][i] - start_time))
                    index_end = min(range(len(stamps_SEC[0])), key=lambda i: abs(stamps_SEC[0][i] - end_time))

                    if index_start == index_end and index_start == 0:
                        index_end += 1
                    elif index_start == index_end and index_start == len(stamps_SEC[0]) - 1:
                        index_start -= 1

                    # Extract the corresponding presentation timestamps from stamps_PTS
                    list_of_frames = [(self.cfg.source, i) for i in stamps_PTS[0][index_start:index_end]]

        # read from image folder
        elif self.cfg.source.is_dir():
            video_name = self.cfg.source.name
            list_of_frames = sorted(self.cfg.source.glob("*.jpg"))

        # pkl files are used to track ground truth videos with given bounding box
        # these gt_id, gt_bbox will be stored in additional_data, ground truth bbox should be in the format of [x1, y1, w, h]
        elif self.cfg.source.is_file() and self.cfg.source.suffix == ".pkl":
            gt_data = joblib.load(self.cfg.source)
            video_name = self.cfg.source.stem
            list_of_frames = [self.cfg.base_path / key for key in sorted(list(gt_data.keys()))]

            # for adding gt bbox for detection
            # the main key is the bbox, rest (class label, track id) are in extra data.
            for frame_name in list_of_frames:
                frame_id = frame_name.split("/")[-1]
                if len(gt_data[frame_id]["gt_bbox"]) > 0:
                    additional_data[frame_name] = gt_data[frame_id]
                    """
                    gt_data structure:
                    gt_data[frame_id] = {
                                            "gt_bbox": gt_boxes,
                                            "extra_data": {
                                                "gt_class": [],
                                                "gt_track_id": [],
                                            }
                                        }
                    """
        else:
            raise Exception("Invalid source path")

        return video_name, list_of_frames, additional_data

    @staticmethod
    def read_frame(frame_path):
        frame = None
        # frame path can be either a path to an image or a list of [video_path, frame_id in pts]
        if isinstance(frame_path, tuple):
            frame = torchvision.io.read_video(
                frame_path[0], pts_unit="pts", start_pts=frame_path[1], end_pts=frame_path[1] + 1
            )[0][0]
            frame = frame.numpy()[:, :, ::-1]
        elif isinstance(frame_path, Path):
            frame = cv2.imread(str(frame_path))
        else:
            raise Exception("Invalid frame path")

        return frame

    @staticmethod
    def read_from_video_pts(video_path, frame_pts):
        frame = torchvision.io.read_video(video_path, pts_unit="pts", start_pts=frame_pts, end_pts=frame_pts + 1)[0][0]
        frame = frame.numpy()[:, :, ::-1]
        return frame

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
        if self.cfg.delete_frame_dir and self.frames_dir is not None:
            shutil.rmtree(self.frames_dir)
