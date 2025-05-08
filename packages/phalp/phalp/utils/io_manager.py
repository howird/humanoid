import math
import shutil
import datetime
import itertools

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import torchvision

from pytube import YouTube

from phalp.configs.base import VideoConfig
from phalp.utils import get_pylogger

log = get_pylogger(__name__)


# TODO(howird): this code is terrible
class FrameExtractor:
    """
    Class used for extracting frames from a video file.
    """

    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        duration = self.n_frames / self.fps
        print(f"Duration: {datetime.timedelta(seconds=duration)}")

    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f"Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.")

    def extract_frames(
        self,
        dest_path: Path,
        every_x_frame: int = 1,
        img_name: str = "frame",
        img_ext: str = ".jpg",
        start_frame: int = 0,
        end_frame: int = 2000,
    ) -> List[Path]:
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        frame_cnt = 0
        img_cnt = 0
        while self.vid_cap.isOpened():
            success, image = self.vid_cap.read()
            if not success:
                break
            if (
                frame_cnt % every_x_frame == 0
                and frame_cnt >= start_frame
                and (frame_cnt < end_frame or end_frame == -1)
            ):
                img_path = dest_path / f"{img_name}{img_cnt:06}{img_ext}"
                cv2.imwrite(str(img_path), image)
                img_cnt += 1
            frame_cnt += 1
        self.vid_cap.release()
        cv2.destroyAllWindows()

        list_of_frames = sorted(dest_path.glob("*.jpg"))

        num_imgs = len(list_of_frames)
        if num_imgs != img_cnt:
            log.warning(f"Number of frames should have been {img_cnt} but {num_imgs} found.")
        if num_imgs < (max_frames := end_frame - start_frame - 1):
            log.warning(f"Number of frames should have been {max_frames} but {num_imgs} found.")

        return list_of_frames


class IOManager:
    """
    Class used for loading and saving videos.
    """

    def __init__(self, cfg: VideoConfig, fps: int):
        self.cfg = cfg
        self.output_fps = fps
        self.frames_dir = None

    def get_frames_from_source(self, source: str) -> Tuple[str, List[Path]]:
        if source.startswith("https://") or source.startswith("http://"):
            video_name = source[-11:]  # youtube video id

            youtube_dir = self.cfg.output_dir / "youtube_downloads"
            youtube_dir.mkdir(parents=True, exist_ok=True)

            yt_video_filename = f"{video_name}.mp4"
            video_path = youtube_dir / yt_video_filename

            youtube_video = YouTube(str(video_path))
            youtube_video.streams.get_highest_resolution().download(
                output_path=str(youtube_dir), filename=yt_video_filename
            )
        else:
            video_path = Path(source)

        if not video_path.exists():
            raise ValueError(f"Input video, {video_path}, does not exist.")

        if video_path.is_file() and video_path.suffix == ".mp4":
            video_name = video_path.stem
            self.frames_dir = video_path.parent / video_name

            if self.cfg.extract_video:
                if self.frames_dir.is_file():
                    raise ValueError(f"Directory for frames, {self.frames_dir}, is a file.")

                self.frames_dir.mkdir(parents=True, exist_ok=True)

                if not any(self.frames_dir.glob("*.jpg")):
                    fe = FrameExtractor(str(video_path))
                    log.info(f"Extracting video at {video_path} to {self.frames_dir}")
                    log.info(f"Number of frames: {fe.n_frames}")
                    list_of_frames = fe.extract_frames(
                        dest_path=self.frames_dir,
                        every_x_frame=1,
                        start_frame=self.cfg.start_frame,
                        end_frame=self.cfg.end_frame,
                    )
                else:
                    list_of_frames = sorted(self.frames_dir.glob("*.jpg"))
                    log.warning(f"Found {len(list_of_frames)} frames for video at {video_path} in {self.frames_dir}")
            else:
                raise NotImplementedError("TODO(howird)")
                start_time, end_time = int(self.cfg.start_time[:-1]), int(self.cfg.end_time[:-1])
                try:
                    # TODO: check if torchvision is compiled from source
                    raise Exception("torchvision error")
                    # https://github.com/pytorch/vision/issues/3188
                    reader = torchvision.io.VideoReader(video_path, "video")
                    list_of_frames = []
                    for frame in itertools.takewhile(lambda x: x["pts"] <= end_time, reader.seek(start_time)):
                        list_of_frames.append(frame["data"])
                except:
                    log.warning("torchvision is NOT compliled from source!!!")

                    stamps_PTS = torchvision.io.read_video_timestamps(str(video_path), pts_unit="pts")
                    stamps_SEC = torchvision.io.read_video_timestamps(str(video_path), pts_unit="sec")

                    index_start = min(range(len(stamps_SEC[0])), key=lambda i: abs(stamps_SEC[0][i] - start_time))
                    index_end = min(range(len(stamps_SEC[0])), key=lambda i: abs(stamps_SEC[0][i] - end_time))

                    if index_start == index_end and index_start == 0:
                        index_end += 1
                    elif index_start == index_end and index_start == len(stamps_SEC[0]) - 1:
                        index_start -= 1

                    # Extract the corresponding presentation timestamps from stamps_PTS
                    list_of_frames = [(video_path, i) for i in stamps_PTS[0][index_start:index_end]]

        # read from image folder
        elif video_path.is_dir():
            video_name = video_path.name
            list_of_frames = sorted(video_path.glob("*.jpg"))

        else:
            raise Exception("Invalid source path")

        return video_name, list_of_frames

    def get_frames_and_gt_from_source(self, source: Path) -> Tuple[str, List[Path], Dict]:
        # {key: frame name, value: {"gt_bbox": None, "extra data": None}}
        additional_data = {}

        # pkl files are used to track ground truth videos with given bounding box
        # these gt_id, gt_bbox will be stored in additional_data, ground truth bbox should be in the format of [x1, y1, w, h]
        video_path = Path(source)

        if video_path.is_file() and video_path.suffix == ".pkl":
            gt_data = joblib.load(video_path)
            video_name = video_path.stem
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

    def close_video(self):
        if self.cfg.delete_frame_dir and self.frames_dir is not None:
            shutil.rmtree(self.frames_dir)
