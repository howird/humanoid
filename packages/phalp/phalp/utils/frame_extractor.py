import datetime
import math

from typing import List
from pathlib import Path

import cv2

from phalp.utils import get_pylogger

log = get_pylogger(__name__)


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
