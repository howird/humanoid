from pathlib import Path

import cv2
import torchvision


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


def read_from_video_pts(video_path, frame_pts):
    frame = torchvision.io.read_video(video_path, pts_unit="pts", start_pts=frame_pts, end_pts=frame_pts + 1)[0][0]
    frame = frame.numpy()[:, :, ::-1]
    return frame
