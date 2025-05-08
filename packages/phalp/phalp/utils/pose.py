import numpy as np


def get_pose_distance(track_pose, detect_pose, pose_distance="smpl"):
    """Compute pair-wise squared l2 distances between points in `track_pose` and `detect_pose`."""
    track_pose, detect_pose = np.asarray(track_pose), np.asarray(detect_pose)

    if pose_distance == "smpl":
        # remove additional dimension used for encoding location (last 3 elements)
        track_pose = track_pose[:, :-3]
        detect_pose = detect_pose[:, :-3]

    if len(track_pose) == 0 or len(detect_pose) == 0:
        return np.zeros((len(track_pose), len(detect_pose)))
    track_pose2, detect_pose2 = np.square(track_pose).sum(axis=1), np.square(detect_pose).sum(axis=1)
    r2 = -2.0 * np.dot(track_pose, detect_pose.T) + track_pose2[:, None] + detect_pose2[None, :]
    r2 = np.clip(r2, 0.0, float(np.inf))

    return r2
