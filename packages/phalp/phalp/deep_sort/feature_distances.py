import numpy as np
import torch

from phalp.models.hmar.hmr2 import HMR2023TextureSampler


def get_uv_distance(HMAR: HMR2023TextureSampler, t_uv, d_uv):
    t_uv = torch.from_numpy(t_uv).cuda().float()
    d_uv = torch.from_numpy(d_uv).cuda().float()
    d_mask = d_uv[3:, :, :] > 0.5
    t_mask = t_uv[3:, :, :] > 0.5

    mask_dt = torch.logical_and(d_mask, t_mask)
    mask_dt = mask_dt.repeat(4, 1, 1)
    mask_ = torch.logical_not(mask_dt)

    t_uv[mask_] = 0.0
    d_uv[mask_] = 0.0

    with torch.no_grad():
        t_emb = HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
        d_emb = HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
    t_emb = t_emb.view(-1) / 10**3
    d_emb = d_emb.view(-1) / 10**3
    return t_emb.cpu().numpy(), d_emb.cpu().numpy(), torch.sum(mask_dt).cpu().numpy() / 4 / 256 / 256 / 2


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
