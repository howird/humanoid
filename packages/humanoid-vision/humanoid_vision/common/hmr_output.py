from dataclasses import dataclass
from typing import Optional, Dict

import torch

from humanoid_vision.common.smpl_output import HMRSMPLOutput


@dataclass
class HMROutput(HMRSMPLOutput):
    """Output of the HMR2 model.

    Inherits from HMRSMPLOutput and adds additional fields for camera parameters,
    keypoints, and vertices.

    Attributes:
        pred_cam: Camera parameters of shape (B, 3)
        pred_cam_t: Camera translation of shape (B, 3)
        focal_length: Focal length of shape (B, 2)
        pred_keypoints_3d: 3D keypoints of shape (B, N, 3)
        pred_keypoints_2d: 2D keypoints of shape (B, N, 2)
        pred_vertices: Mesh vertices of shape (B, V, 3)
        losses: Optional dictionary of losses
    """

    pred_cam: torch.Tensor
    pred_cam_t: torch.Tensor
    focal_length: torch.Tensor
    pred_keypoints_3d: torch.Tensor
    pred_keypoints_2d: torch.Tensor
    pred_vertices: torch.Tensor
    losses: Optional[Dict[str, torch.Tensor]] = None
