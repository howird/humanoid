from dataclasses import dataclass, field

import torch

from humanoid_vision.common.hmr_output import HMROutput


@dataclass
class HMAROutput(HMROutput):
    """Output of the HMAR2 model.

    Inherits from HMRSMPLOutput and adds additional fields for appearance.

    Attributes:
        pred_cam: Camera parameters of shape (B, 3)
        pred_cam_t: Camera translation of shape (B, 3)
        focal_length: Focal length of shape (B, 2)
        pred_keypoints_3d: 3D keypoints of shape (B, N, 3)
        pred_keypoints_2d: 2D keypoints of shape (B, N, 2)
        pred_vertices: Mesh vertices of shape (B, V, 3)
        uv_image: TODO(howird) (B, 4, 256, 256)
        uv_vector: TODO(howird) (B, 4, 256, 256)
        losses: Optional dictionary of losses
    """

    uv_image: torch.Tensor = field(default_factory=lambda: ValueError("must provide uv_image"))
    uv_vector: torch.Tensor = field(default_factory=lambda: ValueError("must provide uv_vector"))
