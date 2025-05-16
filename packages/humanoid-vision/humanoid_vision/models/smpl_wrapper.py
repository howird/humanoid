import torch
import pickle

from dataclasses import asdict
from typing import Optional

import smplx
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput

from humanoid_vision.common.smpl_output import HMRSMPLOutput


class SMPL(smplx.SMPLLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, update_hips: bool = False, **kwargs):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as SMPLLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(SMPL, self).__init__(*args, **kwargs)
        smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

        if joint_regressor_extra is not None:
            self.register_buffer(
                "joint_regressor_extra",
                torch.tensor(pickle.load(open(joint_regressor_extra, "rb"), encoding="latin1"), dtype=torch.float32),
            )
        self.register_buffer("joint_map", torch.tensor(smpl_to_openpose, dtype=torch.long))
        self.update_hips = update_hips

    def forward(self, hmr_smpl_output: HMRSMPLOutput, *args, **kwargs) -> SMPLOutput:
        """
        Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smpl_output = super(SMPL, self).forward(pose2rot=False, *args, **asdict(hmr_smpl_output), **kwargs)
        assert smpl_output.joints is not None, "SMPLOutput does not contain joints output"
        assert smpl_output.vertices is not None, "SMPLOutput does not contain vertices output"

        joints = smpl_output.joints[:, self.joint_map, :]

        if self.update_hips:
            joints[:, [9, 12]] = (
                joints[:, [9, 12]]
                + 0.25 * (joints[:, [9, 12]] - joints[:, [12, 9]])
                + 0.5 * (joints[:, [8]] - 0.5 * (joints[:, [9, 12]] + joints[:, [12, 9]]))
            )

        if hasattr(self, "joint_regressor_extra"):
            extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        smpl_output.joints = joints  # type: ignore - smplx aliases the torch.Tensor type
        return smpl_output
