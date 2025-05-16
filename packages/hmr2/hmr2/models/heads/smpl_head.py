import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from typing import Tuple, Dict
from dataclasses import asdict

from phalp.configs.base import BaseConfig
from phalp.common.smpl_output import HMRSMPLOutput

from hmr2.utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from hmr2.models.components.pose_transformer import TransformerDecoder


def build_smpl_head(cfg):
    smpl_head_type = cfg.MODEL.SMPL_HEAD.get("TYPE", "hmr")
    if smpl_head_type == "transformer_decoder":
        return SMPLTransformerDecoderHead(cfg)
    else:
        raise ValueError("Unknown SMPL head type: {}".format(smpl_head_type))


class SMPLTransformerDecoderHead(nn.Module):
    """Cross-attention based SMPL Transformer decoder"""

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = self.cfg.MODEL.SMPL_HEAD.JOINT_REP
        self.joint_rep_dim = {"6d": 6, "aa": 3}[self.joint_rep_type]
        self.npose = self.joint_rep_dim * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.input_is_mean_shape = cfg.MODEL.SMPL_HEAD.TRANSFORMER_INPUT == "mean_shape"
        transformer_args = dict(
            num_tokens=1,
            token_dim=(self.npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args.update(asdict(cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER))
        self.transformer = TransformerDecoder(**transformer_args)
        dim = transformer_args["dim"]
        self.decpose = nn.Linear(dim, self.npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        if cfg.MODEL.SMPL_HEAD.INIT_DECODER_XAVIER:
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_body_pose = torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params["shape"].astype("float32")).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_body_pose", init_body_pose)
        self.register_buffer("init_betas", init_betas)
        self.register_buffer("init_cam", init_cam)

    def forward(self, x: torch.Tensor) -> Tuple[HMRSMPLOutput, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of the SMPL Transformer decoder head.

        This method takes a feature map from a backbone network and predicts SMPL parameters
        using an iterative error feedback (IEF) approach. The prediction is done in multiple
        iterations, where each iteration refines the previous prediction.

        Args:
            x: Input feature map from backbone of shape (B, C, H, W)
                where B is batch size, C is number of channels, H and W are spatial dimensions

        Returns:
            pred_smpl_params: HMRSMPLOutput containing:
                - global_orient: Global rotation matrices of shape (B, 1, 3, 3)
                - body_pose: Body joint rotation matrices of shape (B, 23, 3, 3)
                - betas: Shape parameters of shape (B, 10)
            pred_cam: Camera parameters of shape (B, 3)
            pred_smpl_params_list: Dictionary containing lists of SMPL parameters from each iteration:
                - body_pose: Body pose matrices of shape (IEF_ITERS, B, 23, 3, 3)
                - betas: Shape parameters of shape (IEF_ITERS, B, 10)
                - cam: Camera parameters of shape (IEF_ITERS, B, 3)
        """
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == "aa":
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for _ in range(self.cfg.MODEL.SMPL_HEAD.IEF_ITERS):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[:, None, :]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            # Pass through transformer (B, 1, 1) -> (B, 1, 1024)
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)  # (B, C=1024)

            # Readout from token_out
            pred_body_pose = self.decpose(token_out) + pred_body_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam

            pred_body_pose_list.append(pred_body_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {"6d": rot6d_to_rotmat, "aa": lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())}[
            self.joint_rep_type
        ]

        pred_smpl_params_list = {}
        pred_smpl_params_list["body_pose"] = torch.cat(
            [joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_body_pose_list], dim=0
        )
        pred_smpl_params_list["betas"] = torch.cat(pred_betas_list, dim=0)
        pred_smpl_params_list["cam"] = torch.cat(pred_cam_list, dim=0)
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(batch_size, self.cfg.SMPL.NUM_BODY_JOINTS + 1, 3, 3)

        pred_smpl_params = HMRSMPLOutput(
            global_orient=pred_body_pose[:, [0]],
            body_pose=pred_body_pose[:, 1:],
            betas=pred_betas,
        )
        return pred_smpl_params, pred_cam, pred_smpl_params_list
