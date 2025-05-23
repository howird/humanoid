from dataclasses import asdict
import torch
import numpy as np

from humanoid_vision.models.hmar.hmr import HMR2018Predictor
from humanoid_vision.utils.pylogger_phalp import get_pylogger
from humanoid_vision.configs.base import CACHE_DIR, BaseConfig
from humanoid_vision.common.hmr_output import HMROutput
from humanoid_vision.common.hmar_output import HMAROutput

from humanoid_vision.models import download_models
from humanoid_vision.models.hmr2 import HMR2

log = get_pylogger(__name__)


class HMR2023TextureSampler(HMR2018Predictor):
    """HMR2023 model with texture sampling capabilities."""

    def __init__(self, cfg: BaseConfig) -> None:
        super().__init__(cfg)
        download_models()

        self.model = HMR2.load_from_checkpoint(
            CACHE_DIR / "4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt",
            strict=False,
            cfg=cfg,
        )
        self.model.eval()

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        bmap_path = CACHE_DIR / "phalp/3D/bmap_256.npy"
        fmap_path = CACHE_DIR / "phalp/3D/fmap_256.npy"
        bmap = np.load(bmap_path)
        fmap = np.load(fmap_path)
        self.register_buffer("tex_bmap", torch.tensor(bmap, dtype=torch.float))
        self.register_buffer("tex_fmap", torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256  # self.cfg.MODEL.IMAGE_SIZE
        self.focal_length = 5000.0  # self.cfg.EXTRA.FOCAL_LENGTH

        import neural_renderer as nr

        self.neural_renderer = nr.Renderer(
            dist_coeffs=None,
            orig_size=self.img_size,
            image_size=self.img_size,
            light_intensity_ambient=1,
            light_intensity_directional=0,
            anti_aliasing=False,
        )

    def forward(self, x) -> HMAROutput:
        # x: torch.Tensor of shape (num_valid_persons, C+1=4, H=256, W=256)
        batch = {
            "img": x[:, :3, :, :],
            "mask": (x[:, 3, :, :]).clip(0, 1),
        }
        model_out: HMROutput = self.model(batch)

        def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
            # bmap:  256,256,3
            # fmap:  256,256
            # verts: B,V,3
            # faces: F,3
            valid_mask = fmap >= 0

            fmap_flat = fmap[valid_mask]  # N
            bmap_flat = bmap[valid_mask, :]  # N,3

            face_vids = faces[fmap_flat, :]  # N,3
            face_verts = verts[:, face_vids, :]  # B,N,3,3

            bs = face_verts.shape
            map_verts = torch.einsum("bnij,ni->bnj", face_verts, bmap_flat)  # B,N,3

            return map_verts, valid_mask

        pred_verts = model_out.pred_vertices + model_out.pred_cam_t.unsqueeze(1)
        device = pred_verts.device
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=device)
        map_verts, valid_mask = unproject_uvmap_to_mesh(self.tex_bmap, self.tex_fmap, pred_verts, face_tensor)  # B,N,3

        # Project map_verts to image using K,R,t
        # map_verts_view = einsum('bij,bnj->bni', R, map_verts) # R=I t=0
        focal = self.focal_length / (self.img_size / 2)
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3]  # B,N,2
        map_verts_depth = map_verts[:, :, 2]  # B,N

        # Render Depth. Annoying but we need to create this
        K = torch.eye(3, device=device)
        K[0, 0] = K[1, 1] = self.focal_length
        K[1, 2] = K[0, 2] = self.img_size / 2  # Because the neural renderer only support squared images
        K = K.unsqueeze(0)
        R = torch.eye(3, device=device).unsqueeze(0)
        t = torch.zeros(3, device=device).unsqueeze(0)
        rend_depth = self.neural_renderer(
            pred_verts,
            face_tensor[None].expand(pred_verts.shape[0], -1, -1).int(),
            # textures=texture_atlas_rgb,
            mode="depth",
            K=K,
            R=R,
            t=t,
        )

        rend_depth_at_proj = torch.nn.functional.grid_sample(
            rend_depth[:, None, :, :], map_verts_proj[:, None, :, :]
        )  # B,1,1,N
        rend_depth_at_proj = rend_depth_at_proj.squeeze(1).squeeze(1)  # B,N

        img_rgba = torch.cat([batch["img"], batch["mask"][:, None, :, :]], dim=1)  # B,4,H,W
        img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:, None, :, :])  # B,4,1,N
        img_rgba_at_proj = img_rgba_at_proj.squeeze(2)  # B,4,N

        visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4)  # B,N
        img_rgba_at_proj[:, 3, :][~visibility_mask] = 0

        # Paste image back onto square uv_image
        uv_image = torch.zeros((batch["img"].shape[0], 4, 256, 256), dtype=torch.float, device=device)
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        return HMAROutput(uv_image=uv_image, uv_vector=self.hmar_old.process_uv_image(uv_image), **asdict(model_out))
