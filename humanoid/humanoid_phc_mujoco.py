import os
import sys
from enum import Enum
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from gym import spaces

from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES

from puffer_phc import ASSET_DIR
from puffer_phc.poselib_skeleton import SkeletonTree
from puffer_phc.motion_lib import MotionLibSMPL, FixHeightMode
from puffer_phc.torch_utils import (
    to_torch,
    torch_rand_float,
    exp_map_to_quat,
    calc_heading_quat,
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_mul,
    quat_conjugate,
    quat_to_tan_norm,
    quat_to_angle_axis,
)


class StateInit(Enum):
    Default = 0
    Start = 1
    Random = 2
    Hybrid = 3


class MujocoBase:
    def __init__(
        self,
        sim_params=None,
        device_type="cuda",
        device_id=0,
        headless=True,
        sim_timestep=1.0 / 60.0,
        control_freq_inv=2,
    ):
        self.device = "cuda" + ":" + str(device_id) if device_type == "cuda" else "cpu"
        self.dt = control_freq_inv * sim_timestep
        self.control_freq_inv = control_freq_inv

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "smpl_humanoid.xml"))
        self.data = mujoco.MjData(self.model)

        # Set gravity
        self.model.opt.gravity[2] = -9.81

        # Viewer and camera setup
        self.viewer = None if headless else mujoco.viewer.launch_passive(self.model, self.data)
        self.camera = mujoco.MjvCamera()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # Initialize camera position
        self.camera.azimuth = 45
        self.camera.elevation = -30
        self.camera.distance = 5.0
        self.camera.lookat[:] = [0, 0, 1.0]

    def render_to_image(self, width=1280, height=720):
        """Render the scene to an RGB array."""
        self.scene.clear()
        mujoco.mj_updateScene(
            self.model,
            self.data,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
            self.camera,
        )
        
        # Allocate rgb and depth buffers
        rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update context if dimensions have changed
        if self.context.width != width or self.context.height != height:
            self.context.free()
            self.context = mujoco.MjrContext(
                self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value
            )
            self.context.opengl_init()
            self.context.resize(width, height)
            
        viewport = mujoco.MjrRect(0, 0, width, height)
        
        # Render scene in offscreen buffer
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        mujoco.mjr_render(viewport, self.scene, self.context)
        
        # Read pixels from buffer
        mujoco.mjr_readPixels(rgb_buffer, None, viewport, self.context)
        
        return np.flipud(rgb_buffer)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def step(self, actions):
        for _ in range(self.control_freq_inv):
            mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()


class HumanoidPHCMujoco:
    def __init__(
        self,
        cfg,
        sim_params=None,
        device_type="cuda",
        device_id=0,
        headless=True,
    ):
        self.mujoco_base = MujocoBase(sim_params, device_type, device_id, headless)
        
        self.device = self.mujoco_base.device
        self.model = self.mujoco_base.model
        self.data = self.mujoco_base.data
        self.viewer = self.mujoco_base.viewer
        
        self.dt = self.mujoco_base.dt
        self.control_freq_inv = self.mujoco_base.control_freq_inv

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.cfg = cfg
        self.num_envs = cfg["env"]["num_envs"]
        self.all_env_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_file = cfg["env"]["motion_file"]

        ### Robot
        self._config_robot()
        self._create_force_sensors([])

        ### Env
        self._config_env()
        self._create_ground_plane()
        self._create_envs()
        
        self._define_gym_spaces()
        self._setup_tensors()
        self._setup_env_buffers()

        ### Flags
        self.flag_test = False
        self.flag_im_eval = False
        self.flag_debug = self.device == "cpu"

        ### Motion data
        self._load_motion(self.motion_file)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = self.all_env_ids

        self._reset_envs(env_ids)
        return self.obs_buf

    def step(self, actions):
        self._pre_physics_step(actions)
        
        for _ in range(self.control_freq_inv):
            mujoco.mj_step(self.model, self.data)
            
        self._post_physics_step()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def render(self):
        self.mujoco_base.render()

    def close(self):
        self.mujoco_base.close()

    # Additional methods would go here, adapted from humanoid_phc.py
    # The core logic remains similar but implementation details change
    # for MuJoCo vs IsaacGym specifics
