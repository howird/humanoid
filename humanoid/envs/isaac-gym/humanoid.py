from isaacgym import gymapi
# from pydantic import BaseModel
from dataclasses import dataclass, field
import torch
from typing import Dict, List


@dataclass
class RobotCfg:
    mesh: bool = False
    replace_feet: bool = True
    rel_joint_lm: bool = False
    upright_start: bool = True
    remove_toe: bool = False
    freeze_hand: bool = False
    real_weight_porpotion_capsules: bool = True
    real_weight_porpotion_boxes: bool = True
    real_weight: bool = True
    masterfoot: bool = False
    master_range: int = 30
    big_ankle: bool = True
    box_body: bool = True
    body_params: Dict = {}
    joint_params: Dict = {}
    geom_params: Dict = {}
    actuator_params: Dict = {}
    model: str = 'smpl'
    sim: str = 'isaacgym'


@dataclass
class PlaneConfig:
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0


@dataclass
class HumanoidEnv:
    num_envs: int # = 3072
    num_obs: int # = 934

    sim_params: gymapi.SimParams
    sim_type: gymapi.SimType = gymapi.SIM_PHYSX
    control_freq_inv: int = 2
    plane_cfg: PlaneConfig = field(default_factory=PlaneConfig)

    num_states: int = 0     # TODO(howird): why?
    num_actions: int = 69   # TODO(howird): why?
    is_discrete: bool = False

    device_id: int = 0
    graphics_device_id: int = 0
    device_type: str = "cuda"

    obs_buf: torch.Tensor = field(init=False)
    states_buf: torch.Tensor = field(init=False)
    rew_buf: torch.Tensor = field(init=False)
    reset_buf: torch.Tensor = field(init=False)
    progress_buf: torch.Tensor = field(init=False)
    randomize_buf: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.sim_type, self.sim_params)

        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device_id, dtype=torch.float
        )
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device_id, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device_id, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device_id, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device_id, dtype=torch.long
        )
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device_id, dtype=torch.long
        )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_cfg.static_friction
        plane_params.dynamic_friction = self.plane_cfg.dynamic_friction
        plane_params.restitution = self.plane_cfg.restitution
        self.gym.add_ground(self.sim, plane_params)

    def step(self, action: Dict):
        pass

