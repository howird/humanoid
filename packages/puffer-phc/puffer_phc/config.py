from dataclasses import dataclass, field
from typing import Optional, Literal

from tyro.conf import Suppress, Fixed

from puffer_phc.envs.state_init import StateInit


@dataclass
class DeviceConfig:
    device_type: Literal["cpu", "cuda"] = "cuda"
    device_id: int = 0

    @property
    def device(self) -> str:
        """
        NOTE: dataclass will not be aware of this property.
        To do so, add: `device: str = field(init=False)` to the child dataclass
        """
        return "cpu" if self.device_type == "cpu" else f"cuda:{self.device_id}"


@dataclass
class RewardConfig:
    k_pos: float = 100.0
    k_rot: float = 10.0
    k_vel: float = 0.1
    k_ang_vel: float = 0.1
    w_pos: float = 0.5
    w_rot: float = 0.3
    w_vel: float = 0.1
    w_ang_vel: float = 0.1
    # body pos reward, body rot reward, body vel reward, body ang vel reward
    imitation_reward_dim: int = 4
    full_body_reward: bool = True
    use_power_reward: bool = True


@dataclass
class RobotConfig:
    # TODO: support others
    humanoid_type: Literal["smpl"] = "smpl"
    has_self_collision: bool = True
    reduce_action: bool = False
    freeze_hand: bool = True
    freeze_toe: bool = True
    bias_offset: bool = False
    has_smpl_pd_offset: bool = False

    # the following are different from original PHC
    has_upright_start: bool = True
    has_dof_subset: bool = True
    has_mesh: bool = False

    # The below configs have the default value #####
    # NOTE: These are used in the obs/reward compuation. Revisit later.
    has_shape_obs: bool = False
    has_shape_obs_disc: bool = False
    has_limb_weight_obs: bool = False
    has_limb_weight_obs_disc: bool = False

    # NOTE: To customize SMPL, see below links
    # https://github.com/ZhengyiLuo/PHC/blob/master/phc/env/tasks/humanoid.py#L270
    # https://github.com/ZhengyiLuo/PHC/blob/master/phc/env/tasks/humanoid.py#L782

    # reduce_action, _freeze_hand, _freeze_toe are used in self.pre_physics_step()
    reduce_action: bool = False
    freeze_hand: bool = True
    freeze_toe: bool = True
    reduced_action_idx = (0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 42, 43, 44, 47, 48, 49, 50, 57, 58, 59, 62, 63, 64, 65)  # fmt: skip

    # See self._build_pd_action_offset_scale()
    bias_offset: bool = False
    has_smpl_pd_offset: bool = False


@dataclass
class EnvConfig(DeviceConfig):
    """Environment configuration"""

    name: str = "humanoid_phc"
    motion_file: str = "data/motion/amass_train_take6_upright.pkl"
    num_envs: int = 4096
    headless: bool = True
    exp_name: str = "puffer_phc"

    clip_actions: bool = True
    use_amp_obs: bool = False
    enable_early_termination: bool = True
    termination_distance: float = 0.25
    max_episode_length: int = 300

    auto_pmcp: bool = False
    auto_pmcp_soft: bool = True

    kp_scale: float = 1.0
    kd_scale: float = 1.0
    log_interval: int = 32

    res_action: bool = False

    rew_power_coef: float = 0.0005
    env_spacing: int = 5
    state_init: StateInit = StateInit.Random

    # NOTE: Related to inter-group collision. If False, there is no inter-env collision. See self._build_env()
    divide_group: bool = False
    # for offline RL
    collect_dataset: bool = False

    local_root_obs: bool = True
    root_height_obs: bool = True
    add_obs_noise: bool = True

    add_action_noise: bool = True
    action_noise_std: bool = True

    # TODO(howird): not used
    num_states: int = 0
    control_mode: Fixed[Literal["isaac_pd"]] = "isaac_pd"

    # Motion/AMP-related
    seq_motions: bool = False
    min_motion_len: int = 5

    # Some AMASS motion is over 7000 frames, and it substantially slows down the evaluation
    max_motion_len: int = 600
    hybrid_init_prob: float = 0.5

    num_amp_obs_steps: int = 10
    amp_root_height_obs: bool = True

    robot: Suppress[RobotConfig] = field(default_factory=RobotConfig)
    reward: Suppress[RewardConfig] = field(default_factory=RewardConfig)

    @property
    def num_agents(self) -> int:
        return self.num_envs


@dataclass
class PolicyConfig:
    """Policy configuration"""

    hidden_size: int = 512


@dataclass
class RNNConfig:
    """RNN configuration"""

    input_size: int = 512
    hidden_size: int = 512


@dataclass
class TrainConfig(DeviceConfig):
    """Training configuration"""

    seed: int = 1
    torch_deterministic: bool = True
    cpu_offload: bool = False
    compile: bool = False
    norm_adv: bool = True
    target_kl: Optional[float] = None

    total_timesteps: int = 500_000_000
    eval_timesteps: int = 1_310_000

    data_dir: str = "experiments"
    checkpoint_interval: int = 1500
    motion_resample_interval: int = 500

    num_workers: int = 1
    num_envs: int = 1
    batch_size: int = 131072
    minibatch_size: int = 32768

    learning_rate: float = 0.0001
    anneal_lr: bool = False
    lr_decay_rate: float = 1.5e-4
    lr_decay_floor: float = 0.2

    update_epochs: int = 4
    bptt_horizon: int = 8
    gae_lambda: float = 0.2
    gamma: float = 0.98
    clip_coef: float = 0.01
    vf_coef: float = 1.2
    clip_vloss: bool = True
    vf_clip_coef: float = 0.2
    max_grad_norm: float = 10.0
    ent_coef: float = 0.0
    disc_coef: float = 5.0
    bound_coef: float = 10.0
    l2_reg_coef: float = 0.0

    # 'registers' inherited DeviceConfig.device with dataclasses
    device: str = field(init=False)  # type: ignore
