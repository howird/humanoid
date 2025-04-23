import time
import psutil

from threading import Thread
from collections import deque
from typing import Union, Dict, Any, List
from dataclasses import asdict, dataclass, field

import torch
import numpy as np

import pufferlib
import pufferlib.utils
import pufferlib.pytorch
import pufferlib.cleanrl

from puffer_phc.config import TrainConfig
from puffer_phc.clean_pufferl.env import PHCPufferEnv

torch.set_float32_matmul_precision("high")


class Experience:
    """Flat tensor storage and array views for faster indexing"""

    def __init__(
        self,
        batch_size,
        bptt_horizon,
        minibatch_size,
        num_minibatches,
        minibatch_rows,
        obs_shape,
        obs_dtype,
        atn_shape,
        atn_dtype,
        cpu_offload,
        device,
        lstm,
        lstm_total_agents,
        use_amp_obs,
        amp_obs_size=1960,
        amp_obs_update_prob=0.01,
    ):
        if minibatch_size is None:
            minibatch_size = batch_size

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        atn_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_dtype]
        pin = device == "cuda" and cpu_offload
        self.obs = torch.zeros(
            batch_size, *obs_shape, dtype=obs_dtype, pin_memory=pin, device=device if not pin else "cpu"
        )

        self.actions = torch.zeros(batch_size, *atn_shape, dtype=atn_dtype, pin_memory=pin)
        self.logprobs = torch.zeros(batch_size, pin_memory=pin)
        self.rewards = torch.zeros(batch_size, pin_memory=pin)
        self.dones = torch.zeros(batch_size, pin_memory=pin)
        self.truncateds = torch.zeros(batch_size, pin_memory=pin)
        self.values = torch.zeros(batch_size, pin_memory=pin)

        # self.obs_np = np.asarray(self.obs)
        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)
        self.values_np = np.asarray(self.values)

        self.lstm_h = self.lstm_c = None
        if lstm is not None:
            assert lstm_total_agents > 0
            shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
            self.lstm_h = torch.zeros(shape).to(device)
            self.lstm_c = torch.zeros(shape).to(device)

        self.num_minibatches = num_minibatches
        self.minibatch_rows = minibatch_rows
        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.minibatch_size = minibatch_size
        self.device = device

        self.sort_keys = []
        self.ptr = 0
        self.step = 0

        self.use_amp_obs = use_amp_obs
        if self.use_amp_obs:
            self.amp_obs = torch.zeros(
                batch_size, amp_obs_size, dtype=obs_dtype, pin_memory=pin, device=device if not pin else "cpu"
            )
            self.amp_obs_replay = torch.zeros(
                batch_size, amp_obs_size, dtype=obs_dtype, pin_memory=pin, device=device if not pin else "cpu"
            )
            # self.demo=torch.zeros(batch_size, 358, dtype=obs_dtype,
            #     pin_memory=pin, device=device if not pin else 'cpu')
            # self.info=torch.zeros(batch_size, 358, dtype=obs_dtype,
            #     pin_memory=pin, device=device if not pin else 'cpu')

            self.amp_obs_replay_filled = False
            self.amp_obs_update_prob = amp_obs_update_prob

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(self, obs, amp_obs, value, action, logprob, reward, done, trunc, env_id, mask):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = torch.where(mask)[0].numpy()[: self.batch_size - ptr]
        end = ptr + len(indices)

        self.obs[ptr:end] = obs.to(self.obs.device)[indices]
        if self.use_amp_obs:
            self.amp_obs[ptr:end] = amp_obs.to(self.amp_obs.device)[indices]

        self.values_np[ptr:end] = value.cpu().numpy()[indices]
        self.actions_np[ptr:end] = action[indices]
        self.logprobs_np[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_np[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_np[ptr:end] = done.cpu().numpy()[indices]
        self.truncateds_np[ptr:end] = trunc.cpu().numpy()[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = np.asarray(sorted(range(len(self.sort_keys)), key=self.sort_keys.__getitem__))
        self.b_idxs_obs = (
            torch.as_tensor(
                idxs.reshape(self.minibatch_rows, self.num_minibatches, self.bptt_horizon).transpose(1, 0, -1)
            )
            .to(self.obs.device)
            .long()
        )

        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(self.num_minibatches, self.minibatch_size)

        self.sort_keys = []
        return idxs

    def flatten_batch(self):
        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat

        self.b_obs = self.obs[self.b_idxs_obs]

        self.b_actions = self.actions.to(self.device, non_blocking=True)
        self.b_actions = self.b_actions[b_idxs].contiguous()

        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_logprobs = self.b_logprobs[b_idxs]

        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_dones = self.b_dones[b_idxs]

        self.b_truncated = self.truncateds.to(self.device, non_blocking=True)
        self.b_truncated = self.b_truncated[b_idxs]

        self.b_values = self.values.to(self.device, non_blocking=True)
        self.b_values = self.b_values[b_flat]

        # AMP, only used for discriminator training
        if self.use_amp_obs:
            self.b_amp_obs = self.amp_obs[b_flat]

            # Update the amp obs replay
            if not self.amp_obs_replay_filled:
                self.amp_obs_replay[:] = self.amp_obs[:]
                self.amp_obs_replay_filled = True
            else:
                # Only update the fraction of the replay buffer
                update_idx = torch.rand(self.batch_size) < self.amp_obs_update_prob
                self.amp_obs_replay[update_idx] = self.amp_obs[update_idx]

            # For the replay, the order does not matter
            rep_idx = torch.randperm(self.batch_size).reshape(self.num_minibatches, self.minibatch_size)
            self.b_amp_obs_replay = self.amp_obs_replay[rep_idx]


@dataclass
class LossComponents:
    policy_loss: float = 0.0
    value_loss: float = 0.0

    disc_loss: float = 0.0
    disc_agent_acc: float = 0.0
    disc_demo_acc: float = 0.0

    entropy: float = 0.0
    old_approx_kl: float = 0.0
    approx_kl: float = 0.0
    clipfrac: float = 0.0
    explained_variance: float = 0.0
    mean_bound_loss: float = 0.0
    before_clip_grad_norm: float = 0.0
    # after_clip_grad_norm: float = 0.0
    l2_init_reg_loss: float = 0.0


@dataclass
class StatsData:
    """Holds environment statistics for logging"""

    # Common environment metrics
    episode_length: List[float] = field(default_factory=list)
    episode_return: List[float] = field(default_factory=list)
    episode_reward: List[float] = field(default_factory=list)

    # Task-specific metrics
    fall_penalty: List[float] = field(default_factory=list)
    tracking_lin_vel_reward: List[float] = field(default_factory=list)
    tracking_ang_vel_reward: List[float] = field(default_factory=list)
    motion_reward: List[float] = field(default_factory=list)

    # Termination metrics
    truncated_rate: List[float] = field(default_factory=list)
    termination_height: List[float] = field(default_factory=list)
    termination_orientation: List[float] = field(default_factory=list)
    termination_foot_contact: List[float] = field(default_factory=list)
    termination_foot_slip: List[float] = field(default_factory=list)
    termination_joint_limit: List[float] = field(default_factory=list)
    termination_limb_velocity: List[float] = field(default_factory=list)

    # Reward Metrics
    rew_body_pos: List[float] = field(default_factory=list)
    rew_body_rot: List[float] = field(default_factory=list)
    rew_lin_vel: List[float] = field(default_factory=list)
    rew_ang_vel: List[float] = field(default_factory=list)
    rew_power: List[float] = field(default_factory=list)

    # Tracking metrics
    tracking_error: List[float] = field(default_factory=list)
    tracking_success_rate: List[float] = field(default_factory=list)

    # Velocity metrics
    lin_vel_x: List[float] = field(default_factory=list)
    lin_vel_y: List[float] = field(default_factory=list)
    lin_vel_z: List[float] = field(default_factory=list)
    ang_vel_x: List[float] = field(default_factory=list)
    ang_vel_y: List[float] = field(default_factory=list)
    ang_vel_z: List[float] = field(default_factory=list)

    # Pose metrics
    pose_error: List[float] = field(default_factory=list)
    root_height: List[float] = field(default_factory=list)

    # Media items (for wandb)
    media_items: Dict[str, Any] = field(default_factory=dict)

    def add(self, key: str, value: Any) -> None:
        """Add a value to the stats collection"""
        getattr(self, key).append(value)

    def extend(self, key: str, values: List[Any]) -> None:
        """Extend a list of values to the stats collection"""
        getattr(self, key).extend(values)

    def clear(self) -> None:
        """Clear all stats"""
        for field_name in self.__dataclass_fields__:
            if field_name != "media_items":
                setattr(self, field_name, [])
        self.media_items.clear()

    def keys(self) -> List[str]:
        """Get all stat keys"""
        return [f for f in self.__dataclass_fields__ if f != "media_items" and len(getattr(self, f)) > 0]

    def items(self) -> Dict[str, Any]:
        """Get all stats as mean values"""
        result = {}
        # Process all fields except media_items
        for field_name in self.__dataclass_fields__:
            if field_name != "media_items":
                values = getattr(self, field_name)
                if values:  # Only process non-empty lists
                    try:
                        result[field_name] = np.mean(values)
                    except:  # noqa
                        pass  # Skip values that can't be averaged

        # Add media items directly
        result.update(self.media_items)

        return result

    def mean_and_log(self, components: "TrainComponents", info: "TrainInfo", losses: LossComponents):
        """Calculate means and log to wandb (formerly a standalone function)"""
        if info.wandb is None:
            return

        info.last_log_time = time.time()
        info.wandb.log(
            {
                "0verview/SPS": info.profile.SPS,
                "0verview/agent_steps": info.global_step,
                "0verview/epoch": info.epoch,
                "0verview/learning_rate": components.optimizer.param_groups[0]["lr"],
                **{f"environment/{k}": v for k, v in self.items().items() if k != "media_items"},
                **self.media_items,
                **{f"losses/{k}": v for k, v in asdict(losses).items()},
                **{f"performance/{k}": v for k, v in asdict(info.profile).items()},
            }
        )


@dataclass
class TrainComponents:
    """Holds the core training components"""

    vecenv: PHCPufferEnv
    policy: Union[pufferlib.cleanrl.Policy, pufferlib.cleanrl.RecurrentPolicy]
    uncompiled_policy: Union[pufferlib.cleanrl.Policy, pufferlib.cleanrl.RecurrentPolicy]
    experience: Experience
    optimizer: torch.optim.Optimizer


@dataclass
class Profile:
    SPS: float = 0
    uptime: float = 0
    remaining: float = 0
    eval_time: float = 0
    env_time: float = 0
    eval_forward_time: float = 0
    eval_misc_time: float = 0
    train_time: float = 0
    train_forward_time: float = 0
    learn_time: float = 0
    train_misc_time: float = 0

    def __post_init__(self):
        self.start = time.time()
        self.env = pufferlib.utils.Profiler()
        self.eval_forward = pufferlib.utils.Profiler()
        self.eval_misc = pufferlib.utils.Profiler()
        self.train_forward = pufferlib.utils.Profiler()
        self.learn = pufferlib.utils.Profiler()
        self.train_misc = pufferlib.utils.Profiler()
        self.prev_steps = 0

    @property
    def epoch_time(self):
        return self.train_time + self.eval_time

    def update(self, components: TrainComponents, info: "TrainInfo", interval_s=1):
        global_step = info.global_step
        if global_step == 0:
            return True

        uptime = time.time() - self.start
        if uptime - self.uptime < interval_s:
            return False

        self.SPS = (global_step - self.prev_steps) / (uptime - self.uptime)
        self.prev_steps = global_step
        self.uptime = uptime

        self.remaining = (info.config.total_timesteps - global_step) / self.SPS
        # TODO(howird): bunch of errors here
        self.eval_time = components._timers["evaluate"].elapsed
        self.eval_forward_time = self.eval_forward.elapsed
        self.env_time = self.env.elapsed
        self.eval_misc_time = self.eval_misc.elapsed
        self.train_time = components._timers["train"].elapsed
        self.train_forward_time = self.train_forward.elapsed
        self.learn_time = self.learn.elapsed
        self.train_misc_time = self.train_misc.elapsed
        return True


@dataclass
class TrainInfo:
    """Holds the training info and counters"""

    config: TrainConfig
    exp_id: str
    env_name: str
    use_amp_obs: bool

    initial_params: Dict[str, torch.Tensor]

    msg: str
    last_log_time: float

    stats: StatsData
    profile: Profile
    wandb: Any

    global_step: int = 0
    epoch: int = 0


class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque(maxlen=maxlen)
        self.cpu_util = deque(maxlen=maxlen)
        self.gpu_util = deque(maxlen=maxlen)
        self.gpu_mem = deque(maxlen=maxlen)

        self.delay = delay
        self.stopped = False
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100 * psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100 * mem.active / mem.total)
            if torch.cuda.is_available():
                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100 * free / total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
