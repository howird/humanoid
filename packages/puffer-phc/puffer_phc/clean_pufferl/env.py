import time
import argparse
import functools
from puffer_phc.envs.humanoid_phc import HumanoidPHC
from puffer_phc.envs.render_env import HumanoidRenderEnv
from puffer_phc.config import EnvConfig

import torch
import numpy as np

import pufferlib


def env_creator(name="puffer_phc"):
    return functools.partial(make, name)


def make(cfg):
    return PHCPufferEnv(cfg)


class PufferEnvBuffers:
    observations: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    truncations: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor

    def __init__(self, obs_buf, rew_buf, reset_buf, num_agents, single_action_space_shape, device):
        self.observations = obs_buf
        self.rewards = rew_buf
        self.terminals = torch.zeros(num_agents, dtype=torch.bool, device=device)
        self.truncations = torch.zeros_like(reset_buf)
        self.masks = torch.ones_like(reset_buf)
        self.actions = torch.zeros((num_agents, *single_action_space_shape), dtype=torch.float, device=device)


class PHCPufferEnv(pufferlib.PufferEnv, PufferEnvBuffers):
    """Wrapper of HumanoidPHC for Pufferlib.
    HACK: PufferEnvBuffers inherit is mostly just for typing purposes, may break something if PufferEnv calls `super().__init__()`
    """

    def __init__(
        self,
        cfg,
    ):
        self.render_mode = "native"
        self.cfg: EnvConfig = cfg

        if self.cfg.headless:
            self.env = HumanoidPHC(cfg)
        else:
            self.env = HumanoidRenderEnv(cfg)

        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space

        self.amp_observation_space = self.env.amp_observation_space if self.cfg.use_amp_obs else None

        # Check the buffer data types, match them to puffer
        buffers = PufferEnvBuffers(
            self.env.obs_buf,
            self.env.rew_buf,
            self.env.reset_buf,
            self.cfg.num_agents,
            self.single_action_space.shape,
            self.cfg.device,
        )

        super().__init__(buffers)

        self.episode_returns = torch.zeros(self.cfg.num_envs, dtype=torch.float32, device=self.cfg.device)
        self.episode_lengths = torch.zeros(self.cfg.num_envs, dtype=torch.int32, device=self.cfg.device)
        self.episode_count = 0
        self._infos = {
            "episode_return": [],
            "episode_length": [],
            "truncated_rate": [],
        }

        self.raw_rewards = torch.zeros(5, dtype=torch.float32, device=self.cfg.device)

    @property
    def num_agents(self):
        return self.cfg.num_envs

    def reset(self, seed=None):
        self.tick = 0
        self.env.reset()

        # self.demo = self.env.demo
        # self.state = self.env.state
        self.amp_obs = self.env.amp_obs if self.cfg.use_amp_obs else None

        # Clear the buffers
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True
        self.actions[:] = 0
        self.raw_rewards[:] = 0
        self._infos["episode_return"].clear()
        self._infos["episode_length"].clear()
        self._infos["truncated_rate"].clear()

        return self.observations, []

    def step(self, actions):
        if self.cfg.clip_actions:
            actions = np.clip(actions, -1, 1)
        self.actions[:] = actions

        # obs, reward, done are put into the buffers
        self.env.step(self.actions)

        # self.demo = self.env.demo
        # self.state = self.env.state
        self.amp_obs = self.env.amp_obs if self.cfg.use_amp_obs else None

        rew = self.rewards.clone()

        # Extract reward-related info for logging
        self.raw_rewards += self.env.extras["reward_raw"].mean(dim=0)

        # reset_buf flags the envs that are (early-) terminated or truncated.
        # Early-terminated envs are in self.env.extras["terminate"]
        # NOTE: Truncated does NOT mean all the all parts of the motion has been played out because
        # during reset, the initial frame is randomly selected, so it could start from the very end.
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True
        reset_indices = torch.nonzero(self.env.reset_buf).squeeze(-1)
        if len(reset_indices) > 0:
            self.env.reset(reset_indices)
            self.episode_count += len(reset_indices)
            self._infos["episode_return"] += self.episode_returns[reset_indices].tolist()
            self._infos["episode_length"] += self.episode_lengths[reset_indices].tolist()
            self.episode_returns[reset_indices] = 0
            self.episode_lengths[reset_indices] = 0

            # Set terminals and truncations
            term_envs = torch.nonzero(self.env.extras["terminate"]).squeeze(-1)
            self.terminals[term_envs] = True
            self._infos["truncated_rate"] += [0.0] * len(term_envs)

            trunc_envs = reset_indices[~torch.isin(reset_indices, term_envs)]
            self.truncations[trunc_envs] = True
            self._infos["truncated_rate"] += [1.0] * len(trunc_envs)

            # Mask out the truncations
            self.masks[trunc_envs] = False

            # Set rew to 0 for "terminated" envs
            # CHECK ME: Useful? Not in the original PHC
            # rew[term_envs] = 0

        self.episode_returns[~self.env.reset_buf] += self.rewards[~self.env.reset_buf]
        self.episode_lengths[~self.env.reset_buf] += 1

        # TODO: self.env.extras has infos. Extract useful info?
        info = []
        self.tick += 1
        if self.tick % self.cfg.log_interval == 0:
            info = self.mean_and_log()

            # Extract reward-related info
            reward_info = {
                "rew_body_pos": self.raw_rewards[0].item() / self.cfg.log_interval,
                "rew_body_rot": self.raw_rewards[1].item() / self.cfg.log_interval,
                "rew_lin_vel": self.raw_rewards[2].item() / self.cfg.log_interval,
                "rew_ang_vel": self.raw_rewards[3].item() / self.cfg.log_interval,
                "rew_power": self.raw_rewards[4].item() / self.cfg.log_interval,
            }

            self.raw_rewards[:] = 0

            if len(info) > 0:
                info[0].update(reward_info)
            else:
                info.append(reward_info)

        return self.observations, rew, self.terminals, self.truncations, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def mean_and_log(self):
        # if len(self._infos["episode_return"]) < self.log_interval:
        #     return []

        info = {
            "episode_return": np.mean(self._infos["episode_return"]),
            "episode_length": np.mean(self._infos["episode_length"]),
            "truncated_rate": np.mean(self._infos["truncated_rate"]),
        }
        self._infos["episode_return"].clear()
        self._infos["episode_length"].clear()
        self._infos["truncated_rate"].clear()

        return [info]

    def fetch_amp_obs_demo(self):
        return self.env.fetch_amp_obs_demo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_envs", type=int, default=32)
    parser.add_argument("-m", "--motion_file", type=str, default="sample_data/cmu_mocap_05_06.pkl")
    parser.add_argument("--disable_self_collision", action="store_true")
    args = parser.parse_args()

    def test_perf(env, timeout=10):
        steps = 0
        start = time.time()
        env.reset()
        actions = env.action_space.sample()

        print("Starting perf test...")
        while time.time() - start < timeout:
            env.step(actions)
            steps += env.num_agents

        end = time.time()
        sps = int(steps / (end - start))
        print(f"Steps: {steps}, SPS: {sps}")

    env = PHCPufferEnv(
        name="puffer_phc",
        motion_file=args.motion_file,
        has_self_collision=not args.disable_self_collision,
        num_envs=args.num_envs,
    )
    test_perf(env)
