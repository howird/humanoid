import os
import gc
import uuid
import math
import json

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal, Union
from datetime import datetime

import joblib
import tyro
from tqdm import tqdm

import isaacgym  # noqa

import torch
import numpy as np

from smpl_sim.smpllib.smpl_eval import compute_metrics_lite

import pufferlib
import pufferlib.cleanrl
import pufferlib.vector

from puffer_phc import clean_pufferl
from puffer_phc.clean_pufferl.env import PHCPufferEnv, make as env_creator
from puffer_phc.envs.humanoid_phc import HumanoidPHC
import puffer_phc.policies as policy_module

from puffer_phc.config import EnvConfig, PolicyConfig, RNNConfig, TrainConfig


@dataclass
class DebugConfig:
    enable: bool = False
    port: int = 5678

    def __call__(self):
        if self.enable:
            import debugpy

            debugpy.listen(self.port)
            print(f"Waiting for debugger attach to port: {self.port}")
            debugpy.wait_for_client()


@dataclass
class AppConfig:
    """Application configuration"""

    policy_name: Literal["PHCPolicy", "LSTMCriticPolicy", "LSTMActorPolicy"] = "PHCPolicy"
    rnn_name: Literal[None, "Recurrent"] = None
    mode: Literal["train", "play", "eval", "sweep"] = "train"
    checkpoint_path: Optional[str] = None
    track: bool = False
    wandb_project: str = "pufferlib"
    run_name: Optional[str] = None
    skip_resample: bool = False
    final_eval: bool = False
    exp_id: str = field(init=False)

    # Configuration sections
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sweep: Dict[str, Any] = field(default_factory=dict)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self):
        self.exp_id = self.env.name + "-" + str(uuid.uuid4())[:8]


class EvalStats:
    def __init__(self, vec_env, failed_save_path=None):
        self.task_env: HumanoidPHC = vec_env.env
        self.num_envs = self.task_env.cfg.num_envs
        device = self.task_env.cfg.device
        self.failed_save_path = failed_save_path

        # Prep the env for evaluation
        self.num_unique_motions = self.task_env.toggle_eval_mode()

        self.terminate_state = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.played_steps_buf = torch.zeros(self.num_envs, dtype=torch.short, device=device)
        self.terminate_memory = []
        self.motion_length = []
        self.played_steps = []

        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_steps = 0
        self.success_rate = 0
        self.failed_keys = []
        self.results = None
        self.results_by_motion = None

        self.pbar = tqdm(range(self.num_unique_motions // self.num_envs))
        self.pbar.set_description("")

    def post_step_eval(self):
        motion_num_steps = self.task_env.get_motion_steps()
        next_batch = False

        # Eval-related info is stored in the extras
        info = self.task_env.extras

        # If terminate after the last frame, then it is not a termination. curr_step is one step behind simulation.
        termination_state = torch.logical_and(self.curr_steps < motion_num_steps, info["terminate"])
        self.terminate_state = torch.logical_or(termination_state, self.terminate_state, out=self.terminate_state)

        # Record the number of steps played
        current_envs = torch.logical_and(~self.terminate_state, self.curr_steps < motion_num_steps)
        if current_envs.any():
            self.played_steps_buf[current_envs] += 1

        if (~self.terminate_state).sum() > 0:
            # NOTE: This is to handle when there are more envs than the motions
            max_possible_id = self.num_unique_motions - 1
            curr_ids = self.task_env.current_motion_ids
            if (max_possible_id == curr_ids).sum() > 0:
                bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                if (~self.terminate_state[:bound]).sum() > 0:
                    curr_max = motion_num_steps[:bound][~self.terminate_state[:bound]].max()
                else:
                    curr_max = self.curr_steps - 1  # the ones that should be counted have terminated
                    # The remaining envs are not counted. So set all the remaining envs to terminated
                    self.terminate_state[bound:] = True
            else:
                curr_max = motion_num_steps[~self.terminate_state].max()

            if self.curr_steps >= curr_max:
                curr_max = self.curr_steps + 1  # For matching up the current steps and max steps.
        else:
            curr_max = motion_num_steps.max()

        self.mpjpe.append(info["mpjpe"])
        self.gt_pos.append(info["body_pos_gt"])
        self.pred_pos.append(info["body_pos"])
        self.curr_steps += 1

        # All motions fully played out, or all envs are terminated
        if self.curr_steps >= curr_max or self.terminate_state.sum() == self.num_envs:
            self.curr_steps = 0
            self.terminate_memory.append(self.terminate_state.cpu().numpy())
            self.motion_length.append(self.task_env.get_motion_steps().cpu().numpy())
            self.played_steps.append(self.played_steps_buf.cpu().numpy())

            self.success_rate = 1 - np.concatenate(self.terminate_memory)[: self.num_unique_motions].mean()

            # MPJPE
            all_mpjpe = torch.stack(self.mpjpe)
            # Max should be the same as the number of frames in the motion.
            assert all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == self.num_envs

            all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(motion_num_steps)]
            all_body_pos_pred = np.stack(self.pred_pos)
            all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(motion_num_steps)]
            all_body_pos_gt = np.stack(self.gt_pos)
            all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(motion_num_steps)]

            self.mpjpe_all.append(all_mpjpe)
            self.pred_pos_all += all_body_pos_pred
            self.gt_pos_all += all_body_pos_gt

            # All motions have been fully evaluated
            if self.task_env.motion_sample_start_idx + self.num_envs >= self.num_unique_motions:
                return self.get_final_stats(), next_batch

            # Move on to the next motion
            next_batch = True
            self.task_env.forward_motion_samples()
            self.terminate_state[:] = False
            self.played_steps_buf[:] = 0

            self.pbar.update(1)
            self.pbar.refresh()
            self.mpjpe, self.gt_pos, self.pred_pos = [], [], []

        update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_steps} | Start: {self.task_env.motion_sample_start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
        self.pbar.set_description(update_str)

        return False, next_batch

    def get_final_stats(self):
        self.pbar.clear()
        terminate_hist = np.concatenate(self.terminate_memory)
        succ_idxes = np.flatnonzero(~terminate_hist[: self.num_unique_motions]).tolist()

        pred_pos_all_succ = [(self.pred_pos_all[: self.num_unique_motions])[i] for i in succ_idxes]
        gt_pos_all_succ = [(self.gt_pos_all[: self.num_unique_motions])[i] for i in succ_idxes]

        pred_pos_all = self.pred_pos_all[: self.num_unique_motions]
        gt_pos_all = self.gt_pos_all[: self.num_unique_motions]

        self.failed_keys = self.task_env.motion_data_keys[terminate_hist[: self.num_unique_motions]]
        # success_keys = self.task_env.motion_data_keys[~terminate_hist[:self.num_unique_motions]]

        metrics_all = compute_metrics_lite(pred_pos_all, gt_pos_all)
        metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

        metrics_all_print = {m: float(np.mean(v)) for m, v in metrics_all.items()}
        metrics_succ_print = {m: float(np.mean(v)) for m, v in metrics_succ.items()}

        if len(metrics_succ_print) == 0:
            print("No success!!!")
            metrics_succ_print = metrics_all_print

        print("------------------------------------------")
        print(f"Success Rate: {self.success_rate:.10f}")
        print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
        print(
            "Succ: ",
            " \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]),
        )
        print("Failed keys: ", len(self.failed_keys), ",", self.failed_keys)

        self.results = {
            "eval/success_rate": float(self.success_rate),
            "eval/mpjpe_all": metrics_all_print["mpjpe_g"],
            "eval/mpjpe_succ": metrics_succ_print["mpjpe_g"],
            "eval/accel_dist": metrics_succ_print["accel_dist"],
            "eval/vel_dist": metrics_succ_print["vel_dist"],
            "eval/mpjpel_all": metrics_all_print["mpjpe_l"],
            "eval/mpjpel_succ": metrics_succ_print["mpjpe_l"],
            "eval/mpjpe_pa": metrics_succ_print["mpjpe_pa"],
        }

        self.results_by_motion = {
            "motion_keys": self.task_env.motion_data_keys.tolist(),
            "motion_length": np.concatenate(self.motion_length)[: self.num_unique_motions],
            "played_steps": np.concatenate(self.played_steps)[: self.num_unique_motions],
            "success": ~terminate_hist[: self.num_unique_motions],
        }

        return True

    def update_env_and_close(self):
        # NOTE: Assuming that resampling motion will happen right after the eval,
        # so not resetting the env here.
        termination_history = self.task_env.untoggle_eval_mode(self.failed_keys)

        torch.cuda.empty_cache()
        gc.collect()

        if self.failed_save_path:
            joblib.dump(
                {
                    "failed_keys": self.failed_keys,
                    "termination_history": termination_history,
                },
                self.failed_save_path,
            )

        return self.results


def make_policy(
    env: PHCPufferEnv, args: AppConfig
) -> Union[pufferlib.cleanrl.Policy, pufferlib.cleanrl.RecurrentPolicy]:
    """Creates pufferlib policy, checking whether to use RNN based AppConfig.r"""
    policy_cls = getattr(policy_module, args.policy_name)
    policy = policy_cls(env, **asdict(args.policy))
    if args.rnn_name:
        rnn_cls = getattr(policy_module, args.rnn_name)
        policy = rnn_cls(env, policy, **asdict(args.rnn))
        policy = pufferlib.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.cleanrl.Policy(policy)

    return policy.to(args.train.device)


def init_wandb(project_name: str, exp_id: str, env_name: str, resume=True):
    import wandb

    wandb.init(
        id=exp_id,
        project=project_name,
        allow_val_change=True,
        save_code=True,
        resume=resume,
        config=asdict(args),
        name=env_name,
    )
    return wandb


def train(
    args: AppConfig,
    vec_env: PHCPufferEnv,
    policy: Union[pufferlib.cleanrl.Policy, pufferlib.cleanrl.RecurrentPolicy],
):
    wandb = (
        init_wandb(args.wandb_project, args.exp_id, args.run_name if args.run_name else args.env.name)
        if args.track
        else None
    )

    train_config: TrainConfig = args.train

    components, state, utilization = clean_pufferl.create(
        args.exp_id, args.train, args.env, vec_env, policy, wandb=wandb
    )

    data_dir = os.path.join(train_config.data_dir, args.exp_id)
    os.makedirs(data_dir, exist_ok=True)

    while state.global_step < train_config.total_timesteps:
        if not args.skip_resample and state.epoch > 0 and state.epoch % train_config.motion_resample_interval == 0:
            # Evaluate the model every 600 epochs (train_config.checkpoint_interval)
            if state.epoch % train_config.checkpoint_interval == 0:
                eval_stats = EvalStats(
                    vec_env,
                    failed_save_path=os.path.join(data_dir, f"failed_{state.epoch:06d}.pkl"),
                )
                rollout(vec_env, policy, eval_stats)
                eval_results = eval_stats.update_env_and_close()
                if state.wandb:
                    eval_results["0verview/agent_steps"] = state.global_step
                    eval_results["0verview/epoch"] = state.epoch
                    wandb.log(eval_results)

            # Resample motions every 200 epochs (train_config.motion_resample_interval)
            vec_env.env.resample_motions()

            # Reset the envs and lstm hidden states
            vec_env.reset()
            if components.experience.lstm_c is not None and components.experience.lstm_h is not None:
                components.experience.lstm_h[:] = 0
                components.experience.lstm_c[:] = 0

        # Collect data
        results, _ = clean_pufferl.evaluate(components, state)

        # Update obs running mean and std
        # During evaluate() and train(), the obs_norm is NOT updated.
        rms_update_fn = getattr(components.policy.policy, "update_obs_rms", None)
        if rms_update_fn:
            rms_update_fn(components.experience.obs)

        amp_rms_update_fn = getattr(components.policy.policy, "update_amp_obs_rms", None)
        if state.use_amp_obs and amp_rms_update_fn:
            amp_rms_update_fn(components.experience.amp_obs)

        # Update policy
        clean_pufferl.train(components, state, utilization)

        # Apply learning rate exp decay
        if state.config.lr_decay_rate > 0:
            decay = math.exp(-state.config.lr_decay_rate * state.epoch)
            if decay < state.config.lr_decay_floor:
                decay = state.config.lr_decay_floor
            components.optimizer.param_groups[0]["lr"] = state.config.learning_rate * decay

    uptime = state.profile.uptime

    # Final evaluation
    if args.final_eval:
        eval_stats = EvalStats(vec_env)
        rollout(vec_env, policy, eval_stats)
        results.update(eval_stats.update_env_and_close())
        if state.wandb:
            results["0verview/agent_steps"] = state.global_step
            results["0verview/epoch"] = state.epoch
            wandb.log(results)

    # NOTE: Not using standard eval
    # steps_evaluated = 0
    # steps_to_eval = int(train_config.eval_timesteps)
    # batch_size = int(train_config.batch_size)
    # while steps_evaluated < steps_to_eval:
    #     stats, _ = clean_pufferl.evaluate(components, state)
    #     steps_evaluated += batch_size
    # clean_pufferl.mean_and_log(components, state)

    clean_pufferl.close(components, state, utilization)

    return results, uptime


def rollout(
    vec_env: PHCPufferEnv, policy: Union[pufferlib.cleanrl.Policy, pufferlib.cleanrl.RecurrentPolicy], eval_stats=None
):
    # NOTE (Important): Using deterministic action for evaluation
    policy.policy.set_deterministic_action(True)  # Ugly... but...

    obs, _ = vec_env.reset()
    state = None

    ep_cnt = 0
    while True:
        with torch.no_grad():
            # TODO: hardcoded device
            obs = torch.as_tensor(obs).to("cuda")
            if hasattr(policy, "lstm"):
                action, _, _, _, state = policy(obs, state)
            else:
                action, _, _, _ = policy(obs)

            action = action.cpu().numpy().reshape(vec_env.action_space.shape)

        obs, _, done, trunc, info = vec_env.step(action)

        if hasattr(policy, "lstm"):
            # Reset lstm states for the reset
            reset_envs = torch.logical_or(done, trunc)
            if reset_envs.any():
                state[0][:, reset_envs] = 0
                state[1][:, reset_envs] = 0

        # Get episode-related info here
        if len(info) > 0:
            ep_ret = info[0]["episode_return"]
            ep_len = info[0]["episode_length"]
            print(f"Episode cnt: {vec_env.episode_count - ep_cnt}, Reward: {ep_ret:.3f}, Length: {ep_len:.3f}")
            ep_cnt = vec_env.episode_count

        if eval_stats:
            is_done, next_batch = eval_stats.post_step_eval()
            if is_done:
                policy.policy.set_deterministic_action(False)
                break

            if next_batch and state is not None:
                # Reset the states
                state[0][:] = 0
                state[1][:] = 0


if __name__ == "__main__":
    # Parse arguments with tyro
    args: AppConfig = tyro.cli(AppConfig)

    if "cuda" in (args.env.device_type, args.train.device_type):
        assert torch.cuda.is_available(), "CUDA is not available"

    args.debug()

    # If play mode, adjust environment settings
    if args.mode == "play":
        args.env.num_envs = 16
        args.env.headless = False

    # Create the environment and policy
    vec_env: PHCPufferEnv = pufferlib.vector.make(env_creator, env_args=[args.env])  # type: ignore
    policy = make_policy(vec_env.driver_env, args)

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.train.device)
        policy.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint_path}")

    # Train or evaluate based on mode
    # TODO(py310): use match statement
    if args.mode == "train":
        train(args, vec_env, policy)

    elif args.mode == "play":
        # Just play and render without collecting stats
        vec_env.env.set_termination_distances(10)
        rollout(vec_env, policy)

    elif args.mode == "eval":
        import polars as pl

        eval_stats = EvalStats(vec_env)
        rollout(vec_env, policy, eval_stats)

        with open(f"eval_summary_{datetime.now().strftime('%m%d_%H%M')}.json", "w") as f:
            json.dump(eval_stats.results, f, indent=4)

        df = pl.DataFrame(eval_stats.results_by_motion)
        df.write_csv(
            f"results_by_motion_{datetime.now().strftime('%m%d_%H%M')}.tsv",
            separator="\t",
        )

        eval_stats.update_env_and_close()
