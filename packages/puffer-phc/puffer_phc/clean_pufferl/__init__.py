import os
import random
import pyximport

from collections import defaultdict
from typing import Union, Tuple, Dict, Any, List

import torch
import numpy as np

import pufferlib
import pufferlib.utils
import pufferlib.pytorch
import pufferlib.cleanrl

from puffer_phc.config import TrainConfig, EnvConfig
from puffer_phc.env_pufferl import PHCPufferEnv

from puffer_phc.clean_pufferl.structs import (
    Experience,
    TrainInfo,
    StatsData,
    LossComponents,
    TrainComponents,
    Profile,
    Utilization,
)
from puffer_phc.clean_pufferl.pretty import abbreviate, print_dashboard

# For the fast Cython GAE implementation
pyximport.install(setup_args={"include_dirs": np.get_include()})
from c_gae import compute_gae  # noqa

torch.set_float32_matmul_precision("high")


def create(
    exp_id: str,
    train_cfg: TrainConfig,
    env_cfg: EnvConfig,
    vecenv: PHCPufferEnv,
    policy: Union[pufferlib.cleanrl.Policy, pufferlib.cleanrl.RecurrentPolicy],
    optimizer=None,
    wandb=None,
) -> Tuple[TrainComponents, TrainInfo, Utilization]:
    seed_everything(train_cfg.seed, train_cfg.torch_deterministic)

    profile = Profile()
    utilization = Utilization()

    msg = f"Model Size: {abbreviate(count_params(policy))} parameters"
    print_dashboard(env_cfg.name, utilization, 0, 0, profile, LossComponents(), {}, msg, clear=True)

    vecenv.async_reset(train_cfg.seed)

    obs_shape = vecenv.single_observation_space.shape
    obs_dtype = vecenv.single_observation_space.dtype
    atn_shape = vecenv.single_action_space.shape
    atn_dtype = vecenv.single_action_space.dtype
    total_agents = vecenv.num_agents

    lstm = policy.lstm if hasattr(policy, "lstm") else None
    experience = Experience(
        train_cfg.batch_size,
        train_cfg.bptt_horizon,
        train_cfg.minibatch_size,
        obs_shape,
        obs_dtype,
        atn_shape,
        atn_dtype,
        train_cfg.cpu_offload,
        train_cfg.device,
        lstm,
        total_agents,
        env_cfg.use_amp_obs,
    )

    uncompiled_policy = policy

    if train_cfg.compile:
        policy = torch.compile(policy)

    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.learning_rate, eps=1e-5)

    # Store initial policy weights for regenerative regularization
    # https://arxiv.org/pdf/2308.11958
    initial_params = {}
    for name, param in policy.named_parameters():
        initial_params[name] = param.detach().clone()

    components = TrainComponents(
        vecenv=vecenv,
        policy=policy,
        uncompiled_policy=uncompiled_policy,
        experience=experience,
        optimizer=optimizer,
    )

    info = TrainInfo(
        config=train_cfg,
        exp_id=exp_id,
        env_name=env_cfg.name,
        stats=StatsData(),
        msg=msg,
        last_log_time=0,
        use_amp_obs=env_cfg.use_amp_obs,
        initial_params=initial_params,
        profile=profile,
        wandb=wandb,
    )

    return components, info, utilization


@pufferlib.utils.profile
def evaluate(components: TrainComponents, info: TrainInfo) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    train_cfg, profile, experience = info.config, info.profile, components.experience

    with profile.eval_misc:
        policy = components.policy
        env_infos = defaultdict(list)
        lstm_h, lstm_c = experience.lstm_h, experience.lstm_c

    while not experience.full:
        with profile.env:
            o, r, d, t, env_info, env_id, mask = components.vecenv.recv()
            env_id = env_id.tolist()

        with profile.eval_misc:
            if isinstance(mask, torch.Tensor):
                info.global_step += int(mask.sum().item())
            else:
                info.global_step += int(sum(mask))

            o = torch.as_tensor(o)
            o_device = o.to(train_cfg.device)
            r = torch.as_tensor(r)
            d = torch.as_tensor(d)
            t = torch.as_tensor(t)

        with profile.eval_forward, torch.no_grad():
            # TODO: In place-update should be faster. Leaking 7% speed max
            # Also should be using a cuda tensor to index
            if lstm_h is not None and lstm_c is not None:
                # Reset the hidden states for the done/truncated envs
                reset_envs = torch.logical_or(d, t)
                if reset_envs.any():
                    lstm_h[:, reset_envs] = 0
                    lstm_c[:, reset_envs] = 0

                h = lstm_h[:, env_id]
                c = lstm_c[:, env_id]
                actions, logprob, _, value, (h, c) = policy(o_device, (h, c))
                lstm_h[:, env_id] = h
                lstm_c[:, env_id] = c
            else:
                actions, logprob, _, value = policy(o_device)

            if train_cfg.device == "cuda":
                torch.cuda.synchronize()

        with profile.eval_misc:
            value = value.flatten()
            actions = actions.cpu().numpy()
            mask = torch.as_tensor(mask)  # * policy.mask)
            o = o if train_cfg.cpu_offload else o_device

            amp_obs = components.vecenv.amp_obs if info.use_amp_obs else None
            experience.store(o, amp_obs, value, actions, logprob, r, d, t, env_id, mask)

            for i in env_info:
                for k, v in pufferlib.utils.unroll_nested_dict(i):
                    env_infos[k].append(v)

        with profile.env:
            components.vecenv.send(actions)

    with profile.eval_misc:
        for k, v in env_infos.items():
            if "_map" in k and info.wandb is not None:
                # Handle media items differently
                info.stats.media_items[f"Media/{k}"] = info.wandb.Image(v[0])
                continue

            if isinstance(v, np.ndarray):
                v = v.tolist()
            try:
                iter(v)
            except TypeError:
                info.stats.add(k, v)
            else:
                info.stats.extend(k, v)

    # TODO: Better way to enable multiple collects
    components.experience.ptr = 0
    components.experience.step = 0
    return info.stats, env_infos


@pufferlib.utils.profile
def train(components: TrainComponents, info: TrainInfo, utilization: Utilization) -> None:
    train_cfg, profile = info.config, info.profile
    experience = components.experience
    losses = LossComponents()

    with profile.train_misc:
        idxs = experience.sort_training_data()
        dones_np = experience.dones_np[idxs]
        # trunc_np = experience.truncateds_np[idxs]
        values_np = experience.values_np[idxs]
        rewards_np = experience.rewards_np[idxs]
        experience.flatten_batch()

        if info.use_amp_obs:
            amp_obs_demo = components.vecenv.fetch_amp_obs_demo()  # [num_envs, amp_obs_size]
            amp_minibatch_size = amp_obs_demo.shape[0]

        # Mean bound loss attribute
        mean_bound_loss = getattr(components.policy.policy, "mean_bound_loss", None)

    # Compute adversarial reward. Note: discriminator doesn't get
    # updated as often this way, but GAE is more accurate
    adversarial_reward = torch.zeros(experience.num_minibatches, train_cfg.minibatch_size).to(train_cfg.device)

    discriminate = getattr(components.policy.policy, "discriminate", None)
    if info.use_amp_obs and discriminate is not None:
        with torch.no_grad():
            for mb in range(experience.num_minibatches):
                disc_logits = discriminate(experience.b_amp_obs[mb]).squeeze()
                prob = 1 / (1 + torch.exp(-disc_logits))
                adversarial_reward[mb] = -torch.log(
                    torch.maximum(1 - prob, torch.tensor(0.0001, device=train_cfg.device))
                )

    # TODO: Nans in adversarial reward and gae
    adversarial_reward_np = adversarial_reward.cpu().numpy().ravel()

    # For motion imitation, done is True ONLY when the env is terminated early.
    # Successful replay of motions will get done=False, truncation=True
    # Since gae is using only dones, the advantages for truncated steps are
    # computed as the same as the nonterminal steps.
    # NOTE: The imitation reward and adversarial reward are equally weighted.
    advantages_np = compute_gae(
        dones_np, values_np, rewards_np + adversarial_reward_np, train_cfg.gamma, train_cfg.gae_lambda
    )

    advantages = torch.as_tensor(advantages_np).to(train_cfg.device)
    experience.b_advantages = (
        advantages.reshape(experience.minibatch_rows, experience.num_minibatches, experience.bptt_horizon)
        .transpose(0, 1)
        .reshape(experience.num_minibatches, experience.minibatch_size)
    )
    experience.returns_np = advantages_np + experience.values_np
    experience.b_returns = experience.b_advantages + experience.b_values

    # Optimizing the policy and value network
    total_minibatches = experience.num_minibatches * train_cfg.update_epochs
    # mean_pg_loss, mean_v_loss, mean_entropy_loss = 0, 0, 0
    # mean_old_kl, mean_kl, mean_clipfrac = 0, 0, 0
    for _epoch in range(train_cfg.update_epochs):
        lstm_state = None
        for mb in range(experience.num_minibatches):
            with profile.train_misc:
                obs = experience.b_obs[mb].to(train_cfg.device)
                atn = experience.b_actions[mb]
                log_probs = experience.b_logprobs[mb]
                val = experience.b_values[mb]
                adv = experience.b_advantages[mb]
                ret = experience.b_returns[mb]

                if info.use_amp_obs:
                    amp_obs_agent = torch.cat(
                        [
                            experience.b_amp_obs[mb][:amp_minibatch_size],
                            experience.b_amp_obs_replay[mb][:amp_minibatch_size],
                        ]
                    )

            with profile.train_forward:
                if experience.lstm_h is not None:
                    _, newlogprob, entropy, newvalue, lstm_state = components.policy(obs, info=lstm_state, action=atn)
                    lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                else:
                    _, newlogprob, entropy, newvalue = components.policy(
                        obs.reshape(-1, *components.vecenv.single_observation_space.shape),
                        action=atn,
                    )

                if train_cfg.device == "cuda":
                    torch.cuda.synchronize()

            with profile.train_misc:
                logratio = newlogprob - log_probs.reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > train_cfg.clip_coef).float().mean()

                adv = adv.reshape(-1)
                if train_cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - train_cfg.clip_coef, 1 + train_cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if train_cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - ret) ** 2
                    v_clipped = val + torch.clamp(
                        newvalue - val,
                        -train_cfg.vf_clip_coef,
                        train_cfg.vf_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - ret) ** 2
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = ((newvalue - ret) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - train_cfg.ent_coef * entropy_loss + v_loss * train_cfg.vf_coef

                # Discriminator loss
                disc_loss = 0.0
                if info.use_amp_obs:
                    disc_agent_logits = discriminate(amp_obs_agent)
                    disc_demo_logits = discriminate(amp_obs_demo)
                    disc_loss_agent = torch.nn.BCEWithLogitsLoss()(
                        disc_agent_logits, torch.zeros_like(disc_agent_logits)
                    )
                    disc_loss_demo = torch.nn.BCEWithLogitsLoss()(disc_demo_logits, torch.ones_like(disc_demo_logits))
                    disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

                if train_cfg.disc_coef > 0:
                    loss += disc_loss * train_cfg.disc_coef

                if train_cfg.bound_coef > 0 and mean_bound_loss is not None:
                    loss += mean_bound_loss * train_cfg.bound_coef

                # Regenerative regularization, https://arxiv.org/pdf/2308.11958
                l2_init_reg_loss = 0
                for name, param in components.policy.named_parameters():
                    if name in info.initial_params:
                        l2_init_reg_loss += (param - info.initial_params[name]).pow(2).mean()

                if train_cfg.l2_reg_coef > 0:
                    loss += l2_init_reg_loss * train_cfg.l2_reg_coef

            with profile.learn:
                components.optimizer.zero_grad()
                loss.backward()

                before_clip_grad_norm = 0
                for p in components.policy.parameters():
                    if p.grad is not None:
                        before_clip_grad_norm += p.grad.norm().item()

                torch.nn.utils.clip_grad_norm_(components.policy.parameters(), train_cfg.max_grad_norm)

                # after_clip_grad_norm = 0
                # for p in components.policy.parameters():
                #     if p.grad is not None:
                #         after_clip_grad_norm += p.grad.norm().item()

                components.optimizer.step()
                if train_cfg.device == "cuda":
                    torch.cuda.synchronize()

            with profile.train_misc:
                losses.policy_loss += pg_loss.item() / total_minibatches
                losses.value_loss += v_loss.item() / total_minibatches
                losses.entropy += entropy_loss.item() / total_minibatches
                losses.old_approx_kl += old_approx_kl.item() / total_minibatches
                losses.approx_kl += approx_kl.item() / total_minibatches
                losses.clipfrac += clipfrac.item() / total_minibatches
                losses.before_clip_grad_norm += before_clip_grad_norm / total_minibatches
                # losses.after_clip_grad_norm += after_clip_grad_norm / total_minibatches
                losses.l2_init_reg_loss += l2_init_reg_loss.item() / total_minibatches

                if info.use_amp_obs:
                    losses.disc_loss += disc_loss.item() / total_minibatches
                    losses.disc_agent_acc += (disc_agent_logits < 0).float().mean() / total_minibatches
                    losses.disc_demo_acc += (disc_demo_logits > 0).float().mean() / total_minibatches

                if mean_bound_loss:
                    losses.mean_bound_loss += mean_bound_loss.item() / total_minibatches

        if train_cfg.target_kl is not None:
            if approx_kl > train_cfg.target_kl:
                break

    with profile.train_misc:
        if train_cfg.anneal_lr:
            frac = 1.0 - info.global_step / train_cfg.total_timesteps
            lrnow = frac * train_cfg.learning_rate
            components.optimizer.param_groups[0]["lr"] = lrnow

        y_pred = experience.values_np
        y_true = experience.returns_np
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        losses.explained_variance = explained_var
        info.epoch += 1

        done_training = info.global_step >= train_cfg.total_timesteps
        # TODO: beter way to get episode return update without clogging dashboard
        # TODO: make this appear faster
        if done_training or profile.update(components, info):
            # Log stats to wandb using the mean_and_log method
            info.stats.mean_and_log(components, info, losses)

            print_dashboard(
                info.env_name,
                utilization,
                info.global_step,
                info.epoch,
                profile,
                losses,
                info.stats.items(),
                info.msg,
            )
            info.stats.clear()

        if info.epoch % train_cfg.checkpoint_interval == 0 or done_training:
            save_checkpoint(components, info)
            info.msg = f"Checkpoint saved at update {info.epoch}"


def close(components: TrainComponents, info: TrainInfo, utilization: Utilization):
    components.vecenv.close()
    utilization.stop()
    if info.wandb is not None:
        artifact_name = f"{info.exp_id}_model"
        artifact = info.wandb.Artifact(artifact_name, type="model")
        model_path = save_checkpoint(components, info)
        artifact.add_file(model_path)
        # NOTE: PHC model is large to save for all sweep runs
        # components.wandb.run.log_artifact(artifact)
        info.wandb.finish()


def save_checkpoint(components: TrainComponents, info: TrainInfo):
    train_cfg = info.config
    path = os.path.join(train_cfg.data_dir, info.exp_id)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f"model_{info.epoch:06d}.pt"
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        return model_path

    checkpoint = {"config": info.config, "state_dict": components.uncompiled_policy.state_dict()}
    torch.save(checkpoint, model_path)

    checkpoint_state = {
        "optimizer_state_dict": components.optimizer.state_dict(),
        "global_step": info.global_step,
        "agent_step": info.global_step,
        "update": info.epoch,
        "model_name": model_name,
        "exp_id": info.exp_id,
    }
    state_path = os.path.join(path, "trainer_state.pt")
    torch.save(checkpoint_state, state_path + ".tmp")
    os.rename(state_path + ".tmp", state_path)
    return model_path


def try_load_checkpoint(components: TrainComponents, info: TrainInfo):
    train_cfg = info.config
    path = os.path.join(train_cfg.data_dir, info.exp_id)
    if not os.path.exists(path):
        print("No checkpoints found. Assuming new experiment")
        return

    trainer_path = os.path.join(path, "trainer_state.pt")
    resume_state = torch.load(trainer_path)
    model_path = os.path.join(path, resume_state["model_name"])
    components.policy.uncompiled.load_state_dict(model_path, map_location=train_cfg.device)
    components.optimizer.load_state_dict(resume_state["optimizer_state_dict"])
    print(f"Loaded checkpoint {resume_state['model_name']}")


def count_params(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


def seed_everything(seed, torch_deterministic):
    print(f"Seeding Everything with seed: {seed}" + ("and torch_deterministic" if torch_deterministic else ""))
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
