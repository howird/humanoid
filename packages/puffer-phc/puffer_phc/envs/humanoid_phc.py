import os
from types import SimpleNamespace
from typing import Dict, Union
from dataclasses import asdict

from isaacgym import gymapi

try:
    import gymtorch
except ImportError:
    from isaacgym import gymtorch

from gym import spaces
import torch
import numpy as np

from puffer_phc import ASSET_DIR
from puffer_phc.config import EnvConfig
from puffer_phc.envs.state_init import StateInit
from puffer_phc.envs.isaacgym_env import IsaacGymBase
from puffer_phc.envs.common import (
    compute_humanoid_observations_smpl_max,
    compute_imitation_observations_v6,
    build_amp_observations_smpl,
    compute_imitation_reward,
    compute_humanoid_im_reset,
)
from puffer_phc.poselib_skeleton import SkeletonTree
from puffer_phc.motion_lib import MotionLibSMPL, FixHeightMode
from puffer_phc.torch_utils import to_torch, torch_rand_float

from puffer_phc.body_sets import (
    build_body_ids_tensor,
    BODY_NAMES,
    DOF_NAMES,
    REMOVE_NAMES,
    LIMB_WEIGHT_GROUP,
    KEY_BODIES,
    CONTACT_BODIES,
    TRACK_BODIES,
    RESET_BODIES,
    EVAL_BODIES,
)


class HumanoidPHC:
    def __init__(
        self,
        cfg: EnvConfig,
    ):
        self.cfg: EnvConfig = cfg
        self.isaac_base = IsaacGymBase(self.cfg.device_type, self.cfg.device_id, self.cfg.headless)

        self.gym = self.isaac_base.gym
        self.sim = self.isaac_base.sim
        self.viewer = self.isaac_base.viewer

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.all_env_ids = torch.arange(self.cfg.num_envs).to(self.cfg.device)

        ### Robot
        self._config_robot()
        # NOTE: PHC does not use force sensors.
        self._create_force_sensors(sensor_joint_names=[])  # No sensor joints

        ### Env
        self._config_env()  # All env configs should be here
        self._create_ground_plane()
        # TODO: Testing putting the robots in the same env
        self._create_envs()
        self.gym.prepare_sim(self.sim)

        self._define_gym_spaces()
        self._setup_gym_tensors()
        self._setup_env_buffers()

        ### Flags
        # NOTE: These are to replace flags.
        self.flag_test = False
        self.flag_im_eval = False
        self.flag_debug = self.cfg.device == "cpu"  # CHECK ME

        ### Motion data
        # NOTE: self.flag_im_eval is used in _load_motion
        self._load_motion(self.cfg.motion_file)

    def reset(self, env_ids=None):
        safe_reset = (env_ids is None) or len(env_ids) == self.cfg.num_envs
        if env_ids is None:
            env_ids = self.all_env_ids

        self._reset_envs(env_ids)

        # ZL: This way it will simulate one step, then get reset again, squashing any remaining wiredness. Temporary fix
        if safe_reset:
            self.gym.simulate(self.sim)
            self._reset_envs(env_ids)
            torch.cuda.empty_cache()

        return self.obs_buf

    def step(self, actions):
        ### Apply actions, which was self.pre_physics_step(actions)
        if self.cfg.robot.reduce_action:
            # NOTE: not using it now. We don't have to create a new tensor every time?
            actions_full = torch.zeros([actions.shape[0], self.num_dof]).to(self.cfg.device)
            actions_full[:, self.cfg.robot.reduced_action_idx] = actions
            pd_tar = self._action_to_pd_targets(actions_full)

        else:
            pd_tar = self._action_to_pd_targets(actions)

            if self.cfg.robot.freeze_hand:
                hand_idx = DOF_NAMES.index("L_Hand") * 3
                r_hand_idx = DOF_NAMES.index("R_Hand") * 3
                pd_tar[:, hand_idx : hand_idx + 3] = 0
                pd_tar[:, r_hand_idx : r_hand_idx + 3] = 0
            if self.cfg.robot.freeze_toe:
                toe_idx = DOF_NAMES.index("L_Toe") * 3
                r_toe_idx = DOF_NAMES.index("R_Toe") * 3
                pd_tar[:, toe_idx : toe_idx + 3] = 0
                pd_tar[:, r_toe_idx : r_toe_idx + 3] = 0

        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

        ### self._physics_step()
        for _ in range(self.isaac_base.control_freq_inv):
            self.gym.simulate(self.sim)

        self.gym.fetch_results(self.sim, True)

        ### Compute observations, rewards, resets, which was self.post_physics_step()
        # This is after stepping, so progress buffer got + 1. Compute reset/reward do not need to forward 1 timestep since they are for "this" frame now.
        self.progress_buf += 1

        self._refresh_sim_tensors()

        self._compute_reward()

        # NOTE: Which envs must be reset is computed here, but the envs get reset outside the env
        self._compute_reset()

        # TODO: Move the code for resetting the env here?

        self._compute_observations()  # observation for the next step.

        self.extras["terminate"] = self._terminate_buf.clone()
        self.extras["reward_raw"] = self.reward_raw.detach()

        if self.cfg.use_amp_obs:
            self._update_hist_amp_obs()  # One step for the amp obs
            self._compute_amp_observations()
            self.extras["amp_obs"] = self.amp_obs  ## ZL: hooks for adding amp_obs for training

        if self.flag_im_eval:
            motion_times = (
                (self.progress_buf) * self.isaac_base.dt + self._motion_start_times + self._motion_start_times_offset
            )  # already has time + 1, so don't need to + 1 to get the target for "this frame"
            motion_res = self._get_state_from_motionlib_cache(
                self._sampled_motion_ids, motion_times, self._global_offset
            )  # pass in the env_ids such that the motion is in synced.
            body_pos = self._rigid_body_pos
            self.extras["mpjpe"] = (body_pos - motion_res["rg_pos"]).norm(dim=-1).mean(dim=-1)
            self.extras["body_pos"] = body_pos.cpu().numpy()
            self.extras["body_pos_gt"] = motion_res["rg_pos"].cpu().numpy()

        # obs, reward, done, info
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def render(self):
        if self.viewer:
            self.isaac_base.render()

    def close(self):
        self.isaac_base.close()

    #####################################################################
    ### __init__()
    #####################################################################

    def _config_robot(self):
        # Calculate dof_subset
        disc_idxes = []
        for idx, name in enumerate(DOF_NAMES):
            if name not in REMOVE_NAMES:
                disc_idxes.append(np.arange(idx * 3, (idx + 1) * 3))

        self.dof_subset = (
            torch.from_numpy(np.concatenate(disc_idxes)) if len(disc_idxes) > 0 else torch.tensor([]).long()
        )

        ### Load the Neutral SMPL humanoid asset only
        self.gender_beta = np.zeros(17)  # NOTE: gender (1) + betas (16)

        # And we use the same humanoid shapes for all the agents.
        self.humanoid_shapes = (
            torch.tensor(np.array([self.gender_beta] * self.cfg.num_envs)).float().to(self.cfg.device)
        )

        # NOTE: The below SMPL assets must be present.
        asset_file_real = str(ASSET_DIR / "smpl_humanoid.xml")
        assert os.path.exists(asset_file_real)

        sk_tree = SkeletonTree.from_mjcf(asset_file_real)
        self.skeleton_trees = [sk_tree] * self.cfg.num_envs

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self.humanoid_asset = self.gym.load_asset(self.sim, "/", asset_file_real, asset_options)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(self.humanoid_asset)

        self._dof_offsets = np.linspace(0, self.num_dof, self.num_bodies).astype(int)

        assert self.num_bodies == len(BODY_NAMES), "Number of bodies in asset file does not match number of SMPL bodies"
        assert self.num_dof == len(DOF_NAMES) * 3, "Number of DOF in asset file does not match number of SMPL DOF"

        # Check if the body ids are consistent between humanoid_asset and body_names (SMPL_MUJOCO_NAMES)
        for body_id, body_name in enumerate(BODY_NAMES):
            body_id_asset = self.gym.find_asset_rigid_body_index(self.humanoid_asset, body_name)
            assert body_id == body_id_asset, (
                f"Body id {body_id} does not match index {body_id_asset} for body {body_name}"
            )

    def _create_force_sensors(self, sensor_joint_names):
        sensor_pose = gymapi.Transform()

        for jt in sensor_joint_names:
            joint_idx = self.gym.find_asset_rigid_body_index(self.humanoid_asset, jt)
            self.gym.create_asset_force_sensor(self.humanoid_asset, joint_idx, sensor_pose)

    def _config_env(self):
        self._termination_distances = to_torch(
            np.array([self.cfg.termination_distance] * self.num_bodies), device=self.cfg.device
        )
        self._termination_distances_backup = self._termination_distances.clone()  # keep backup for train/eval

        self._key_body_ids = build_body_ids_tensor(BODY_NAMES, KEY_BODIES, self.cfg.device)
        self._contact_body_ids = build_body_ids_tensor(BODY_NAMES, CONTACT_BODIES, self.cfg.device)
        self._track_bodies_id = build_body_ids_tensor(BODY_NAMES, TRACK_BODIES, self.cfg.device)

        self._reset_bodies_id = build_body_ids_tensor(BODY_NAMES, RESET_BODIES, self.cfg.device)
        self._reset_bodies_id_backup = self._reset_bodies_id.clone()  # keep backup for train/eval

        # Used in https://github.com/kywch/PHC/blob/pixi/phc/learning/im_amp.py#L181
        self._eval_track_bodies_id = build_body_ids_tensor(BODY_NAMES, EVAL_BODIES, self.cfg.device)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
        plane_params.static_friction = 1.0  # self.cfg["env"]["plane"]["staticFriction"]
        plane_params.dynamic_friction = 1.0  # self.cfg["env"]["plane"]["dynamicFriction"]
        plane_params.restitution = 0.0  # self.cfg["env"]["plane"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        self.envs = []
        self.env_origins = []
        self.humanoid_handles = []
        self.humanoid_masses = []
        self.humanoid_limb_and_weights = []
        max_agg_bodies, max_agg_shapes = 160, 160

        lower = gymapi.Vec3(-self.cfg.env_spacing, -self.cfg.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg.env_spacing, self.cfg.env_spacing, self.cfg.env_spacing)
        num_per_row = int(np.sqrt(self.cfg.num_envs))

        # Since the same humanoid is used for all the envs ...
        dof_prop = self.gym.get_asset_dof_properties(self.humanoid_asset)
        assert self.cfg.control_mode == "isaac_pd"
        dof_prop["driveMode"] = gymapi.DOF_MODE_POS
        dof_prop["stiffness"] *= self.cfg.kp_scale
        dof_prop["damping"] *= self.cfg.kd_scale

        # NOTE: (from Joseph) You get a small perf boost (~4%) by putting all the actors in the same env
        for i in range(self.cfg.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # NOTE: Different humanoid asset files can be provided to _build_env() for each env
            self._build_single_env(i, env_ptr, self.humanoid_asset, dof_prop)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

            # Save the env origins for the camera work (render_env)
            row = i // num_per_row
            col = i % num_per_row
            self.env_origins.append((col * 2 * self.cfg.env_spacing, row * 2 * self.cfg.env_spacing, 0.0))

        # NOTE: self.humanoid_limb_and_weights comes from self._build_env()
        self.humanoid_limb_and_weights = torch.stack(self.humanoid_limb_and_weights).to(self.cfg.device)

        # These should be all the same because we use the same humanoid for all agents
        print("Humanoid Weights", self.humanoid_masses[:10])

        ### Define dof limits
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            elif dof_prop["lower"][j] == dof_prop["upper"][j]:
                print("Warning: DOF limits are the same")
                if dof_prop["lower"][j] == 0:
                    self.dof_limits_lower.append(-np.pi)
                    self.dof_limits_upper.append(np.pi)
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.cfg.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.cfg.device)
        self.dof_limits = torch.stack([self.dof_limits_lower, self.dof_limits_upper], dim=-1)
        self.torque_limits = to_torch(dof_prop["effort"], device=self.cfg.device)

        self._build_pd_action_offset_scale()

    # NOTE: HumanoidRenderEnv overrides this method to add marker actors
    def _build_single_env(self, env_id, env_ptr, humanoid_asset, dof_prop):
        # Collision settings: probably affect speed a lot
        if self.cfg.divide_group:
            # TODO(howird): idk what this does
            col_group = self._group_ids[env_id]
        else:
            col_group = env_id  # no inter-environment collision
        col_filter = 0 if self.cfg.robot.has_self_collision else 1

        pos = torch.tensor((0, 0, 0.89)).to(self.cfg.device)  # NOTE: char_h (0.89) hard coded
        pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.cfg.device).squeeze(
            1
        )  # ZL: segfault if we do not randomize the position

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # NOTE: Domain randomization code was here. Search for self.cfg.domain_rand.has_domain_rand in the original repos.

        humanoid_handle = self.gym.create_actor(
            env_ptr, humanoid_asset, start_pose, f"humanoid_{env_id}", col_group, col_filter, 0
        )
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        mass_ind = [prop.mass for prop in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)]
        humanoid_mass = np.sum(mass_ind)
        self.humanoid_masses.append(humanoid_mass)

        curr_skeleton_tree = self.skeleton_trees[env_id]
        limb_lengths = torch.norm(curr_skeleton_tree.local_translation, dim=-1)
        limb_lengths = [limb_lengths[group].sum() for group in LIMB_WEIGHT_GROUP]
        masses = torch.tensor(mass_ind)
        masses = [masses[group].sum() for group in LIMB_WEIGHT_GROUP]
        humanoid_limb_weight = torch.tensor(limb_lengths + masses)
        self.humanoid_limb_and_weights.append(humanoid_limb_weight)  # ZL: attach limb lengths and full body weight.

        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        if self.cfg.robot.has_self_collision:
            if self.cfg.robot.has_mesh:
                filter_ints = [0, 1, 224, 512, 384, 1, 1792, 64, 1056, 4096, 6, 6168, 0, 2048, 0, 20, 0, 0, 0, 0, 10, 0, 0, 0]  # fmt: skip
            else:
                filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # fmt: skip

            props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
            assert len(filter_ints) == len(props)

            for p_idx in range(len(props)):
                props[p_idx].filter = filter_ints[p_idx]
            self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        self.humanoid_handles.append(humanoid_handle)

    def _build_pd_action_offset_scale(self):
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        num_joints = len(self._dof_offsets) - 1
        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]
            if not self.cfg.robot.bias_offset:
                if dof_size == 3:
                    curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
                    curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
                    curr_low = np.max(np.abs(curr_low))
                    curr_high = np.max(np.abs(curr_high))
                    curr_scale = max([curr_low, curr_high])
                    curr_scale = 1.2 * curr_scale
                    curr_scale = min([curr_scale, np.pi])

                    lim_low[dof_offset : (dof_offset + dof_size)] = -curr_scale
                    lim_high[dof_offset : (dof_offset + dof_size)] = curr_scale

                    # lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                    # lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

                elif dof_size == 1:
                    curr_low = lim_low[dof_offset]
                    curr_high = lim_high[dof_offset]
                    curr_mid = 0.5 * (curr_high + curr_low)

                    # extend the action range to be a bit beyond the joint limits so that the motors
                    # don't lose their strength as they approach the joint limits
                    curr_scale = 0.7 * (curr_high - curr_low)
                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale

                    lim_low[dof_offset] = curr_low
                    lim_high[dof_offset] = curr_high
            else:
                curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
                curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset : (dof_offset + dof_size)] = curr_low
                lim_high[dof_offset : (dof_offset + dof_size)] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.cfg.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.cfg.device)

        self._L_knee_dof_idx = DOF_NAMES.index("L_Knee") * 3 + 1
        self._R_knee_dof_idx = DOF_NAMES.index("R_Knee") * 3 + 1

        # ZL: Modified SMPL to give stronger knee
        self._pd_action_scale[self._L_knee_dof_idx] = 5
        self._pd_action_scale[self._R_knee_dof_idx] = 5

        if self.cfg.robot.has_smpl_pd_offset:
            if self.cfg.robot.has_upright_start:
                self._pd_action_offset[DOF_NAMES.index("L_Shoulder") * 3] = -np.pi / 2
                self._pd_action_offset[DOF_NAMES.index("R_Shoulder") * 3] = np.pi / 2
            else:
                self._pd_action_offset[DOF_NAMES.index("L_Shoulder") * 3] = -np.pi / 6
                self._pd_action_offset[DOF_NAMES.index("L_Shoulder") * 3 + 2] = -np.pi / 2
                self._pd_action_offset[DOF_NAMES.index("R_Shoulder") * 3] = -np.pi / 3
                self._pd_action_offset[DOF_NAMES.index("R_Shoulder") * 3 + 2] = np.pi / 2

    def _define_gym_spaces(self):
        ### Observations
        # Self obs: height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
        self._num_self_obs = 1 + self.num_bodies * (3 + 6 + 3 + 3) - 3

        # Task obs: what goes into this? Check compute obs
        self._task_obs_size = len(TRACK_BODIES) * self.num_bodies

        self.num_obs = self._num_self_obs + self._task_obs_size  # = 934
        assert self.num_obs == 934

        # AMP obs
        self._dof_obs_size = len(DOF_NAMES) * 6
        self._num_amp_obs_per_step = (
            13 + self._dof_obs_size + self.num_dof + 3 * len(KEY_BODIES)
        )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

        if self.cfg.robot.has_dof_subset:
            self._num_amp_obs_per_step -= (6 + 3) * int((self.num_dof - len(self.dof_subset)) / 3)

        self.num_amp_obs = self.cfg.num_amp_obs_steps * self._num_amp_obs_per_step

        ### Actions
        if self.cfg.robot.reduce_action:
            self.num_actions = len(self.cfg.robot.reduced_action_idx)
        else:
            self.num_actions = self.num_dof

        ### Gym/puffer spaces
        self.single_observation_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf, dtype=np.float32
        )
        self.amp_observation_space = spaces.Box(
            np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf, dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0, dtype=np.float32
        )

    def _setup_gym_tensors(self):
        ### get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)  # Keep this as reference
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.cfg.num_envs, self.num_dof)

        self._refresh_sim_tensors()

        # NOTE: refresh_force_sensor_tensor, refresh_dof_force_tensor were not here. Any change in learning?
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self._root_states.shape[0] // self.cfg.num_envs

        self._humanoid_root_states = self._root_states.view(self.cfg.num_envs, num_actors, actor_root_state.shape[-1])[
            ..., 0, :
        ]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0
        # NOTE: 13 comes from pos 3 + rot 4 + vel 3 + ang vel 3.
        # root_states[:, 7:13] = 0 means zeroing vel and ang vel.

        self._humanoid_actor_ids = num_actors * torch.arange(
            self.cfg.num_envs, device=self.cfg.device, dtype=torch.int32
        )

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.cfg.num_envs
        self._dof_pos = self._dof_state.view(self.cfg.num_envs, dofs_per_env, 2)[..., : self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.cfg.num_envs, dofs_per_env, 2)[..., : self.num_dof, 1]

        # NOTE: These are used in self._reset_default(), along with self._initial_humanoid_root_states
        # CHECK ME: Is it ok to use zeros for _initial_dof_pos and _initial_dof_vel?
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.cfg.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.cfg.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.cfg.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.cfg.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = self._rigid_body_state_reshaped[..., : self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., : self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., : self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., : self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.cfg.num_envs, bodies_per_env, 3)[
            ..., : self.num_bodies, :
        ]

    def _setup_env_buffers(self):
        self.obs_buf = torch.zeros((self.cfg.num_envs, self.num_obs), device=self.cfg.device, dtype=torch.float)
        # self.self_obs_buf = torch.zeros((self.cfg.num_envs, self._num_self_obs), device=self.cfg.device, dtype=torch.float)

        self.rew_buf = torch.zeros(self.cfg.num_envs, device=self.cfg.device, dtype=torch.float)
        # TODO(howird): store indiviaul reward components. 4 and 5 are hardcoded for now.
        self.reward_raw = torch.zeros(
            (
                self.cfg.num_envs,
                self.cfg.reward.imitation_reward_dim + 1
                if self.cfg.reward.use_power_reward
                else self.cfg.reward.imitation_reward_dim,
            )
        ).to(self.cfg.device)

        self.progress_buf = torch.zeros(self.cfg.num_envs, device=self.cfg.device, dtype=torch.short)

        self.reset_buf = torch.ones(self.cfg.num_envs, device=self.cfg.device, dtype=torch.bool)  # This is dones
        # _terminate_buf records early termination
        self._terminate_buf = torch.ones(self.cfg.num_envs, device=self.cfg.device, dtype=torch.bool)

        self.extras = {}  # Stores info

        # TODO(howird): states not used here, but keeping it for now
        self.states_buf = torch.zeros(
            (self.cfg.num_envs, self.cfg.num_states), device=self.cfg.device, dtype=torch.float
        )

        # NOTE: related to domain randomization. Not used here.
        # self.randomize_buf = torch.zeros(self.cfg.num_envs, device=self.cfg.device, dtype=torch.long)

        # AMP/Motion-related
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._state_reset_happened = False

        self._global_offset = torch.zeros([self.cfg.num_envs, 3]).to(self.cfg.device)  # pos offset, so dim is 3
        self._motion_start_times = torch.zeros(self.cfg.num_envs).to(self.cfg.device)
        self._motion_start_times_offset = torch.zeros(self.cfg.num_envs).to(self.cfg.device)

        self._motion_sample_start_idx = 0
        self._sampled_motion_ids = torch.arange(self.cfg.num_envs).to(self.cfg.device)
        self.ref_motion_cache = {}

        self._amp_obs_buf = torch.zeros(
            (self.cfg.num_envs, self.cfg.num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.cfg.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        # amp_obs_demo_buf is fed into the discriminator training as the real motion data
        # This replaces the demo replay buffer in the original PHC code
        # amp_batch_size is fixed to the number of envs
        self._amp_obs_demo_buf = torch.zeros_like(self._amp_obs_buf)

        # NOTE: These don't seem to be used, except ref_dof_pos when self.cfg.res_action is True
        # self.ref_body_pos = torch.zeros_like(self._rigid_body_pos)
        # self.ref_body_vel = torch.zeros_like(self._rigid_body_vel)
        # self.ref_body_rot = torch.zeros_like(self._rigid_body_rot)
        # self.ref_body_pos_subset = torch.zeros_like(self._rigid_body_pos[:, self._track_bodies_id])
        self.ref_dof_pos = torch.zeros_like(self._dof_pos)

    def _load_motion(self, motion_train_file):
        motion_lib_cfg = SimpleNamespace(
            motion_file=motion_train_file,
            device=self.cfg.device,
            fix_height=FixHeightMode.full_fix,
            min_length=self.cfg.min_motion_len,
            # NOTE: this max_length determines the training time, so using 300 for now
            # TODO: find a way to evaluate full motion, probably not during training
            max_length=self.cfg.max_episode_length,
            im_eval=self.flag_im_eval,
            num_thread=4,
            smpl_type=self.cfg.robot.humanoid_type,
            step_dt=self.isaac_base.dt,
            is_deterministic=self.flag_debug,
        )
        self._motion_train_lib = MotionLibSMPL(motion_lib_cfg)
        self._motion_lib = self._motion_train_lib

        # TODO: Use motion_test_file for eval?
        motion_lib_cfg.im_eval = True
        self._motion_eval_lib = MotionLibSMPL(motion_lib_cfg)

        # When loading the motions the first time, use even sampling
        interval = self.num_unique_motions / (self.cfg.num_envs + 50)  # 50 is arbitrary
        sample_idxes = np.arange(0, self.num_unique_motions, interval)
        sample_idxes = np.floor(sample_idxes).astype(int)[: self.cfg.num_envs]
        sample_idxes = torch.from_numpy(sample_idxes).to(self.cfg.device)

        self._motion_lib.load_motions(
            skeleton_trees=self.skeleton_trees,
            gender_betas=self.humanoid_shapes.cpu(),
            limb_weights=self.humanoid_limb_and_weights.cpu(),
            # NOTE: During initial loading, use even sampling
            sample_idxes=sample_idxes,
            # random_sample=(not self.flag_test) and (not self.seq_motions),
            # max_len=-1 if self.flag_test else self.max_episode_length,  # NOTE: this is ignored in motion lib
            # start_idx=self._motion_sample_start_idx,
        )

    #####################################################################
    ### reset()
    #####################################################################

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) > 0:
            self._reset_actors(env_ids)  # this funciton call _set_env_state, and should set all state vectors
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._state_reset_happened = True

        if self.cfg.use_amp_obs:
            self._init_amp_obs(env_ids)

    def _reset_actors(self, env_ids):
        if self.cfg.state_init == StateInit.Default:
            self._reset_default(env_ids)
        elif self.cfg.state_init == StateInit.Start or self.cfg.state_init == StateInit.Random:
            self._reset_ref_state_init(env_ids)
        elif self.cfg.state_init == StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            raise ValueError(f"Unsupported state initialization strategy: {str(self.cfg.state_init)}")

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

    def _reset_ref_state_init(self, env_ids):
        (
            motion_ids,
            motion_times,
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            rb_pos,
            rb_rot,
            body_vel,
            body_ang_vel,
        ) = self._sample_ref_state(env_ids)

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            rigid_body_pos=rb_pos,
            rigid_body_rot=rb_rot,
            rigid_body_vel=body_vel,
            rigid_body_ang_vel=body_ang_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

        self._global_offset[env_ids] = 0  # Reset the global offset when resampling.
        self._motion_start_times[env_ids] = motion_times
        self._motion_start_times_offset[env_ids] = 0  # Reset the motion time offsets
        self._sampled_motion_ids[env_ids] = motion_ids

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self.cfg.hybrid_init_prob] * num_envs), device=self.cfg.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self._reset_default(default_reset_ids)

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_pos.contiguous()),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # print("#################### refreshing ####################")
        # print("rb", (self._rigid_body_state_reshaped[None, :] - self._rigid_body_state_reshaped[:, None]).abs().sum())
        # print("contact", (self._contact_forces[None, :] - self._contact_forces[:, None]).abs().sum())
        # print('dof_pos', (self._dof_pos[None, :] - self._dof_pos[:, None]).abs().sum())
        # print("dof_vel", (self._dof_vel[None, :] - self._dof_vel[:, None]).abs().sum())
        # print("root_states", (self._humanoid_root_states[None, :] - self._humanoid_root_states[:, None]).abs().sum())
        # print("#################### refreshing ####################")

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._contact_forces[env_ids] = 0

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if len(self._reset_default_env_ids) > 0:
            raise NotImplementedError("Not tested yet")
            # self._init_amp_obs_default(self._reset_default_env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids, self._reset_ref_motion_times)

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self.cfg.num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)

        time_steps = -self.isaac_base.dt * (torch.arange(0, self.cfg.num_amp_obs_steps - 1, device=self.cfg.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        amp_obs_demo = self._get_amp_obs(motion_ids, motion_times)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)

        # amp_obs_demo_buf is fed into the discriminator training as the real motion data
        self._amp_obs_demo_buf[env_ids] = self._amp_obs_buf[env_ids]

    def _get_amp_obs(self, motion_ids, motion_times):
        motion_res = self._get_state_from_motionlib_cache(motion_ids, motion_times)
        key_pos = motion_res["rg_pos"][:, self._key_body_ids]
        key_vel = motion_res["body_vel"][:, self._key_body_ids]

        return self._compute_amp_observations_from_state(
            motion_res["root_pos"],
            motion_res["root_rot"],
            motion_res["root_vel"],
            motion_res["root_ang_vel"],
            motion_res["dof_pos"],
            motion_res["dof_vel"],
            key_pos,
            key_vel,
            motion_res["motion_bodies"],
            motion_res["motion_limb_weights"],
            self.dof_subset,
        )

    def _sample_time(self, motion_ids):
        # Motion imitation, no more blending and only sample at certain locations
        return self._motion_lib.sample_time_interval(motion_ids)
        # return self._motion_lib.sample_time(motion_ids)

    def _sample_ref_state(self, env_ids):
        num_envs = env_ids.shape[0]

        if self.cfg.state_init == StateInit.Random or self.cfg.state_init == StateInit.Hybrid:
            motion_times = self._sample_time(self._sampled_motion_ids[env_ids])
        elif self.cfg.state_init == StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.cfg.device)
        else:
            raise ValueError("Unsupported state initialization strategy: {:s}".format(str(self.cfg.state_init)))

        if self.flag_test:
            motion_times[:] = 0

        motion_res = self._get_state_from_motionlib_cache(
            self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids]
        )

        return (
            self._sampled_motion_ids[env_ids],
            motion_times,
            motion_res["root_pos"],
            motion_res["root_rot"],
            motion_res["dof_pos"],
            motion_res["root_vel"],
            motion_res["root_ang_vel"],
            motion_res["dof_vel"],
            motion_res["rg_pos"],  # rb_pos
            motion_res["rb_rot"],
            motion_res["body_vel"],
            motion_res["body_ang_vel"],
        )

    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ### Cache the motion + offset
        if (
            offset is None
            or "motion_ids" not in self.ref_motion_cache
            or self.ref_motion_cache["offset"] is None
            or len(self.ref_motion_cache["motion_ids"]) != len(motion_ids)
            or len(self.ref_motion_cache["offset"]) != len(offset)
            or (self.ref_motion_cache["motion_ids"] - motion_ids).abs().sum()
            + (self.ref_motion_cache["motion_times"] - motion_times).abs().sum()
            + (self.ref_motion_cache["offset"] - offset).abs().sum()
            > 0
        ):
            self.ref_motion_cache["motion_ids"] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache["motion_times"] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache["offset"] = offset.clone() if offset is not None else None

        else:
            return self.ref_motion_cache

        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        self.ref_motion_cache.update(motion_res)
        return self.ref_motion_cache

    def _set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        rigid_body_pos=None,
        rigid_body_rot=None,
        rigid_body_vel=None,
        rigid_body_ang_vel=None,
    ):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        if (rigid_body_pos is not None) and (rigid_body_rot is not None):
            self._rigid_body_pos[env_ids] = rigid_body_pos
            self._rigid_body_rot[env_ids] = rigid_body_rot
            self._rigid_body_vel[env_ids] = rigid_body_vel
            self._rigid_body_ang_vel[env_ids] = rigid_body_ang_vel

            self._reset_rb_pos = self._rigid_body_pos[env_ids].clone()
            self._reset_rb_rot = self._rigid_body_rot[env_ids].clone()
            self._reset_rb_vel = self._rigid_body_vel[env_ids].clone()
            self._reset_rb_ang_vel = self._rigid_body_ang_vel[env_ids].clone()

    #####################################################################
    ### compute observations
    #####################################################################

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = self.all_env_ids

        # This is the normalized state of the humanoid
        state = self._compute_humanoid_obs(env_ids)

        # This is the difference of state with the demo, but it is
        # called "state" in the paper.
        imitation = self._compute_task_obs(env_ids)

        # Possible the original paper only uses imitation
        obs = torch.cat([state, imitation], dim=-1)

        # NOTE: Not using it for now.
        # This is the normalized vector with position, rotation, velocity, and
        # angular velocity for the simulated humanoid and the demo data
        # self.state, self.demo = self._compute_state_obs(env_ids)

        if self.cfg.add_obs_noise and not self.flag_test:
            obs = obs + torch.randn_like(obs) * 0.1

        self.obs_buf[env_ids] = obs

        return obs

    def _compute_humanoid_obs(self, env_ids=None):
        with torch.no_grad():
            if env_ids is None:
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                body_shape_params = self.humanoid_shapes[:, :-6]
                limb_weights = self.humanoid_limb_and_weights

            else:
                body_pos = self._rigid_body_pos[env_ids]
                body_rot = self._rigid_body_rot[env_ids]
                body_vel = self._rigid_body_vel[env_ids]
                body_ang_vel = self._rigid_body_ang_vel[env_ids]
                body_shape_params = self.humanoid_shapes[env_ids, :-6]
                limb_weights = self.humanoid_limb_and_weights[env_ids]

            return compute_humanoid_observations_smpl_max(
                body_pos,
                body_rot,
                body_vel,
                body_ang_vel,
                body_shape_params,
                limb_weights,
                self.cfg.local_root_obs,  # Constant: True
                self.cfg.root_height_obs,  # Constant: True
                self.cfg.robot.has_upright_start,  # Constant: True
                self.cfg.robot.has_shape_obs,  # Constant: False
                self.cfg.robot.has_limb_weight_obs,  # Constant: False
            )

    # NOTE: This produces "simplified" amp obs, which goes into the discriminator
    def _compute_state_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        body_pos = self._rigid_body_pos[env_ids]  # [..., self._track_bodies_id]
        body_rot = self._rigid_body_rot[env_ids]  # [..., self._track_bodies_id]
        body_vel = self._rigid_body_vel[env_ids]  # [..., self._track_bodies_id]
        body_ang_vel = self._rigid_body_ang_vel[env_ids]  # [..., self._track_bodies_id]

        sim_obs = compute_humanoid_observations_smpl_max(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            None,
            None,
            self.cfg.local_root_obs,  # Constant: True
            self.cfg.root_height_obs,  # Constant: True
            self.cfg.robot.has_upright_start,  # Constant: True
            self.cfg.robot.has_shape_obs,  # Constant: False
            self.cfg.robot.has_limb_weight_obs,  # Constant: False
        )

        motion_times = (
            (self.progress_buf[env_ids] + 1) * self.isaac_base.dt
            + self._motion_start_times[env_ids]
            + self._motion_start_times_offset[env_ids]
        )  # Next frame, so +1

        motion_res = self._get_state_from_motionlib_cache(
            self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids]
        )  # pass in the env_ids such that the motion is in synced.

        demo_pos = motion_res["rg_pos"]  # [..., self._track_bodies_id]
        demo_rot = motion_res["rb_rot"]  # [..., self._track_bodies_id]
        demo_vel = motion_res["body_vel"]  # [..., self._track_bodies_id]
        demo_ang_vel = motion_res["body_ang_vel"]  # [..., self._track_bodies_id]

        demo_obs = compute_humanoid_observations_smpl_max(
            demo_pos,
            demo_rot,
            demo_vel,
            demo_ang_vel,
            None,
            None,
            True,  # Constant: True
            self.cfg.root_height_obs,  # Constant: True
            self.cfg.robot.has_upright_start,  # Constant: True
            self.cfg.robot.has_shape_obs,  # Constant: False
            self.cfg.robot.has_limb_weight_obs,  # Constant: False
        )

        return sim_obs, demo_obs

    def _compute_task_obs(self, env_ids=None, save_buffer=True):
        if env_ids is None:
            env_ids = self.all_env_ids
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]

        motion_times = (
            (self.progress_buf[env_ids] + 1) * self.isaac_base.dt
            + self._motion_start_times[env_ids]
            + self._motion_start_times_offset[env_ids]
        )  # Next frame, so +1

        motion_res = self._get_state_from_motionlib_cache(
            self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids]
        )  # pass in the env_ids such that the motion is in synced.

        (
            ref_dof_pos,
            ref_rb_pos,
            ref_rb_rot,
            ref_body_vel,
            ref_body_ang_vel,
        ) = (
            motion_res["dof_pos"],
            motion_res["rg_pos"],  # ref_rb_pos
            motion_res["rb_rot"],
            motion_res["body_vel"],
            motion_res["body_ang_vel"],
        )
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
        ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
        ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
        ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]

        # TODO: revisit constant args
        time_steps = 1  # Necessary?
        obs = compute_imitation_observations_v6(
            root_pos,
            root_rot,
            body_pos_subset,
            body_rot_subset,
            body_vel_subset,
            body_ang_vel_subset,
            ref_rb_pos_subset,
            ref_rb_rot_subset,
            ref_body_vel_subset,
            ref_body_ang_vel_subset,
            time_steps,  # Constant: 1
            self.cfg.robot.has_upright_start,  # Constant: True
        )

        if self.cfg.res_action and save_buffer:
            # self.ref_body_pos[env_ids] = ref_rb_pos
            # self.ref_body_vel[env_ids] = ref_body_vel
            # self.ref_body_rot[env_ids] = ref_rb_rot
            # self.ref_body_pos_subset[env_ids] = ref_rb_pos_subset
            self.ref_dof_pos[env_ids] = ref_dof_pos

        return obs

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        key_body_vel = self._rigid_body_vel[:, self._key_body_ids, :]

        # assert self.humanoid_type == "smpl"
        if self.dof_subset is None:
            # ZL hack
            (
                self._dof_pos[:, 9:12],
                self._dof_pos[:, 21:24],
                self._dof_pos[:, 51:54],
                self._dof_pos[:, 66:69],
            ) = 0, 0, 0, 0
            (
                self._dof_vel[:, 9:12],
                self._dof_vel[:, 21:24],
                self._dof_vel[:, 51:54],
                self._dof_vel[:, 66:69],
            ) = 0, 0, 0, 0

        if env_ids is None:
            # TODO: revisit constant args
            self._curr_amp_obs_buf[:] = self._compute_amp_observations_from_state(
                self._rigid_body_pos[:, 0, :],
                self._rigid_body_rot[:, 0, :],
                self._rigid_body_vel[:, 0, :],
                self._rigid_body_ang_vel[:, 0, :],
                self._dof_pos,
                self._dof_vel,
                key_body_pos,
                key_body_vel,
                self.humanoid_shapes,
                self.humanoid_limb_and_weights,
                self.dof_subset,
            )
        else:
            if len(env_ids) == 0:
                return

            self._curr_amp_obs_buf[env_ids] = self._compute_amp_observations_from_state(
                self._rigid_body_pos[env_ids][:, 0, :],
                self._rigid_body_rot[env_ids][:, 0, :],
                self._rigid_body_vel[env_ids][:, 0, :],
                self._rigid_body_ang_vel[env_ids][:, 0, :],
                self._dof_pos[env_ids],
                self._dof_vel[env_ids],
                key_body_pos[env_ids],
                key_body_vel[env_ids],
                self.humanoid_shapes[env_ids],
                self.humanoid_limb_and_weights[env_ids],
                self.dof_subset,
            )

    def _compute_amp_observations_from_state(
        self,
        root_pos,
        root_rot,
        root_vel,
        root_ang_vel,
        dof_pos,
        dof_vel,
        key_body_pos,
        key_body_vels,
        smpl_params,
        limb_weight_params,
        dof_subset,
    ):
        smpl_params = smpl_params[:, :-6]

        # TODO: revisit constant args
        return build_amp_observations_smpl(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_body_pos,
            smpl_params,
            limb_weight_params,
            dof_subset,
            self.cfg.local_root_obs,  # Constant: True
            self.cfg.amp_root_height_obs,  # Constant: True
            self.cfg.robot.has_dof_subset,  # Constant: True
            self.cfg.robot.has_shape_obs_disc,  # Constant: False
            self.cfg.robot.has_limb_weight_obs_disc,  # Constant: False
            self.cfg.robot.has_upright_start,  # Constant: True
        )

    #####################################################################
    ### step() -- pre_physics_step(), post_physics_step()
    #####################################################################

    def _action_to_pd_targets(self, action):
        # NOTE: self.cfg.res_action is False by default
        if self.cfg.res_action:
            pd_tar = self.ref_dof_pos + self._pd_action_scale * action
            pd_lower = self._dof_pos - np.pi / 2
            pd_upper = self._dof_pos + np.pi / 2
            pd_tar = torch.maximum(torch.minimum(pd_tar, pd_upper), pd_lower)
        else:
            pd_tar = self._pd_action_offset + self._pd_action_scale * action

        return pd_tar

    def _compute_reward(self):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel

        motion_times = (
            self.progress_buf * self.isaac_base.dt + self._motion_start_times + self._motion_start_times_offset
        )  # reward is computed after physics step, and progress_buf is already updated for next time step.

        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)

        (
            ref_rb_pos,
            ref_rb_rot,
            ref_body_vel,
            ref_body_ang_vel,
        ) = (
            motion_res["rg_pos"],  # ref_rb_pos
            motion_res["rb_rot"],
            motion_res["body_vel"],
            motion_res["body_ang_vel"],
        )

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        # NOTE: self._full_body_reward is True by default
        if self.cfg.reward.full_body_reward:
            self.rew_buf[:], self.reward_raw[:, : self.cfg.reward.imitation_reward_dim] = compute_imitation_reward(
                root_pos,
                root_rot,
                body_pos,
                body_rot,
                body_vel,
                body_ang_vel,
                ref_rb_pos,
                ref_rb_rot,
                ref_body_vel,
                ref_body_ang_vel,
                self.rwd_specs,
            )

        else:
            body_pos_subset = body_pos[..., self._track_bodies_id, :]
            body_rot_subset = body_rot[..., self._track_bodies_id, :]
            body_vel_subset = body_vel[..., self._track_bodies_id, :]
            body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

            ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
            ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
            ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
            ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]
            self.rew_buf[:], self.reward_raw[:, : self.cfg.reward.imitation_reward_dim] = compute_imitation_reward(
                root_pos,
                root_rot,
                body_pos_subset,
                body_rot_subset,
                body_vel_subset,
                body_ang_vel_subset,
                ref_rb_pos_subset,
                ref_rb_rot_subset,
                ref_body_vel_subset,
                ref_body_ang_vel_subset,
                self.rwd_specs,
            )

        if self.cfg.reward.use_power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)
            # power_reward = -0.00005 * (power ** 2)
            power_reward = -self.cfg.rew_power_coef * power
            # First 3 frame power reward should not be counted. since they could be dropped.
            power_reward[self.progress_buf <= 3] = 0

            self.rew_buf[:] += power_reward
            self.reward_raw[:, -1] = power_reward

    @property
    def rwd_specs(self) -> Dict[str, Union[float, bool]]:
        if not hasattr(self, "_rwd_specs"):
            self._rwd_specs = asdict(self.cfg.reward)
        return self._rwd_specs

    def _compute_reset(self):
        time = (
            (self.progress_buf) * self.isaac_base.dt + self._motion_start_times + self._motion_start_times_offset
        )  # Reset is also called after the progress_buf is updated.
        pass_time = time >= self._motion_lib._motion_lengths

        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, time, self._global_offset)

        body_pos = self._rigid_body_pos[..., self._reset_bodies_id, :].clone()
        ref_body_pos = motion_res["rg_pos"][..., self._reset_bodies_id, :].clone()

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_im_reset(
            self.reset_buf,
            self.progress_buf,
            self._contact_forces,
            self._contact_body_ids,
            body_pos,
            ref_body_pos,
            pass_time,
            self.cfg.enable_early_termination,
            self._termination_distances[..., self._reset_bodies_id],
            self.flag_im_eval,
        )

    # NOTE: Training/eval code changes the termination distances.
    def set_termination_distances(self, termination_distances):
        self._termination_distances[:] = termination_distances

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            # Got RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.
            try:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0 : (self.cfg.num_amp_obs_steps - 1)]
            except:  # noqa
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0 : (self.cfg.num_amp_obs_steps - 1)].clone()

        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0 : (self.cfg.num_amp_obs_steps - 1)]

    #####################################################################
    ### Motion/AMP
    #####################################################################

    @property
    def amp_obs(self):
        return self._amp_obs_buf.view(-1, self.num_amp_obs) if self.cfg.use_amp_obs else None

    def fetch_amp_obs_demo(self):
        return self._amp_obs_demo_buf.view(-1, self.num_amp_obs) if self.cfg.use_amp_obs else None

    def resample_motions(self):
        if self.flag_test:
            self.forward_motion_samples()

        else:
            self._motion_lib.load_motions(
                skeleton_trees=self.skeleton_trees,
                limb_weights=self.humanoid_limb_and_weights.cpu(),
                gender_betas=self.humanoid_shapes.cpu(),
                random_sample=(not self.flag_test) and (not self.cfg.seq_motions),
                # max_len=-1 if self.flag_test else self.max_episode_length,  # NOTE: this is ignored in motion lib
            )

            time = self.progress_buf * self.isaac_base.dt + self._motion_start_times + self._motion_start_times_offset
            root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids, time)
            self._global_offset[:, :2] = self._humanoid_root_states[:, :2] - root_res["root_pos"][:, :2]
            self.reset()

    def begin_seq_motion_samples(self):
        # For evaluation
        self._motion_sample_start_idx = 0
        self._motion_lib.load_motions(
            skeleton_trees=self.skeleton_trees,
            gender_betas=self.humanoid_shapes.cpu(),
            limb_weights=self.humanoid_limb_and_weights.cpu(),
            random_sample=False,
            start_idx=self._motion_sample_start_idx,
        )
        self.reset()

    def forward_motion_samples(self):
        self._motion_sample_start_idx += self.cfg.num_envs
        self._motion_lib.load_motions(
            skeleton_trees=self.skeleton_trees,
            gender_betas=self.humanoid_shapes.cpu(),
            limb_weights=self.humanoid_limb_and_weights.cpu(),
            random_sample=False,
            start_idx=self._motion_sample_start_idx,
        )
        self.reset()

    @property
    def num_unique_motions(self):
        return self._motion_lib._num_unique_motions

    @property
    def current_motion_ids(self):
        return self._motion_lib._curr_motion_ids

    @property
    def motion_sample_start_idx(self):
        return self._motion_sample_start_idx

    @property
    def motion_data_keys(self):
        return self._motion_lib._motion_data_keys

    def get_motion_steps(self):
        return self._motion_lib.get_motion_num_steps()

    #####################################################################
    ### Toggle train/eval model. Used in the training/eval code
    #####################################################################
    def toggle_eval_mode(self):
        self.flag_test = True
        self.flag_im_eval = True

        # Relax the early termination condition for evaluation
        self.set_termination_distances(0.5)  # NOTE: hardcoded

        self._motion_lib = self._motion_eval_lib
        self.begin_seq_motion_samples()  # using _motion_eval_lib
        if len(self._reset_bodies_id) > 15:
            # Following UHC. Only do it for full body, not for three point/two point trackings.
            self._reset_bodies_id = self._eval_track_bodies_id

        # Return the number of motions
        return self._motion_lib._num_unique_motions

    def untoggle_eval_mode(self, failed_keys):
        self.flag_test = False
        self.flag_im_eval = False

        self.set_termination_distances(self._termination_distances_backup)
        self._motion_lib = self._motion_train_lib
        self._reset_bodies_id = self._reset_bodies_id_backup

        if self.cfg.auto_pmcp:
            self._motion_lib.update_hard_sampling_weight(failed_keys)
        elif self.cfg.auto_pmcp_soft:
            self._motion_lib.update_soft_sampling_weight(failed_keys)

        # Return the motion lib termination history
        return self._motion_lib._termination_history.clone()
