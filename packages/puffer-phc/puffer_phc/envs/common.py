import torch

from puffer_phc.torch_utils import (
    exp_map_to_quat,
    calc_heading_quat,
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_mul,
    quat_conjugate,
    quat_to_tan_norm,
    quat_to_angle_axis,
)


@torch.jit.script
def remove_base_rot(quat):
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))
    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))


# @torch.jit.script
def compute_humanoid_observations_smpl_max(
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    smpl_params,
    limb_weight_params,
    local_root_obs,
    root_height_obs,
    upright,
    has_smpl_params,
    has_limb_weight_params,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]  # (num_envs, 3)
    root_rot = body_rot[:, 0, :]  # (num_envs, 4)

    root_h = root_pos[:, 2:3]  # (num_envs, 1) get z coords of pelvis
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = calc_heading_quat_inv(root_rot)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],
        heading_rot_inv_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    if not (local_root_obs):
        root_rot_obs = quat_to_tan_norm(root_rot)  # If not local root obs, you override it.
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]

    if has_smpl_params:
        obs_list.append(smpl_params)

    if has_limb_weight_params:
        obs_list.append(limb_weight_params)

    obs = torch.cat(obs_list, dim=-1)
    return obs


@torch.jit.script
def compute_imitation_observations_v6(
    root_pos,
    root_rot,
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    ref_body_pos,
    ref_body_rot,
    ref_body_vel,
    ref_body_ang_vel,
    time_steps,
    upright,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = calc_heading_quat_inv(root_rot)
    heading_rot = calc_heading_quat(root_rot)
    heading_inv_rot_expand = (
        heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    )
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    body_rot[:, None].repeat_interleave(time_steps, 1)
    diff_global_body_rot = quat_mul(
        ref_body_rot.view(B, time_steps, J, 4),
        quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)),
    )
    diff_local_body_rot_flat = quat_mul(
        quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)),
        heading_rot_expand.view(-1, 4),
    )  # Need to be change of basis

    ##### linear and angular  Velocity differences
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))

    diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3) - body_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))

    ##### body pos + Dof_pos This part will have proper futures.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(
        B, 1, 1, 3
    )  # preserves the body position
    local_ref_body_pos = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
    local_ref_body_rot = quat_to_tan_norm(local_ref_body_rot)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * 24 * 3
    obs.append(quat_to_tan_norm(diff_local_body_rot_flat).view(B, time_steps, -1))  #  1 * timestep * 24 * 6
    obs.append(diff_local_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(diff_local_ang_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_rot.view(B, time_steps, -1))  # timestep  * 24 * 6

    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs


@torch.jit.script
def dof_to_obs_smpl(pose):
    # type: (Tensor) -> Tensor
    joint_obs_size = 6
    B, jts = pose.shape
    num_joints = int(jts / 3)

    joint_dof_obs = quat_to_tan_norm(exp_map_to_quat(pose.reshape(-1, 3))).reshape(B, -1)
    assert (num_joints * joint_obs_size) == joint_dof_obs.shape[1]

    return joint_dof_obs


@torch.jit.script
def build_amp_observations_smpl(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    shape_params,
    limb_weight_params,
    dof_subset,
    local_root_obs,
    root_height_obs,
    has_dof_subset,
    has_shape_obs_disc,
    has_limb_weight_obs,
    upright,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool, bool) -> Tensor
    B, N = root_pos.shape
    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot_inv, root_rot)
    else:
        root_rot_obs = root_rot

    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot_inv, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot_inv, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2]
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2]
    )
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2]
    )

    if has_dof_subset:
        dof_vel = dof_vel[:, dof_subset]
        dof_pos = dof_pos[:, dof_subset]

    dof_obs = dof_to_obs_smpl(dof_pos)
    obs_list = []
    if root_height_obs:
        obs_list.append(root_h)
    obs_list += [
        root_rot_obs,
        local_root_vel,
        local_root_ang_vel,
        dof_obs,
        dof_vel,
        flat_local_key_pos,
    ]
    # 1? + 6 + 3 + 3 + 114 + 57 + 12
    if has_shape_obs_disc:
        obs_list.append(shape_params)
    if has_limb_weight_obs:
        obs_list.append(limb_weight_params)
    obs = torch.cat(obs_list, dim=-1)

    return obs


@torch.jit.script
def compute_imitation_reward(
    root_pos,
    root_rot,
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    ref_body_pos,
    ref_body_rot,
    ref_body_vel,
    ref_body_ang_vel,
    rwd_specs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = (
        rwd_specs["k_pos"],
        rwd_specs["k_rot"],
        rwd_specs["k_vel"],
        rwd_specs["k_ang_vel"],
    )
    w_pos, w_rot, w_vel, w_ang_vel = (
        rwd_specs["w_pos"],
        rwd_specs["w_rot"],
        rwd_specs["w_vel"],
        rwd_specs["w_ang_vel"],
    )

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    diff_global_body_rot = quat_mul(ref_body_rot, quat_conjugate(body_rot))
    diff_global_body_angle = quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # body angular velocity reward
    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
    reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel], dim=-1)

    return reward, reward_raw


@torch.jit.script
def compute_humanoid_im_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    rigid_body_pos,
    ref_body_pos,
    pass_time,
    enable_early_termination,
    termination_distance,
    use_mean,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    if enable_early_termination:
        # NOTE: When evaluating, using mean is relaxed vs. max is strict.
        if use_mean:
            has_fallen = torch.any(
                torch.norm(rigid_body_pos - ref_body_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance[0],
                dim=-1,
            )  # using average, same as UHC"s termination condition
        else:
            has_fallen = torch.any(
                torch.norm(rigid_body_pos - ref_body_pos, dim=-1) > termination_distance, dim=-1
            )  # using max

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1

        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

        # if (contact_buf.abs().sum(dim=-1)[0] > 0).sum() > 2:
        #     np.set_printoptions(precision=4, suppress=1)
        #     print(contact_buf.numpy(), contact_buf.abs().sum(dim=-1)[0].nonzero().squeeze())

    reset = torch.where(pass_time, torch.ones_like(reset_buf), terminated)

    return reset, terminated
