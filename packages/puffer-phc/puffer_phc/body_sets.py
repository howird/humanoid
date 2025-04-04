"""
Body part sets used in the PHC humanoid environment.

This module contains the definitions of various body part sets used for different
purposes in the PHC environment, such as tracking, contact detection, and evaluation.
"""

from typing import List, Tuple, Union

# Complete list of all body parts in the SMPL humanoid model
BODY_NAMES = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)

# DOF names (all body parts except the root)
DOF_NAMES = BODY_NAMES[1:]

# Body parts excluded from certain calculations due to unreliable motion data
REMOVE_NAMES = ("L_Hand", "R_Hand", "L_Toe", "R_Toe")

# Important body parts used for AMP observations
KEY_BODIES = ("R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist")

# Body parts monitored for contact with the ground
CONTACT_BODIES = ("R_Ankle", "L_Ankle", "R_Toe", "L_Toe")

# All body parts used for tracking (default is all body parts)
TRACK_BODIES = BODY_NAMES

# Body parts that trigger early termination if they deviate too much from reference
RESET_BODIES = TRACK_BODIES

# Body parts used during evaluation (all except those in REMOVE_NAMES)
EVAL_BODIES = tuple(name for name in BODY_NAMES if name not in REMOVE_NAMES)

# Joint groups for limb weight calculations
JOINT_GROUPS = [
    ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"],
    ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
    ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"],
    ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
    ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
]

# Pre-computed indices for common body parts
LEFT_INDEXES = [idx for idx, name in enumerate(DOF_NAMES) if name.startswith("L")]
LEFT_LOWER_INDEXES = [
    idx for idx, name in enumerate(DOF_NAMES) if name.startswith("L") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]
]
RIGHT_INDEXES = [idx for idx, name in enumerate(DOF_NAMES) if name.startswith("R")]
RIGHT_LOWER_INDEXES = [
    idx for idx, name in enumerate(DOF_NAMES) if name.startswith("R") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]
]

# Pre-computed limb weight group for the standard BODY_NAMES
LIMB_WEIGHT_GROUP = [[BODY_NAMES.index(joint_name) for joint_name in joint_group] for joint_group in JOINT_GROUPS]


def get_limb_weight_group(body_names: List[str]) -> List[List[int]]:
    """
    Convert joint groups from names to indices based on the provided body_names list.

    Args:
        body_names: List of body part names

    Returns:
        List of lists containing indices for each joint group
    """
    # If using the standard body names, return the pre-computed indices
    if body_names == BODY_NAMES:
        return LIMB_WEIGHT_GROUP

    return [[body_names.index(joint_name) for joint_name in joint_group] for joint_group in JOINT_GROUPS]


def get_left_indexes(dof_names: List[str]) -> List[int]:
    """Get indices of left-side joints"""
    # If using the standard DOF names, return the pre-computed indices
    if dof_names == DOF_NAMES:
        return LEFT_INDEXES

    return [idx for idx, name in enumerate(dof_names) if name.startswith("L")]


def get_left_lower_indexes(dof_names: List[str]) -> List[int]:
    """Get indices of left-side lower body joints"""
    # If using the standard DOF names, return the pre-computed indices
    if dof_names == DOF_NAMES:
        return LEFT_LOWER_INDEXES

    return [
        idx
        for idx, name in enumerate(dof_names)
        if name.startswith("L") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]
    ]


def get_right_indexes(dof_names: List[str]) -> List[int]:
    """Get indices of right-side joints"""
    # If using the standard DOF names, return the pre-computed indices
    if dof_names == DOF_NAMES:
        return RIGHT_INDEXES

    return [idx for idx, name in enumerate(dof_names) if name.startswith("R")]


def get_right_lower_indexes(dof_names: List[str]) -> List[int]:
    """Get indices of right-side lower body joints"""
    # If using the standard DOF names, return the pre-computed indices
    if dof_names == DOF_NAMES:
        return RIGHT_LOWER_INDEXES

    return [
        idx
        for idx, name in enumerate(dof_names)
        if name.startswith("R") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]
    ]


def build_body_ids_tensor(body_names: Tuple[str, ...], target_names: Union[List[str], Tuple[str, ...]], device: str):
    """
    Build a tensor of body IDs from a list of body names.

    Args:
        body_names: List of all body part names
        target_names: List of target body part names to get indices for
        device: Device to place the tensor on

    Returns:
        Tensor of body IDs
    """
    import torch

    body_ids = [body_names.index(name) for name in target_names]
    return torch.tensor(body_ids, device=device, dtype=torch.long)
