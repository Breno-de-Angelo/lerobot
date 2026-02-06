#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unitree G1 Robot Constants and Joint Definitions.

This module provides a single source of truth for:
- Body joint indices (G1_29_JointIndex)
- Arm joint indices (G1_29_JointArmIndex)
- Dex3 hand joint indices and limits
- DDS topic names
- Joint name mappings
"""

import importlib
from enum import IntEnum

import numpy as np

# ruff: noqa: N801, N815

# ==============================================================================
# Body Constants
# ==============================================================================

# Juntas do corpo do G1 (29 DoF). NÃO confundir com o tamanho do array de motores
# do LowState do DDS, que tem 35 slots — o commit 1d49a4a7 do fork definia
# NUM_MOTORS = 35 aqui, com esse outro sentido. Mantido 29 (o significado do
# upstream, usado por unitree_g1.py, gr00t_locomotion.py e holosoma_locomotion.py);
# quem precisa dos 35 slots do DDS declara a constante localmente, como já faz o
# run_g1_server.py. Nenhum código do fork importa NUM_MOTORS daqui.
NUM_MOTORS = 29

REMOTE_AXES = ("remote.lx", "remote.ly", "remote.rx", "remote.ry")
REMOTE_BUTTONS = tuple(f"remote.button.{i}" for i in range(16))
REMOTE_KEYS = REMOTE_AXES + REMOTE_BUTTONS


def default_remote_input() -> dict[str, float]:
    """Return a zeroed-out remote input dict (axes + buttons)."""
    return dict.fromkeys(REMOTE_KEYS, 0.0)


def get_gravity_orientation(quaternion: list[float] | np.ndarray) -> np.ndarray:
    """Get gravity orientation from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


def make_locomotion_controller(name: str | None):
    """Instantiate a locomotion controller by class name. Returns None if name is None."""
    if name is None:
        return None
    controllers = {
        "GrootLocomotionController": "lerobot.robots.unitree_g1.gr00t_locomotion",
        "HolosomaLocomotionController": "lerobot.robots.unitree_g1.holosoma_locomotion",
    }
    module_path = controllers.get(name)
    if module_path is None:
        raise ValueError(f"Unknown controller: {name!r}. Available: {list(controllers)}")
    module = importlib.import_module(module_path)
    return getattr(module, name)()


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


# ==============================================================================
# Dex3 Hand Constants
# ==============================================================================

# DDS Topic names for Dex3 hand communication
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

# Number of motors per hand
Dex3_Num_Motors = 7


class Dex3_1_Left_JointIndex(IntEnum):
    """Left Dex3-1 hand joint indices (matches DDS message structure)."""
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6


class Dex3_1_Right_JointIndex(IntEnum):
    """Right Dex3-1 hand joint indices (matches DDS message structure).
    
    Note: Right hand has different finger order than left (index before middle).
    """
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


# Joint position limits (radians)
DEX3_LEFT_LOWER_LIMITS = np.array([-1.047, -0.724, 0.0, -1.57, -1.74, -1.57, -1.74], dtype=np.float32)
DEX3_LEFT_UPPER_LIMITS = np.array([1.047, 0.920, 1.74, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEX3_RIGHT_LOWER_LIMITS = np.array([-1.047, -0.920, -1.74, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEX3_RIGHT_UPPER_LIMITS = np.array([1.047, 0.724, 0.0, 1.57, 1.74, 1.57, 1.74], dtype=np.float32)


# URDF-compatible joint names for LeRobot interface
LEFT_HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]

# Note: Right hand has different order (thumb, INDEX, middle) to match Dex3_1_Right_JointIndex
RIGHT_HAND_JOINT_NAMES = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]

# Arm joint names (for IK and teleoperation)
LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
HAND_JOINT_NAMES = LEFT_HAND_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

