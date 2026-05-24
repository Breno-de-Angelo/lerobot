#!/usr/bin/env python
"""
Unitree G1 Dex3 Robot - Unified DDS Architecture

This implementation uses the same DDS threading pattern as the body/arm code,
eliminating the multiprocessing complexity of the original Dex3_1_Controller.
"""

from dataclasses import dataclass, field
import logging
import threading
import time
import numpy as np
from functools import cached_property

from .unitree_g1 import UnitreeG1, UnitreeG1Config
from ..config import RobotConfig
from .g1_utils import (
    Dex3_1_Left_JointIndex, 
    Dex3_1_Right_JointIndex,
    Dex3_Num_Motors,
    DEX3_LEFT_LOWER_LIMITS,
    DEX3_LEFT_UPPER_LIMITS,
    DEX3_RIGHT_LOWER_LIMITS,
    DEX3_RIGHT_UPPER_LIMITS,
    kTopicDex3LeftCommand,
    kTopicDex3RightCommand,
    kTopicDex3LeftState,
    kTopicDex3RightState,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
)

from lerobot.processor import RobotAction, RobotObservation

logger = logging.getLogger(__name__)


def _flatten_pressure_to_33(press_sensor_state: list) -> np.ndarray:
    """Flatten Dex3 tactile sensors into a 33-dim float32 vector.

    Matches the data-collection convention (lerobot-ext/robot/unitree_g1/unitree_g1_dex3.py):
    concatenate `pressure` across all sensors, pad with zeros if short, truncate to 33.
    """
    flat: list[float] = []
    for sensor in press_sensor_state or []:
        pressure = sensor.get("pressure", []) if isinstance(sensor, dict) else []
        flat.extend(float(x) for x in pressure)
    if len(flat) < 33:
        flat.extend([0.0] * (33 - len(flat)))
    return np.asarray(flat[:33], dtype=np.float32)


@dataclass
class HandMotorState:
    """State of a single hand motor."""
    q: float = 0.0  # position


@dataclass
class HandState:
    """State of a single hand (7 motors for Dex3-1)."""
    motor_state: list[HandMotorState] = field(
        default_factory=lambda: [HandMotorState() for _ in range(Dex3_Num_Motors)]
    )


@RobotConfig.register_subclass("unitree_g1_dex3")
@dataclass
class UnitreeG1Dex3Config(UnitreeG1Config):
    """Configuration for Unitree G1 with Dex3-1 hands."""
    hand_kp: float = 1.5  # Position gain for hand motors
    hand_kd: float = 0.2  # Damping gain for hand motors
    hand_control_dt: float = 0.01  # 100 Hz control loop
    
    def __post_init__(self):
        # Add default ZMQ camera if no cameras specified.
        # Keep the historic "cam_rgb_high" dict key for backwards compatibility with
        # the ACT policy (trained dataset uses that exact observation name). Policies
        # that need a different key (e.g. pi05's "head_camera") remap at the runtime
        # layer instead of mutating the driver default.
        if not self.cameras and not self.is_simulation:
            from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
            self.cameras = {
                "cam_rgb_high": ZMQCameraConfig(
                    server_address=self.robot_ip,
                    port=5555,
                    camera_name="head_camera",
                    width=640,
                    height=480,
                ),
            }


class UnitreeG1Dex3(UnitreeG1):
    """
    Unitree G1 Robot with Dex3-1 Dexterous Hands.
    
    Uses the same DDS threading architecture as the body motors:
    - A background thread subscribes to hand state at 100Hz
    - Hand commands are published directly via DDS
    """
    config_class = UnitreeG1Dex3Config
    name = "unitree_g1_dex3"

    def __init__(self, config: UnitreeG1Dex3Config):
        super().__init__(config)
        
        # Hand state (similar to _lowstate for body)
        self._left_hand_state: HandState | None = None
        self._right_hand_state: HandState | None = None

        # Tactile pressure (33-dim per hand). Matches the dataset encoding:
        # flatten press_sensor_state[*].pressure across all sensors, pad/truncate to 33.
        # See init_lerobot_record_v2.py + robot/unitree_g1/unitree_g1_dex3.py in lerobot-ext.
        self._left_hand_pressure: np.ndarray = np.zeros(33, dtype=np.float32)
        self._right_hand_pressure: np.ndarray = np.zeros(33, dtype=np.float32)
        
        # Threading control
        self._hand_shutdown_event = threading.Event()
        self._hand_subscribe_thread: threading.Thread | None = None
        
        # DDS publishers/subscribers (initialized in connect)
        self._left_hand_cmd_pub = None
        self._right_hand_cmd_pub = None
        self._left_hand_state_sub = None
        self._right_hand_state_sub = None
        
        # Command messages (initialized in connect)
        self._left_hand_msg = None
        self._right_hand_msg = None
        
        # Use joint name constants from g1_utils
        self.left_hand_joint_names = LEFT_HAND_JOINT_NAMES
        self.right_hand_joint_names = RIGHT_HAND_JOINT_NAMES

    def _subscribe_hand_state(self):
        """
        Background thread that polls hand state via DDS at ~100Hz.
        Similar to _subscribe_motor_state() in UnitreeG1.
        """
        while not self._hand_shutdown_event.is_set():
            start_time = time.time()
            
            # Read left hand state
            left_msg = self._left_hand_state_sub.Read()
            if left_msg is not None:
                left_state = HandState()
                for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
                    left_state.motor_state[idx].q = left_msg.motor_state[joint_id].q
                self._left_hand_state = left_state
                self._left_hand_pressure = _flatten_pressure_to_33(left_msg.press_sensor_state)

            # Read right hand state
            right_msg = self._right_hand_state_sub.Read()
            if right_msg is not None:
                right_state = HandState()
                for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
                    right_state.motor_state[idx].q = right_msg.motor_state[joint_id].q
                self._right_hand_state = right_state
                self._right_hand_pressure = _flatten_pressure_to_33(right_msg.press_sensor_state)
            
            # Maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.hand_control_dt - elapsed)
            time.sleep(sleep_time)

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot body and hands."""
        # Connect body first
        super().connect(calibrate=calibrate)
        
        # Skip hand connection in simulation mode
        if self.config.is_simulation:
            logger.info("Simulation mode: Skipping Dex3 hand connection.")
            return
        
        # Use ZMQ-based hand communication (SDK-free, no circular import issues)
        # The server on the robot bridges hand DDS topics to ZMQ sockets
        from lerobot.robots.unitree_g1.unitree_sdk2_socket import (
            ChannelSubscriber,
            ChannelPublisher,
            HandCmdMsg,
            kTopicDex3LeftCommand,
            kTopicDex3RightCommand,
            kTopicDex3LeftState,
            kTopicDex3RightState,
        )
        
        # Initialize hand state subscribers (ZMQ-based)
        self._left_hand_state_sub = ChannelSubscriber(kTopicDex3LeftState, None)
        self._left_hand_state_sub.Init()
        self._right_hand_state_sub = ChannelSubscriber(kTopicDex3RightState, None)
        self._right_hand_state_sub.Init()
        
        # Initialize hand command publishers (ZMQ-based)
        self._left_hand_cmd_pub = ChannelPublisher(kTopicDex3LeftCommand, None)
        self._left_hand_cmd_pub.Init()
        self._right_hand_cmd_pub = ChannelPublisher(kTopicDex3RightCommand, None)
        self._right_hand_cmd_pub.Init()
        
        # Initialize command messages with default gains
        self._left_hand_msg = HandCmdMsg()
        self._right_hand_msg = HandCmdMsg()
        
        kp = self.config.hand_kp
        kd = self.config.hand_kd
        
        for joint_id in Dex3_1_Left_JointIndex:
            # Mode byte: bits 0-3 = id, bits 4-6 = status (0x01 = enabled), bit 7 = timeout
            mode = (joint_id & 0x0F) | (0x01 << 4)
            self._left_hand_msg.motor_cmd[joint_id].mode = mode
            self._left_hand_msg.motor_cmd[joint_id].kp = kp
            self._left_hand_msg.motor_cmd[joint_id].kd = kd
            
        for joint_id in Dex3_1_Right_JointIndex:
            mode = (joint_id & 0x0F) | (0x01 << 4)
            self._right_hand_msg.motor_cmd[joint_id].mode = mode
            self._right_hand_msg.motor_cmd[joint_id].kp = kp
            self._right_hand_msg.motor_cmd[joint_id].kd = kd
        
        # Start hand state subscription thread
        self._hand_subscribe_thread = threading.Thread(
            target=self._subscribe_hand_state, 
            daemon=True,
            name="Dex3HandStateSubscriber"
        )
        self._hand_subscribe_thread.start()
        
        # Wait for first hand state
        timeout = 3.0
        start = time.time()
        while self._left_hand_state is None or self._right_hand_state is None:
            if time.time() - start > timeout:
                logger.warning("Timeout waiting for Dex3 hand state. Hands may not be connected.")
                break
            time.sleep(0.01)
        
        if self._left_hand_state is not None and self._right_hand_state is not None:
            logger.info("Connected to Dex3 Hands via ZMQ.")
        else:
            logger.warning("Dex3 Hands not fully connected - hand state unavailable.")

    def reset_hands(self, default_positions: list[float] | None = None, duration_s: float = 2.0):
        """Move both hands smoothly to a target pose (default: q=0, fully open).

        Backport from lerobot-ext's data-collection driver. Used at the start of
        inference to align the physical hand pose with the dataset's typical
        initial state (q≈0) before the policy starts predicting — otherwise the
        first chunk's command jumps from current state to ±1.5 in one frame and
        the motor "espasms".

        Args:
            default_positions: 14-element list, [left_7, right_7]. If None, opens
                both hands fully (q=0 each joint).
            duration_s: total interpolation time (default 2 s, at hand_control_dt).
        """
        if default_positions is None:
            default_left = np.zeros(Dex3_Num_Motors, dtype=np.float32)
            default_right = np.zeros(Dex3_Num_Motors, dtype=np.float32)
        else:
            default_left = np.asarray(default_positions[:Dex3_Num_Motors], dtype=np.float32)
            default_right = np.asarray(default_positions[Dex3_Num_Motors:], dtype=np.float32)

        dt = self.config.hand_control_dt
        steps = max(1, int(duration_s / dt))

        # Snapshot current q (from subscribed state)
        left_current = np.zeros(Dex3_Num_Motors, dtype=np.float32)
        right_current = np.zeros(Dex3_Num_Motors, dtype=np.float32)
        if self._left_hand_state is not None:
            left_current = np.array([s.q for s in self._left_hand_state.motor_state], dtype=np.float32)
        if self._right_hand_state is not None:
            right_current = np.array([s.q for s in self._right_hand_state.motor_state], dtype=np.float32)

        logger.info(f"reset_hands: interpolating L {left_current.round(2).tolist()} -> {default_left.round(2).tolist()} ({duration_s:.1f}s)")
        for step in range(steps):
            alpha = (step + 1) / steps  # 1/steps .. 1.0
            left_q = left_current * (1 - alpha) + default_left * alpha
            right_q = right_current * (1 - alpha) + default_right * alpha

            action: dict[str, float] = {}
            for i, name in enumerate(self.left_hand_joint_names):
                action[f"{name}.q"] = float(left_q[i])
            for i, name in enumerate(self.right_hand_joint_names):
                action[f"{name}.q"] = float(right_q[i])
            self.send_action(action)
            time.sleep(dt)
        logger.info("reset_hands: done")

    def disconnect(self):
        """Disconnect from robot body and hands.

        Before terminating the ZMQ link, put every hand motor into Limp Mode
        (mode=0, kp=0, kd=0, tau=0) so the fingers go completely soft instead of
        staying locked on the last command. Published 5x with 10ms gap so the
        Dex3 board has a chance to receive even if a packet drops.
        """
        # 1. Limp Mode for hands so they release before we tear down ZMQ.
        if (
            self._left_hand_cmd_pub is not None
            and self._right_hand_cmd_pub is not None
            and self._left_hand_msg is not None
            and self._right_hand_msg is not None
        ):
            try:
                logger.info("disconnect: putting hands into Limp Mode (kp=kd=tau=0, mode=0)...")
                for joint_id in Dex3_1_Left_JointIndex:
                    self._left_hand_msg.motor_cmd[joint_id].mode = (joint_id & 0x0F) | (0x00 << 4)
                    self._left_hand_msg.motor_cmd[joint_id].kp = 0.0
                    self._left_hand_msg.motor_cmd[joint_id].kd = 0.0
                    self._left_hand_msg.motor_cmd[joint_id].tau = 0.0
                for joint_id in Dex3_1_Right_JointIndex:
                    self._right_hand_msg.motor_cmd[joint_id].mode = (joint_id & 0x0F) | (0x00 << 4)
                    self._right_hand_msg.motor_cmd[joint_id].kp = 0.0
                    self._right_hand_msg.motor_cmd[joint_id].kd = 0.0
                    self._right_hand_msg.motor_cmd[joint_id].tau = 0.0
                for _ in range(5):
                    self._left_hand_cmd_pub.Write(self._left_hand_msg)
                    self._right_hand_cmd_pub.Write(self._right_hand_msg)
                    time.sleep(0.01)
            except Exception as e:
                logger.warning(f"disconnect: Limp Mode write raised: {e}")

        # 2. Signal hand thread to stop
        self._hand_shutdown_event.set()

        # Wait for hand thread to finish
        if self._hand_subscribe_thread is not None:
            self._hand_subscribe_thread.join(timeout=2.0)
            if self._hand_subscribe_thread.is_alive():
                logger.warning("Hand subscribe thread did not stop cleanly")

        # 3. Disconnect body
        super().disconnect()

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action space: body joints (based on control_mode) + hand joints.
        
        - full_body mode: 29 body + 14 hand = 43 joints
        - upper_body mode: 14 arm + 14 hand = 28 joints
        """
        features = super().action_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation space: body joints + hand joints + tactile pressure.

        - full_body mode: 29 body + 14 hand + 66 pressure
        - upper_body mode: 14 arm + 14 hand + 66 pressure
        """
        features = super().observation_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float
        features["left_hand_pressure"] = (33,)
        features["right_hand_pressure"] = (33,)
        return features

    def get_observation(self) -> RobotObservation:
        """Get observation including hand joint positions and tactile pressure."""
        obs = super().get_observation()

        # Add left hand state (default to 0.0 if hands not connected)
        for i, name in enumerate(self.left_hand_joint_names):
            if self._left_hand_state is not None:
                obs[f"{name}.q"] = float(self._left_hand_state.motor_state[i].q)
            else:
                obs[f"{name}.q"] = 0.0  # Default when hands not available

        # Add right hand state (default to 0.0 if hands not connected)
        for i, name in enumerate(self.right_hand_joint_names):
            if self._right_hand_state is not None:
                obs[f"{name}.q"] = float(self._right_hand_state.motor_state[i].q)
            else:
                obs[f"{name}.q"] = 0.0  # Default when hands not available

        # Tactile pressure per hand (33-dim float32). Consumers that don't need it
        # can ignore these keys; the trained pi05-D injector pulls them by name.
        obs["left_hand_pressure"] = self._left_hand_pressure.copy()
        obs["right_hand_pressure"] = self._right_hand_pressure.copy()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action to robot including hand commands."""
        # Send body action
        super().send_action(action)
        
        # Check if we have hand publishers
        if self._left_hand_cmd_pub is None or self._right_hand_cmd_pub is None:
            return action
        
        # Check if action contains hand commands
        first_joint = self.left_hand_joint_names[0]
        if f"{first_joint}.q" not in action:
            return action
        
        # Extract and clamp left hand targets
        left_q = np.zeros(Dex3_Num_Motors)
        for i, name in enumerate(self.left_hand_joint_names):
            left_q[i] = action.get(f"{name}.q", 0.0)
        left_q = np.clip(left_q, DEX3_LEFT_LOWER_LIMITS, DEX3_LEFT_UPPER_LIMITS)
        
        # Extract and clamp right hand targets
        right_q = np.zeros(Dex3_Num_Motors)
        for i, name in enumerate(self.right_hand_joint_names):
            right_q[i] = action.get(f"{name}.q", 0.0)
        right_q = np.clip(right_q, DEX3_RIGHT_LOWER_LIMITS, DEX3_RIGHT_UPPER_LIMITS)
        
        # Update command messages
        for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
            self._left_hand_msg.motor_cmd[joint_id].q = left_q[idx]
        for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
            self._right_hand_msg.motor_cmd[joint_id].q = right_q[idx]
        
        # Publish commands
        self._left_hand_cmd_pub.Write(self._left_hand_msg)
        self._right_hand_cmd_pub.Write(self._right_hand_msg)
        
        return action
