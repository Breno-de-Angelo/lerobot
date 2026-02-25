import logging
import numpy as np
import pygame
from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.processor import RobotAction

logger = logging.getLogger(__name__)

@TeleoperatorConfig.register_subclass("joystick_g1_Arms")
@dataclass
class JoyConfig(TeleoperatorConfig):
    joystick_id: int = 0
    speed: float = 0.02  # Sensibilidade (reduzida um pouco para mais precisão)
    deadzone: float = 0.1
    fps: int = 60

class JoyTeleoperator(Teleoperator):
    config_class = JoyConfig
    name = "joystick_g1_Arms" 

    def __init__(self, config: JoyConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.joystick = None
        
        from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex, LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES
        
        self._left_hand_names = LEFT_HAND_JOINT_NAMES
        self._right_hand_names = RIGHT_HAND_JOINT_NAMES

        # Dicionários que guardam a posição atual de cada motor
        self.body_joints = {f"{motor.name}.q": 0.0 for motor in G1_29_JointIndex}
        self.hand_joints = {f"{name}.q": 0.0 for name in self._left_hand_names + self._right_hand_names}

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise ConnectionError("Nenhum joystick foi detectado.")

        self.joystick = pygame.joystick.Joystick(self.config.joystick_id)
        self.joystick.init()
        logger.info(f"Joystick conectado: {self.joystick.get_name()}")
        self._is_connected = True

    def disconnect(self) -> None:
        if self._is_connected:
            if self.joystick:
                self.joystick.quit()
            pygame.quit()
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass
    
    @property
    def action_features(self) -> dict:
        features = {}
        for key in self.body_joints.keys():
            features[key] = float
        for key in self.hand_joints.keys():
            features[key] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.config.deadzone:
            return 0.0
        return value

    def get_action(self) -> RobotAction:
        if not self._is_connected:
             raise ConnectionError("Teleoperador Joystick não está conectado.")
        
        pygame.event.pump()

        # LEITURA DOS ANALÓGICOS (Eixos)
        ls_x = self._apply_deadzone(self.joystick.get_axis(0))
        ls_y = self._apply_deadzone(self.joystick.get_axis(1))
        rs_x = self._apply_deadzone(self.joystick.get_axis(3))
        rs_y = self._apply_deadzone(self.joystick.get_axis(4))
        
        # LEITURA DOS GATILHOS (No Xbox, costumam ser eixos 2 e 5. Vão de -1 a 1)
        lt = self.joystick.get_axis(2) if self.joystick.get_numaxes() > 2 else -1.0
        rt = self.joystick.get_axis(5) if self.joystick.get_numaxes() > 5 else -1.0

        # LEITURA DOS BOTÕES LB E RB (Para trocar o modo dos analógicos para o Pulso)
        lb = self.joystick.get_button(4)
        rb = self.joystick.get_button(5)

        # LEITURA DO D-PAD (Setinhas) para o Cotovelo e Rotação do braço Esquerdo
        hat = self.joystick.get_hat(0) if self.joystick.get_numhats() > 0 else (0, 0)
        dpad_x, dpad_y = hat  # dpad_y: 1 é cima, -1 é baixo. dpad_x: 1 é dir, -1 é esq.

        # LEITURA DOS BOTÕES ABXY para o Cotovelo e Rotação do braço Direito
        # Padrão Xbox Pygame: A=0, B=1, X=2, Y=3
        btn_a = self.joystick.get_button(0)
        btn_b = self.joystick.get_button(1)
        btn_x = self.joystick.get_button(2)
        btn_y = self.joystick.get_button(3)


        # ================= CONTROLE DO BRAÇO ESQUERDO =================
        if lb:
            # Se LB está pressionado, o analógico controla o Pulso
            self.body_joints["kLeftWristPitch.q"] += ls_y * self.config.speed
            self.body_joints["kLeftWristRoll.q"]  += ls_x * self.config.speed
        else:
            # Padrão: controla o Ombro
            self.body_joints["kLeftShoulderPitch.q"] += ls_y * self.config.speed
            self.body_joints["kLeftShoulderRoll.q"]  += ls_x * self.config.speed
        
        # D-Pad controla o Cotovelo e Rotação (Yaw)
        self.body_joints["kLeftElbow.q"]       += dpad_y * self.config.speed
        self.body_joints["kLeftShoulderYaw.q"] += dpad_x * self.config.speed


        # ================= CONTROLE DO BRAÇO DIREITO =================
        if rb:
            # Se RB está pressionado, o analógico controla o Pulso
            self.body_joints["kRightWristPitch.q"] += rs_y * self.config.speed
            self.body_joints["kRightWristRoll.q"]  += rs_x * self.config.speed
        else:
            # Padrão: controla o Ombro
            self.body_joints["kRightShoulderPitch.q"] += rs_y * self.config.speed
            self.body_joints["kRightShoulderRoll.q"]  += rs_x * self.config.speed
        
        # Botões Y/A controlam Cotovelo, X/B controlam Rotação (Yaw)
        if btn_y: self.body_joints["kRightElbow.q"] -= self.config.speed
        if btn_a: self.body_joints["kRightElbow.q"] += self.config.speed
        if btn_x: self.body_joints["kRightShoulderYaw.q"] -= self.config.speed
        if btn_b: self.body_joints["kRightShoulderYaw.q"] += self.config.speed


        # ================= CONTROLE DAS MÃOS (Dex3) =================
        # Gatilhos vão de -1.0 (solto) a 1.0 (apertado). Se passar de 0.0, fecha a mão.
        left_hand_val = 1.0 if lt > 0.0 else 0.0
        right_hand_val = 1.0 if rt > 0.0 else 0.0
        
        for name in self._left_hand_names:
            self.hand_joints[f"{name}.q"] = left_hand_val
        for name in self._right_hand_names:
            self.hand_joints[f"{name}.q"] = right_hand_val

        # Concatena e envia
        action_data = {**self.body_joints, **self.hand_joints}
        return action_data