import logging
import pygame
from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.processor import RobotAction

logger = logging.getLogger(__name__)

@TeleoperatorConfig.register_subclass("keyboard_g1_arms")
@dataclass
class KeyboardConfig(TeleoperatorConfig):
    speed: float = 0.02  # Sensibilidade
    fps: int = 60

class KeyboardTeleoperator(Teleoperator):
    config_class = KeyboardConfig
    name = "keyboard_g1_arms" 

    def __init__(self, config: KeyboardConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.screen = None
        
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
        
        # O Pygame precisa de uma janela (display) para capturar inputs do teclado.
        # Criamos uma janela pequena e damos um nome a ela.
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Controle G1 Arm - Mantenha o foco aqui")
        
        logger.info("Controle por teclado ativado. Mantenha a janela do Pygame em foco.")
        self._is_connected = True

    def disconnect(self) -> None:
        if self._is_connected:
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

    def get_action(self) -> RobotAction:
        if not self._is_connected:
             raise ConnectionError("Teleoperador por Teclado não está conectado.")
        
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # ================= MAPEAMENTO DO BRAÇO ESQUERDO =================
        # Analógico Esquerdo (W/A/S/D)
        ls_x = keys[pygame.K_d] - keys[pygame.K_a]
        ls_y = keys[pygame.K_s] - keys[pygame.K_w]
        
        # D-Pad (Cotovelo: R/F | Rotação: Q/E)
        dpad_x = keys[pygame.K_e] - keys[pygame.K_q]
        dpad_y = keys[pygame.K_r] - keys[pygame.K_f]
        
        # Modificador de Pulso Esquerdo (LSHIFT)
        lb = keys[pygame.K_LSHIFT]
        
        # Gatilho Mão Esquerda (Z)
        left_hand_close = keys[pygame.K_z]


        # ================= MAPEAMENTO DO BRAÇO DIREITO =================
        # Analógico Direito (I/J/K/L)
        rs_x = keys[pygame.K_l] - keys[pygame.K_j]
        rs_y = keys[pygame.K_k] - keys[pygame.K_i]
        
        # Botões Y/A/X/B (Cotovelo: U/J | Rotação: O/H -> Adaptado para Y/H e O/U)
        btn_y = keys[pygame.K_y] # Cotovelo -
        btn_a = keys[pygame.K_h] # Cotovelo +
        btn_x = keys[pygame.K_u] # Rotação -
        btn_b = keys[pygame.K_o] # Rotação +
        
        # Modificador de Pulso Direito (RSHIFT)
        rb = keys[pygame.K_RSHIFT]
        
        # Gatilho Mão Direita (M)
        right_hand_close = keys[pygame.K_m]


        # ================= CONTROLE DO BRAÇO ESQUERDO =================
        if lb:
            # Se LSHIFT está pressionado, W/A/S/D controla o Pulso
            self.body_joints["kLeftWristPitch.q"] += ls_y * self.config.speed
            self.body_joints["kLeftWristRoll.q"]  += ls_x * self.config.speed
        else:
            # Padrão: controla o Ombro
            self.body_joints["kLeftShoulderPitch.q"] += ls_y * self.config.speed
            self.body_joints["kLeftShoulderRoll.q"]  += ls_x * self.config.speed
        
        # Cotovelo e Rotação (Yaw)
        self.body_joints["kLeftElbow.q"]       += dpad_y * self.config.speed
        self.body_joints["kLeftShoulderYaw.q"] += dpad_x * self.config.speed


        # ================= CONTROLE DO BRAÇO DIREITO =================
        if rb:
            # Se RSHIFT está pressionado, I/J/K/L controla o Pulso
            self.body_joints["kRightWristPitch.q"] += rs_y * self.config.speed
            self.body_joints["kRightWristRoll.q"]  += rs_x * self.config.speed
        else:
            # Padrão: controla o Ombro
            self.body_joints["kRightShoulderPitch.q"] += rs_y * self.config.speed
            self.body_joints["kRightShoulderRoll.q"]  += rs_x * self.config.speed
        
        # Cotovelo e Rotação (Yaw) Direito
        if btn_y: self.body_joints["kRightElbow.q"] -= self.config.speed
        if btn_a: self.body_joints["kRightElbow.q"] += self.config.speed
        if btn_x: self.body_joints["kRightShoulderYaw.q"] -= self.config.speed
        if btn_b: self.body_joints["kRightShoulderYaw.q"] += self.config.speed


        # ================= CONTROLE DAS MÃOS (Dex3) =================
        hand_speed = self.config.speed * 2  # Mãos podem fechar um pouco mais rápido
        max_hand_val = 1.0  # Mude para 100.0 ou 1.57 se a junta exigir outra escala
        
        # Lógica para Mão Esquerda
        for name in self._left_hand_names:
            key = f"{name}.q"
            if left_hand_close:
                self.hand_joints[key] = min(self.hand_joints[key] + hand_speed, max_hand_val)
            else:
                self.hand_joints[key] = max(self.hand_joints[key] - hand_speed, 0.0)
                
        # Lógica para Mão Direita
        for name in self._right_hand_names:
            key = f"{name}.q"
            if right_hand_close:
                self.hand_joints[key] = min(self.hand_joints[key] + hand_speed, max_hand_val)
            else:
                self.hand_joints[key] = max(self.hand_joints[key] - hand_speed, 0.0)

        # Concatena e envia
        action_data = {**self.body_joints, **self.hand_joints}
        
        # OPCIONAL: Se o seu processador exigir chaves de trigger/pinch ao invés de juntas diretas
        # action_data["left_pinch_value"] = self.hand_joints[f"{self._left_hand_names[0]}.q"] * 100.0
        # action_data["right_pinch_value"] = self.hand_joints[f"{self._right_hand_names[0]}.q"] * 100.0
        
        return action_data