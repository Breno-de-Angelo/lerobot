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
ZMQCamera - Captures frames from remote cameras via ZeroMQ using JSON protocol in the
following legacy format:
    {
        "timestamps": {"camera_name": float},
        "images": {"camera_name": "<base64-jpeg>"}
    }
"""

import base64
import orjson as json
import logging
import time
from threading import Condition, Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
import zmq
from numpy.typing import NDArray

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_zmq import ZMQCameraConfig

logger = logging.getLogger(__name__)


_STREAMS_LOCK = Lock()
_STREAMS: dict[tuple[str, int], "_SharedZMQStream"] = {}

# De quanto em quanto tempo a thread de leitura acorda para checar o `stop_event`.
# Só afeta a latência do encerramento, não a espera por quadros (ver `_SharedZMQStream.start`).
_POLL_STOP_MS = 200


def _decode_zmq_images(parts: list[bytes]) -> dict[str, NDArray[Any]]:
    """Decode all images from one ZMQ message so RGB/depth share the same packet."""
    data = json.loads(parts[0])
    if "images" not in data:
        raise RuntimeError("invalid message: missing 'images' key")

    frames = {}
    protocol = data.get("protocol")
    for name, image_payload in data["images"].items():
        if protocol == "zmq.raw.v1":
            part_index = image_payload["part"]
            if part_index >= len(parts):
                raise RuntimeError(f"invalid raw message: missing image part {part_index}")
            frame = np.frombuffer(parts[part_index], dtype=np.dtype(image_payload["dtype"]))
            frames[name] = frame.reshape(image_payload["shape"]).copy()
            continue

        if protocol == "zmq.compressed.v1":
            part_index = image_payload["part"]
            if part_index >= len(parts):
                raise RuntimeError(f"invalid compressed message: missing image part {part_index}")
            flags = cv2.IMREAD_UNCHANGED if image_payload.get("encoding") == "png" else cv2.IMREAD_COLOR
            frame = cv2.imdecode(np.frombuffer(parts[part_index], np.uint8), flags)
            if frame is None:
                raise RuntimeError(f"failed to decode image '{name}'")
            frames[name] = frame
            continue

        img_bytes = base64.b64decode(image_payload)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to decode image '{name}'")
        frames[name] = frame

    return frames


class _SharedZMQStream:
    def __init__(self, server_address: str, port: int, timeout_ms: int):
        self.server_address = server_address
        self.port = port
        self.timeout_ms = timeout_ms
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.thread: Thread | None = None
        self.stop_event = Event()
        self.condition = Condition()
        self.latest_frames: dict[str, NDArray[Any]] = {}
        self.version = 0
        self.refcount = 0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # RCVTIMEO curto de propósito, e NÃO `self.timeout_ms`: aqui ele só decide de
        # quanto em quanto tempo o laço de leitura acorda para olhar o `stop_event`.
        # Quem define a espera por um quadro é `get_frame`, com deadline próprio sobre a
        # Condition — o socket ficar mudo por mais tempo não muda nada para o chamador.
        # Com o valor de configuração (até 10 s no dex3), o `stop()` desistia do join e
        # fechava o socket com esta thread ainda dentro do `recv_multipart`, o que
        # derruba o processo inteiro (ver `stop`).
        self.socket.setsockopt(zmq.RCVTIMEO, min(self.timeout_ms, _POLL_STOP_MS))
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.connect(f"tcp://{self.server_address}:{self.port}")
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Encerra o laço de leitura e só então libera socket e contexto.

        Sockets do zmq não são thread-safe: fechar um socket enquanto OUTRA thread está
        dentro de `recv_multipart` não é corrida benigna — a libzmq aborta o processo
        (`Fatal Python error: Aborted`, core dump). Era o que acontecia quando o
        servidor de imagem ficava mudo: a thread de leitura ficava presa no recv até o
        RCVTIMEO, o `join(timeout=2.0)` desistia antes disso e o `close()` vinha por
        cima. Numa sessão de teleoperação isso aparece como crash ao desligar, logo
        depois de a câmera cair — a hora em que menos se quer um core dump.

        Agora o join espera de verdade e, se ainda assim a thread não morrer, o socket
        NÃO é fechado: vazar um socket até o fim do processo é barato; abortar não.
        """
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

        if self.thread and self.thread.is_alive():
            # Folga sobre o RCVTIMEO do laço, que é o pior caso de espera lá dentro.
            self.thread.join(timeout=(_POLL_STOP_MS / 1000.0) + 2.0)
            if self.thread.is_alive():
                logger.warning(
                    f"ZMQ stream {self.server_address}:{self.port}: thread de leitura não "
                    f"encerrou; socket e contexto ficam abertos de propósito, para não "
                    f"abortar o processo."
                )
                self.thread = None
                return

        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None
        self.thread = None

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if self.socket is None:
                    break
                parts = self.socket.recv_multipart()
                while True:
                    try:
                        parts = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                frames = _decode_zmq_images(parts)
                with self.condition:
                    self.latest_frames = frames
                    self.version += 1
                    self.condition.notify_all()
            except zmq.Again:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.warning(f"ZMQ stream read error: {e}")

    def get_frame(self, camera_name: str, last_version: int, timeout_ms: float) -> tuple[NDArray[Any], int]:
        deadline = time.monotonic() + timeout_ms / 1000.0
        with self.condition:
            while self.version <= last_version or camera_name not in self.latest_frames:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"ZMQ stream {self.server_address}:{self.port} timeout after {timeout_ms}ms"
                    )
                self.condition.wait(timeout=remaining)
            return self.latest_frames[camera_name], self.version


class ZMQCamera(Camera):
    """
    Example usage:
        ```python
        from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

        config = ZMQCameraConfig(server_address="192.168.123.164", port=5555, camera_name="head_camera")
        camera = ZMQCamera(config)
        camera.connect()
        frame = camera.read()
        camera.disconnect()
        ```
    """

    def __init__(self, config: ZMQCameraConfig):
        super().__init__(config)

        self.config = config
        self.server_address = config.server_address
        self.port = config.port
        self.camera_name = config.camera_name
        self.color_mode = config.color_mode
        self.timeout_ms = config.timeout_ms

        self.stream: _SharedZMQStream | None = None
        self._connected = False
        self._last_version = 0

    def __str__(self) -> str:
        return f"ZMQCamera({self.camera_name}@{self.server_address}:{self.port})"

    @property
    def is_connected(self) -> bool:
        return self._connected and self.stream is not None

    def connect(self, warmup: bool = True) -> None:
        """Connect to ZMQ camera server."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        logger.info(f"Connecting to {self}...")

        try:
            key = (self.server_address, self.port)
            with _STREAMS_LOCK:
                stream = _STREAMS.get(key)
                if stream is None:
                    stream = _SharedZMQStream(self.server_address, self.port, self.timeout_ms)
                    _STREAMS[key] = stream
                stream.refcount += 1
                stream.start()
            self.stream = stream
            self._connected = True

            # Auto-detect resolution
            if self.width is None or self.height is None:
                h, w = self.read().shape[:2]
                self.height = h
                self.width = w
                logger.info(f"{self} resolution: {w}x{h}")

            logger.info(f"{self} connected.")

            if warmup:
                time.sleep(0.1)

        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to connect to {self}: {e}") from e

    def _cleanup(self):
        """Clean up ZMQ resources."""
        self._connected = False
        if self.stream is not None:
            key = (self.server_address, self.port)
            with _STREAMS_LOCK:
                self.stream.refcount -= 1
                if self.stream.refcount <= 0:
                    self.stream.stop()
                    _STREAMS.pop(key, None)
            self.stream = None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """ZMQ cameras require manual configuration (server address/port)."""
        return []

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Read a single frame from the ZMQ camera.

        Returns:
            np.ndarray: Decoded frame (height, width, 3)
        """
        if not self.is_connected or self.stream is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        frame, self._last_version = self.stream.get_frame(self.camera_name, self._last_version, self.timeout_ms)
        if frame.ndim == 2 and self.camera_name.endswith("depth"):
            frame = frame[:, :, None]
        return frame

    def _read_loop(self) -> None:
        return

    def _start_read_thread(self) -> None:
        return

    def _stop_read_thread(self) -> None:
        return

    def async_read(self, timeout_ms: float = 10000) -> NDArray[Any]:
        """Read latest frame asynchronously (non-blocking)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.stream is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        frame, self._last_version = self.stream.get_frame(self.camera_name, self._last_version, timeout_ms)
        if frame.ndim == 2 and self.camera_name.endswith("depth"):
            frame = frame[:, :, None]
        return frame

    def disconnect(self) -> None:
        """Disconnect from ZMQ camera."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        self._cleanup()
        logger.info(f"{self} disconnected.")
