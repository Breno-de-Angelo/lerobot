#!/usr/bin/env python
"""
RealSense server that publishes aligned RGB + depth over the ZMQCamera protocol.

Depth encoding matches the training pipeline (full_realsenser_server.py in lerobot-ext):
  1. Clip raw z16 depth (mm) to [0, 2000].
  2. Linearly rescale to uint8: depth8 = depth_clipped * 255 / 2000.
  3. Replicate the single channel to 3 channels (R=G=B=depth8) so the standard
     JPEG/H.264 pipeline works. The decoder (depth_to_pointcloud) takes channel
     0, reinterprets [0,1] as meters/2.0.

Publishes on a single ZMQ PUB socket. Clients (ZMQCamera) filter by camera_name:
  - "head_camera"       (RGB, HxWx3 uint8)
  - "head_camera_depth" (depth encoded as HxWx3 uint8 per above)

Usage (on the robot):
    python -m lerobot.cameras.zmq.realsense_server \
        --port 5555 --fps 30 --width 640 --height 480 \
        [--serial 327122071538]

If --serial is omitted, the first connected RealSense is used.
"""

import argparse
import base64
import contextlib
import json
import logging
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

logger = logging.getLogger(__name__)

DEPTH_CLIP_MM = 2000  # keep only the first 2 m (focuses on manipulation workspace)


def encode_image(image: np.ndarray, quality: int = 80) -> str:
    """Encode an RGB/uint8 image to a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf).decode("utf-8")


def depth_z16_to_3ch_uint8(depth_z16: np.ndarray) -> np.ndarray:
    """Encode raw z16 depth (mm) into the 3ch uint8 format used during data collection."""
    clipped = np.clip(depth_z16, 0, DEPTH_CLIP_MM)
    depth8 = (clipped.astype(np.float32) * (255.0 / DEPTH_CLIP_MM)).astype(np.uint8)
    return cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)


def run(port: int, fps: int, width: int, height: int, serial: str | None):
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    # Force constant FPS on the color sensor (avoids auto-exposure dropping to 15fps in low light).
    for sensor in profile.get_device().query_sensors():
        if sensor.supports(rs.option.auto_exposure_priority):
            sensor.set_option(rs.option.auto_exposure_priority, 0)

    align = rs.align(rs.stream.color)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 20)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://*:{port}")
    logger.info(f"RealSenseServer publishing RGB+depth on port {port} at {fps} fps")

    frame_times: deque[float] = deque(maxlen=60)
    frame_count = 0
    try:
        while True:
            t0 = time.time()
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue

            rgb = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2RGB)
            depth_3ch = depth_z16_to_3ch_uint8(np.asanyarray(depth.get_data()))

            now = time.time()
            message = {
                "timestamps": {"head_camera": now, "head_camera_depth": now},
                "images": {
                    "head_camera": encode_image(rgb),
                    "head_camera_depth": encode_image(depth_3ch),
                },
            }
            with contextlib.suppress(zmq.Again):
                sock.send_string(json.dumps(message), zmq.NOBLOCK)

            frame_count += 1
            frame_times.append(time.time() - t0)
            if frame_count % 60 == 0:
                measured = len(frame_times) / sum(frame_times) if frame_times else 0.0
                logger.debug(f"publishing at {measured:.1f} fps")

            sleep = (1.0 / fps) - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        sock.close()
        ctx.term()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial (optional)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.port, args.fps, args.width, args.height, args.serial)


if __name__ == "__main__":
    main()
