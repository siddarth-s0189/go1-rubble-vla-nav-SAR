# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""OmniVLA bridge for Go1 SAR low-level pilot.

Sends FPV frame + strategic_instruction (from Gemini VLM) to an OmniVLA
inference endpoint. See https://omnivla-nav.github.io/
"""

import base64
import json
from typing import Tuple

import cv2
import jax.numpy as jp
import numpy as np

try:
    import requests
except ImportError:
    requests = None


def _encode_frame_rgb_to_jpeg_b64(image: np.ndarray) -> str:
    """Encode RGB numpy (H,W,3) uint8 to base64 JPEG."""
    if image.shape[-1] != 3 or image.ndim != 3:
        raise ValueError(f"Expected RGB (H,W,3), got {getattr(image, 'shape', '?')}")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# Zero-action fallback when server is down or request fails
_ZERO_RAW = np.array([0.0, 0.0, 0.0], dtype=np.float32)
_ZERO_SCALED = jp.array([0.0, 0.0, 0.0], dtype=jp.float32)


class OmniVLABridge:
    """Bridge to OmniVLA inference server: FPV + language → [forward, lateral, yaw]."""

    def __init__(self, model_url: str = "http://localhost:8000") -> None:
        self.model_url = model_url.rstrip("/")
        if requests is None:
            print("WARNING: 'requests' not installed. OmniVLA bridge will return zero actions. pip install requests")
        else:
            print(f"OmniVLA Pilot bridge: {self.model_url}")

    def get_vla_action(
        self,
        fpv_frame: np.ndarray,
        strategic_instruction: str,
    ) -> Tuple[jp.ndarray, np.ndarray]:
        """Get low-level [forward, lateral, yaw] from OmniVLA.

        POSTs FPV image (base64 JPEG) and strategic_instruction to the
        inference endpoint. On failure or server down, returns zero action.

        Returns:
            scaled_action: jax.numpy array of 3 floats (clipped to safe range).
            raw_action: numpy array of 3 floats (model output before scaling).
        """
        if requests is None:
            return _ZERO_SCALED, _ZERO_RAW.copy()

        try:
            image_b64 = _encode_frame_rgb_to_jpeg_b64(fpv_frame)
        except Exception as e:
            print(f"OmniVLA: frame encode failed: {e}")
            return _ZERO_SCALED, _ZERO_RAW.copy()

        # REST contract: exactly these keys for OmniVLA /predict endpoint
        payload = {
            "image": image_b64,
            "strategic_instruction": strategic_instruction,
        }
        headers = {"Content-Type": "application/json"}

        try:
            # timeout=5.0 prevents simulation from hanging if OmniVLA server lags
            r = requests.post(
                f"{self.model_url}/predict",
                json=payload,
                headers=headers,
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.RequestException as e:
            print(f"OmniVLA: request failed (zero action): {e}")
            return _ZERO_SCALED, _ZERO_RAW.copy()
        except (json.JSONDecodeError, KeyError) as e:
            print(f"OmniVLA: bad response (zero action): {e}")
            return _ZERO_SCALED, _ZERO_RAW.copy()

        # Parse [forward, lateral, yaw] from response (list or dict)
        raw_action = _parse_action(data)
        if raw_action is None:
            return _ZERO_SCALED, _ZERO_RAW.copy()

        raw_action = np.asarray(raw_action, dtype=np.float32).flatten()
        if raw_action.size < 3:
            raw_action = np.resize(raw_action, 3)
        raw_action = raw_action[:3].copy()

        print(f"OmniVLA raw_action: [forward={raw_action[0]:.4f}, lateral={raw_action[1]:.4f}, yaw={raw_action[2]:.4f}]")

        # NOTE: If OmniVLA outputs are near zero (e.g. 0.0002), change multiplier from 1.0 to 250.0.
        scaled = np.clip(raw_action * 1.0, -2.0, 2.0)
        scaled_action = jp.array(scaled, dtype=jp.float32)
        return scaled_action, raw_action


def _parse_action(data: dict | list) -> list | None:
    """Extract [forward, lateral, yaw] from API response. Handles action nested under 'action' or 'actions'."""
    def to_triple(val):
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return [float(val[0]), float(val[1]), float(val[2])]
        return None

    if isinstance(data, list) and len(data) >= 3:
        return to_triple(data)
    if isinstance(data, dict):
        for key in ("action", "actions", "output"):
            if key in data:
                out = to_triple(data[key])
                if out is not None:
                    return out
        if "forward" in data and "lateral" in data and "yaw" in data:
            return [
                float(data["forward"]),
                float(data["lateral"]),
                float(data["yaw"]),
            ]
    return None
