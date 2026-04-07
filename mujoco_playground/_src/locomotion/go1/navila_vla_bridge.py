# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under this license.
# ==============================================================================
"""Local NaVILA bridge: research-exact prompting + mid-level to command conversion.

Uses the EXACT prompt format from NaVILA research (run_navigation.py, navila_trainer.py):
  "Imagine you are a robot programmed for navigation tasks. You have been given ...
   Your assigned task is: \"{instruction}\" Analyze this series of images to decide
   your next action, which could be turning left or right by a specific degree,
   moving forward a certain distance, or stop if the task is completed."

The VLA outputs natural language mid-level instructions (e.g. "The next action is
move forward 50 cm", "turn left 30 degree"). We parse these and convert to
continuous [vx, vy, yaw] for our PPO policy via midlevel_to_command().

NaVILA VLN-CE uses max_new_tokens=32 (outputs are short phrases, ~10 tokens).
"""

from __future__ import annotations

import copy
import os
import re
import sys
import traceback
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
  import torch
  from transformers import AutoModel, AutoProcessor
  from transformers import BitsAndBytesConfig
except Exception as _navila_import_err:
  torch = None
  AutoModel = None
  AutoProcessor = None
  BitsAndBytesConfig = None


def _use_llava_loader(model_name: str) -> bool:
  """True if model requires NaVILA/llava loader (a8cheng/navila-*)."""
  return "a8cheng/navila" in (model_name or "").lower()


def _ensure_navila_in_path() -> None:
  """Add NaVILA to sys.path so llava imports work."""
  bridge_dir = Path(__file__).resolve().parent
  # go1 -> locomotion -> _src -> mujoco_playground pkg -> workspace (4 levels)
  workspace_root = bridge_dir.parent.parent.parent.parent
  navila_path = workspace_root / "NaVILA"
  if navila_path.exists() and str(navila_path) not in sys.path:
    sys.path.insert(0, str(navila_path))


def _sample_and_pad_images(
    images: list,
    num_frames: int = 64,
    width: int = 448,
    height: int = 448,
) -> list:
  """NaVILA frame sampling: pad with black if needed, uniformly sample + latest."""
  frames = copy.deepcopy(images)
  if len(frames) < num_frames:
    while len(frames) < num_frames:
      frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
  latest = frames[-1]
  sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
  sampled = [frames[i] for i in sampled_indices] + [latest]
  return [f.convert("RGB") if hasattr(f, "convert") else f for f in sampled]


# NaVILA VLN-CE action patterns (from navila_trainer.py)
_STOP_RE = re.compile(r"\bstop\b", re.IGNORECASE)
_FWD_RE = re.compile(r"\b(?:is\s+)?move forward\b", re.IGNORECASE)
_LEFT_RE = re.compile(r"\b(?:is\s+)?turn left\b", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\b(?:is\s+)?turn right\b", re.IGNORECASE)
_FWD_CM_RE = re.compile(r"move forward\s+(\d+)\s*cm", re.IGNORECASE)
_LEFT_DEG_RE = re.compile(r"turn left\s+(\d+)\s*degree", re.IGNORECASE)
_RIGHT_DEG_RE = re.compile(r"turn right\s+(\d+)\s*degree", re.IGNORECASE)

# Paper: maps to fixed velocities {0.5 m/s, π/6 rad/s, -π/6 rad/s, 0}
_FWD_VX = 0.35
_TURN_YAW = 0.52  # ~π/6 rad/s scaled for our policy
_MODE_MAP = {"normal": "normal", "cautious": "cautious", "rough": "normal", "parkour": "cautious"}


def midlevel_to_command(raw_text: str) -> tuple[float, float, float, str]:
  """Convert NaVILA mid-level natural language output to (vx, vy, yaw, reasoning).

  NaVILA research (paper, navila_trainer.py): outputs action-only, e.g. "The next action
  is move forward 50 cm" | "turn left 30 degree" | "stop". No reasoning requested.
  Maps to continuous commands for PPO. reasoning is always "" per paper.
  """
  text = raw_text.strip()
  # Paper does not use reasoning; we keep empty for compatibility with NavilaResult
  _reasoning = ""

  if _STOP_RE.search(text):
    return 0.0, 0.0, 0.0, _reasoning

  if _FWD_RE.search(text):
    m = _FWD_CM_RE.search(text)
    cm = int(m.group(1)) if m else 50
    # Scale: 50cm -> base vx, 25->0.5x, 75->1.0x
    scale = min(1.0, max(0.5, cm / 50.0))
    return float(_FWD_VX * scale), 0.0, 0.0, _reasoning

  if _LEFT_RE.search(text):
    m = _LEFT_DEG_RE.search(text)
    deg = int(m.group(1)) if m else 30
    yaw = _TURN_YAW * min(1.0, deg / 30.0)
    return 0.1, 0.0, float(yaw), _reasoning  # Slight forward while turning

  if _RIGHT_RE.search(text):
    m = _RIGHT_DEG_RE.search(text)
    deg = int(m.group(1)) if m else 30
    yaw = -_TURN_YAW * min(1.0, deg / 30.0)
    return 0.1, 0.0, float(yaw), _reasoning

  # Fallback: forward cautiously
  return 0.2, 0.0, 0.0, _reasoning


@dataclass
class NavilaResult:
  reasoning: str
  vx: float
  vy: float
  yaw: float
  mode: str
  prompt_tokens_est: int
  inference_ms: float
  parse_ms: float
  fallback: bool
  parse_error: str
  raw_text: str

  def as_dict(self) -> dict[str, Any]:
    return {
        "reasoning": self.reasoning,
        "vx": self.vx,
        "vy": self.vy,
        "yaw": self.yaw,
        "mode": self.mode,
        "prompt_tokens_est": self.prompt_tokens_est,
        "inference_ms": self.inference_ms,
        "parse_ms": self.parse_ms,
        "fallback": self.fallback,
        "parse_error": self.parse_error,
        "raw_text": self.raw_text,
    }


def _flash_attn_available() -> bool:
  try:
    import flash_attn  # noqa: F401
    return True
  except ImportError:
    return False


def _normalize_mode(raw_mode: Any) -> str:
  if raw_mode is None:
    return "cautious"
  return _MODE_MAP.get(str(raw_mode).strip().lower(), "cautious")


class _NaVILALLavaBridgeImpl:
  """LLaVA-format NaVILA models (a8cheng/navila-*) via load_pretrained_model."""

  def __init__(
      self,
      model_name: str,
      device: str,
      max_new_tokens: int,
      num_frames: int,
      temperature: float,
  ) -> None:
    self.model_name = model_name
    self.max_new_tokens = max_new_tokens
    self.num_frames = num_frames
    self.temperature = temperature
    self.enabled = False
    self.device = device
    self._frame_buffer: deque = deque(maxlen=num_frames)
    self._model = None
    self._tokenizer = None
    self._image_processor = None
    self._stop_str = None

    if torch is None:
      print("NaVILA LLaVA bridge: torch missing.")
      return

    _ensure_navila_in_path()
    try:
      from llava.conversation import SeparatorStyle, conv_templates
      from llava.mm_utils import (
          KeywordsStoppingCriteria,
          get_model_name_from_path,
          process_images,
          tokenizer_image_token,
      )
      from llava.model.builder import load_pretrained_model
      from llava.constants import IMAGE_TOKEN_INDEX
    except ImportError as e:
      print(
          "NaVILA LLaVA bridge: llava not found. Install NaVILA (pip install -e NaVILA) "
          f"or use HF model. {e!r}"
      )
      return

    mname = get_model_name_from_path(model_name)
    self._conv_mode = "qwen2" if "qwen2" in model_name.lower() else "llama_3"
    conv = conv_templates.get(self._conv_mode, conv_templates["llama_3"]).copy()
    self._stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    self._conv_template = conv_templates.get(self._conv_mode, conv_templates["llama_3"])

    try:
      tokenizer, model, image_processor, _ = load_pretrained_model(
          model_name, mname, load_8bit=False, load_4bit=False
      )
      # Do NOT call model.cuda() or model.to(device) after device_map="auto" loading —
      # accelerate already dispatched layers. For 4-bit bitsandbytes models this would crash.
      self._model = model
      self._tokenizer = tokenizer
      self._image_processor = image_processor
      num_vf = getattr(model.config, "num_video_frames", num_frames)
      self.num_frames = min(num_vf, num_frames)
      self.enabled = True
      print(
          f"NaVILA LLaVA bridge initialized model={model_name} device={device} "
          f"(max_new_tokens={max_new_tokens}, num_frames={self.num_frames}, conv={self._conv_mode})"
      )
    except Exception as exc:
      print(f"NaVILA LLaVA model load failed: {exc!r}")
      self.enabled = False

  def _build_prompt_llava(
      self,
      mission: str,
      terrain_8x8: str,
      max_step_height: float,
      avg_slope: float,
  ) -> str:
    instruction = (
        f"{mission} "
        f"Terrain grid (^=obstacle, #=tall, .=clear): {terrain_8x8.replace(chr(10), ' ')}. "
        f"Max step {max_step_height:.2f}m, slope {avg_slope:.2f}."
    )
    interleaved = "<image>\n" * (self.num_frames - 1)
    return (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f"of historical observations {interleaved}, and current observation <image>\n. "
        f'Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, which could be turning left or right "
        f"by a specific degree, moving forward a certain distance, or stop if the task is completed."
    )

  def _prepare_image(self, rgb: np.ndarray) -> Image.Image:
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[-1] != 3:
      raise ValueError(f"Expected RGB (H,W,3), got shape={frame.shape}.")
    if frame.dtype != np.uint8:
      if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0) * 255.0
      frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")

  def infer_semantic_command(
      self,
      rgb_frame: np.ndarray,
      mission: str,
      terrain_8x8: str,
      max_step_height: float,
      avg_slope: float,
      base_z: float,
      speed: float,
  ) -> dict[str, Any]:
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )

    prompt = self._build_prompt_llava(mission, terrain_8x8, max_step_height, avg_slope)
    prompt_tokens_est = int(len(prompt.split()) * 1.3)

    if not self.enabled:
      return NavilaResult(
          reasoning="",
          vx=0.0, vy=0.0, yaw=0.0,
          mode="cautious",
          prompt_tokens_est=prompt_tokens_est,
          inference_ms=0.0, parse_ms=0.0,
          fallback=True, parse_error="", raw_text="",
      ).as_dict()

    try:
      curr_pil = self._prepare_image(rgb_frame)
    except Exception as exc:
      return NavilaResult(
          reasoning="",
          vx=0.0, vy=0.0, yaw=0.0,
          mode="cautious",
          prompt_tokens_est=prompt_tokens_est,
          inference_ms=0.0, parse_ms=0.0,
          fallback=True, parse_error=f"image_prepare_failed: {exc!r}", raw_text="",
      ).as_dict()

    self._frame_buffer.append(curr_pil)
    past_and_current = list(self._frame_buffer)
    sampled = _sample_and_pad_images(
        past_and_current,
        num_frames=self.num_frames,
        width=448,
        height=448,
    )

    conv = self._conv_template.copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # Resolve the main compute device: for device_map="auto" with CPU offload,
    # first real parameter gives us the correct cuda device.
    try:
      _model_device = next(
          (p.device for p in self._model.parameters() if p.device.type != "meta"),
          torch.device("cuda")
      )
    except Exception:
      _model_device = torch.device("cuda")

    infer_t0 = time.perf_counter()
    try:
      images_tensor = process_images(
          sampled, self._image_processor, self._model.config
      ).to(_model_device, dtype=torch.float16)
      input_ids = tokenizer_image_token(
          full_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
      ).unsqueeze(0).to(_model_device)
      stop_keywords = [self._stop_str]
      stopping_criteria = KeywordsStoppingCriteria(
          stop_keywords, self._tokenizer, input_ids
      )
      with torch.inference_mode():
        output_ids = self._model.generate(
            input_ids,
            images=images_tensor.to(_model_device, dtype=torch.float16),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=self._tokenizer.eos_token_id,
        )
      outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
      outputs = outputs.strip()
      if outputs.endswith(self._stop_str):
        outputs = outputs[: -len(self._stop_str)]
      raw_text = outputs.strip()
      inference_ms = (time.perf_counter() - infer_t0) * 1000.0
    except Exception as exc:
      inference_ms = (time.perf_counter() - infer_t0) * 1000.0
      return NavilaResult(
          reasoning="",
          vx=0.0, vy=0.0, yaw=0.0,
          mode="cautious",
          prompt_tokens_est=prompt_tokens_est,
          inference_ms=inference_ms, parse_ms=0.0,
          fallback=True, parse_error=f"inference_failed: {exc!r}", raw_text="",
      ).as_dict()

    parse_t0 = time.perf_counter()
    try:
      vx, vy, yaw, reasoning = midlevel_to_command(raw_text)
      vx, vy, yaw = (
          float(np.clip(vx, -1.2, 1.2)),
          float(np.clip(vy, -0.8, 0.8)),
          float(np.clip(yaw, -1.5, 1.5)),
      )
      return NavilaResult(
          reasoning=reasoning,
          vx=vx, vy=vy, yaw=yaw,
          mode="cautious",
          prompt_tokens_est=prompt_tokens_est,
          inference_ms=inference_ms,
          parse_ms=(time.perf_counter() - parse_t0) * 1000.0,
          fallback=False,
          parse_error="",
          raw_text=raw_text,
      ).as_dict()
    except Exception as exc:
      return NavilaResult(
          reasoning="",
          vx=0.0, vy=0.0, yaw=0.0,
          mode="cautious",
          prompt_tokens_est=prompt_tokens_est,
          inference_ms=inference_ms,
          parse_ms=(time.perf_counter() - parse_t0) * 1000.0,
          fallback=True,
          parse_error=f"midlevel_parse_failed: {exc!r}",
          raw_text=raw_text,
      ).as_dict()


class NaVILAVLABridge:
  """NaVILA bridge with research-exact prompting and mid-level → command conversion.

  Routes to LLaVA loader for a8cheng/navila-* models, else uses HuggingFace AutoModel.
  """

  def __init__(
      self,
      model_name: str = "Efficient-Large-Model/NVILA-Lite-2B-hf",
      device: str = "cuda",
      max_new_tokens: int = 32,  # NaVILA research uses 32 (navila_trainer.py); action-only output
      token_budget_target: int = 256,
      temperature: float = 0.0,
      top_p: float = 0.95,
      use_flash_attention: bool = True,
      use_torch_compile: bool = False,
      navila_num_frames: int = 64,  # For LLaVA models: 64 or 8 for latency
  ) -> None:
    self.model_name = model_name
    self.do_sample = temperature > 0.0
    self.temperature = temperature if self.do_sample else 0.0
    self.top_p = top_p
    self.max_new_tokens = max_new_tokens
    self.token_budget_target = token_budget_target
    self.navila_num_frames = navila_num_frames
    self.enabled = False
    self.device = "cpu"
    self._pin_buffer: torch.Tensor | None = None
    self._llava_backend: _NaVILALLavaBridgeImpl | None = None

    has_cuda = torch.cuda.is_available() if torch else False
    dev = "cuda" if (device == "cuda" and has_cuda) else "cpu"

    if _use_llava_loader(model_name):
      self._llava_backend = _NaVILALLavaBridgeImpl(
          model_name=model_name,
          device=dev,
          max_new_tokens=max_new_tokens,
          num_frames=navila_num_frames,
          temperature=temperature,
      )
      self.enabled = self._llava_backend.enabled
      self.device = dev
      return

    if torch is None or AutoModel is None or AutoProcessor is None:
      print("NaVILA bridge dependencies missing. Install torch+transformers.")
      return

    self.device = dev
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if self.device == "cuda":
      if BitsAndBytesConfig is not None:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
      else:
        load_kwargs["load_in_4bit"] = True
      load_kwargs["device_map"] = "auto"
      load_kwargs["torch_dtype"] = torch.bfloat16
      if use_flash_attention and _flash_attn_available():
        load_kwargs["attn_implementation"] = "flash_attention_2"
    else:
      load_kwargs["torch_dtype"] = torch.float32

    try:
      self.model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
      self.processor = AutoProcessor.from_pretrained(
          self.model_name, trust_remote_code=True
      )
      if use_torch_compile and self.device == "cuda" and hasattr(torch, "compile"):
        try:
          self.model = torch.compile(self.model, mode="max-autotune")
          print("NaVILA model compiled with torch.compile(mode='max-autotune')")
        except Exception as compile_err:
          print(f"torch.compile skipped: {compile_err!r}")
      self.enabled = True
      print(
          f"NaVILA bridge initialized model={self.model_name} device={self.device} "
          f"(max_new_tokens={max_new_tokens}, mid-level output)"
      )
    except Exception as exc:
      err_str = str(exc)
      if "llava_llama" in err_str and "does not recognize" in err_str:
        print(
            "NaVILA uses llava_llama (VILA format), which standard transformers "
            "does not support. Install the NaVILA repo and use its environment."
        )
      print(f"NaVILA model load failed, using safe fallback only: {exc!r}")
      self.enabled = False

  def _estimate_tokens(self, prompt: str) -> int:
    words = max(1, len(re.findall(r"\S+", prompt)))
    return int(words * 1.3)

  def _clamp_cmd(self, vx: float, vy: float, yaw: float) -> tuple[float, float, float]:
    return (
        float(np.clip(vx, -1.2, 1.2)),
        float(np.clip(vy, -0.8, 0.8)),
        float(np.clip(yaw, -1.5, 1.5)),
    )

  def _safe_result(
      self,
      prompt_tokens_est: int,
      inference_ms: float = 0.0,
      parse_ms: float = 0.0,
      parse_error: str = "",
      raw_text: str = "",
  ) -> NavilaResult:
    return NavilaResult(
        reasoning="",
        vx=0.0,
        vy=0.0,
        yaw=0.0,
        mode="cautious",
        prompt_tokens_est=prompt_tokens_est,
        inference_ms=inference_ms,
        parse_ms=parse_ms,
        fallback=True,
        parse_error=parse_error,
        raw_text=raw_text,
    )

  def _prepare_pinned_image(self, rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[-1] != 3:
      raise ValueError(f"Expected RGB (H,W,3), got shape={frame.shape}.")
    if frame.dtype != np.uint8:
      if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0) * 255.0
      frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

    if torch is None:
      return frame
    if self.device == "cuda":
      torch.cuda.synchronize()
    src = torch.as_tensor(frame, device="cpu")
    h, w, c = src.shape
    need_alloc = (
        self._pin_buffer is None
        or self._pin_buffer.shape[0] != h
        or self._pin_buffer.shape[1] < w
        or self._pin_buffer.shape[2] != c
    )
    if need_alloc:
      self._pin_buffer = torch.empty((h, w * 2, c), dtype=src.dtype, pin_memory=True)
    view = self._pin_buffer[:, :w, :]
    view.copy_(src, non_blocking=False)
    if self.device == "cuda":
      torch.cuda.synchronize()
    return view.cpu().numpy()

  def _build_prompt(
      self,
      mission: str,
      terrain_8x8: str,
      max_step_height: float,
      avg_slope: float,
      base_z: float,
      speed: float,
  ) -> str:
    # EXACT NaVILA research prompt (navila_trainer.py lines 174-179, run_navigation.py).
    # Paper does NOT ask for reasoning—only the action. Output: "The next action is move forward 50 cm" etc.
    instruction = (
        f"{mission} "
        f"Terrain grid (^=obstacle, #=tall, .=clear): {terrain_8x8.replace(chr(10), ' ')}. "
        f"Max step {max_step_height:.2f}m, slope {avg_slope:.2f}."
    )
    # Single frame: chat template injects <image> for {"type":"image"}; do NOT add <image> in text.
    return (
        f"Imagine you are a robot programmed for navigation tasks. You have been given the current "
        f'observation. Your assigned task is: "{instruction}" '
        f"Decide your next action, which could be turning left or right by a specific degree, moving forward "
        f"a certain distance, or stop if the task is completed. "
        f"Reply with only the action, e.g. 'The next action is move forward 50 cm' or 'The next action is turn left 30 degree'."
    )

  def infer_semantic_command(
      self,
      rgb_frame: np.ndarray,
      mission: str,
      terrain_8x8: str,
      max_step_height: float,
      avg_slope: float,
      base_z: float,
      speed: float,
  ) -> dict[str, Any]:
    if self._llava_backend is not None and self._llava_backend.enabled:
      return self._llava_backend.infer_semantic_command(
          rgb_frame=rgb_frame,
          mission=mission,
          terrain_8x8=terrain_8x8,
          max_step_height=max_step_height,
          avg_slope=avg_slope,
          base_z=base_z,
          speed=speed,
      )

    prompt = self._build_prompt(
        mission=mission,
        terrain_8x8=terrain_8x8,
        max_step_height=max_step_height,
        avg_slope=avg_slope,
        base_z=base_z,
        speed=speed,
    )
    prompt_tokens_est = self._estimate_tokens(prompt)

    if not self.enabled:
      print("[NaVILA] Skipping inference: bridge disabled (model not loaded).")
      return self._safe_result(prompt_tokens_est=prompt_tokens_est).as_dict()

    try:
      img_np = self._prepare_pinned_image(rgb_frame)
      img = Image.fromarray(img_np, mode="RGB")
    except Exception as exc:
      return self._safe_result(
          prompt_tokens_est=prompt_tokens_est,
          parse_error=f"image_prepare_failed: {exc!r}",
      ).as_dict()

    infer_t0 = time.perf_counter()
    try:
      if self.device == "cuda":
        torch.cuda.synchronize()

      # NaVILA: user message only, no system (matches research).
      messages = [
          {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
      ]
      text_for_model = self.processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )
      model_inputs = self.processor(
          text=text_for_model, images=[img], return_tensors="pt"
      )
      model_inputs = {
          k: v.to(self.device) if hasattr(v, "to") else v
          for k, v in model_inputs.items()
      }
      gen_kwargs = {
          "max_new_tokens": self.max_new_tokens,
          "do_sample": self.do_sample,
          "pad_token_id": self.processor.tokenizer.eos_token_id,
      }
      if self.do_sample:
        gen_kwargs["temperature"] = self.temperature
        gen_kwargs["top_p"] = self.top_p
      with torch.inference_mode():
        outputs = self.model.generate(**model_inputs, **gen_kwargs)
      input_len = model_inputs["input_ids"].shape[1]
      generated_ids = outputs[:, input_len:]
      decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
      raw_text = decoded[0] if decoded else ""

      if self.device == "cuda":
        torch.cuda.synchronize()
      inference_ms = (time.perf_counter() - infer_t0) * 1000.0
    except Exception as exc:
      # #region agent log
      import json as _json
      tb = traceback.format_exc()
      _log_path = "/home/siddu/mujoco_playground/.cursor/debug-2658b7.log"
      try:
        with open(_log_path, "a") as _f:
          _f.write(
              _json.dumps({
                  "sessionId": "2658b7",
                  "location": "navila_vla_bridge.py:infer",
                  "message": "inference_failed",
                  "data": {"exc": str(exc), "tb": tb[:2000], "max_new_tokens": self.max_new_tokens},
                  "timestamp": int(time.time() * 1000),
              }) + "\n"
          )
      except Exception:
        pass
      # #endregion
      return self._safe_result(
          prompt_tokens_est=prompt_tokens_est,
          inference_ms=(time.perf_counter() - infer_t0) * 1000.0,
          parse_error=f"inference_failed: {exc!r}",
      ).as_dict()

    parse_t0 = time.perf_counter()
    try:
      vx, vy, yaw, reasoning = midlevel_to_command(raw_text)
      vx, vy, yaw = self._clamp_cmd(vx, vy, yaw)
      out = NavilaResult(
          reasoning=reasoning,
          vx=vx,
          vy=vy,
          yaw=yaw,
          mode="cautious",
          prompt_tokens_est=prompt_tokens_est,
          inference_ms=inference_ms,
          parse_ms=(time.perf_counter() - parse_t0) * 1000.0,
          fallback=False,
          parse_error="",
          raw_text=raw_text,
      )
      return out.as_dict()
    except Exception as exc:
      return self._safe_result(
          prompt_tokens_est=prompt_tokens_est,
          inference_ms=inference_ms,
          parse_ms=(time.perf_counter() - parse_t0) * 1000.0,
          parse_error=f"midlevel_parse_failed: {exc!r}",
          raw_text=raw_text,
      ).as_dict()
