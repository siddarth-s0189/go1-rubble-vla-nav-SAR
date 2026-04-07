#!/bin/bash
set -euo pipefail

# NaVILA-VLA + MuJoCo Go1 Rubble P2P: 3-tier stack (NaVILA -> SkillManager -> Rough PPO)
# vla_interval_steps=20 -> ~2.5 Hz at ctrl_dt=0.02
#
# Test commands (args passed via "$@"):
#   ./master_rollout.sh --episode_length=120 --vla_temp=0.8
#   ./master_rollout.sh --mission="Navigate around rubble to reach the goal marker" --vla_temp=0.8 --episode_length=120
#   ./master_rollout.sh --vla_model="Efficient-Large-Model/NVILA-8B-hf" --episode_length=120  # HF fallback
#   ./master_rollout.sh --disable_hazard_override=false  # re-enable safety override
[ -f .env ] && set -a && source .env && set +a

# DeepSpeed (used by NaVILA/llava) requires CUDA_HOME for op compatibility check.
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc &>/dev/null; then
    export CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")
  elif [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
  elif [ -d /usr/lib/cuda ]; then
    export CUDA_HOME=/usr/lib/cuda
  fi
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

"${BASH_SOURCE%/*}"/venv/bin/python learning/play_vlm_sar.py \
  --env_name=Go1JoystickSARStage5 \
  --episode_length=600 \
  --n_rubble=60 \
  --rubble_seed=42 \
  --vla_interval_steps=20 \
  --mission="cross rubble safely" \
  --vla_command_scale=2.5 \
  --vla_model=a8cheng/navila-qwen2-7b-64k-64f \
  --output_video="navila_rollout.mp4" \
  --narrator_log="navila_narrator.jsonl" \
  "$@"
