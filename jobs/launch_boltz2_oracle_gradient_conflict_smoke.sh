#!/usr/bin/env bash
set -euo pipefail

# Run from the baseline worktree on an allocated Quest GPU node.
LOG_DIR="${LOG_DIR:-/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/manual-boltz2-oracle-gradient-conflict-$(hostname)-${RUN_STAMP}.out}"

export BOLTZ_CACHE="${BOLTZ_CACHE:-/projects/p32572/Jieke/.cache/boltz}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_DIR}"
nohup uv run python -u scripts/run_boltz2_oracle_gradient_conflict.py \
  --steps "${STEPS:-1}" \
  --num-seeds "${NUM_SEEDS:-1}" \
  --method-ids "${METHOD_IDS:-M3,M4,M7c,M8a}" \
  --binder-length "${BINDER_LENGTH:-24}" \
  --target-length "${TARGET_LENGTH:-48}" \
  --recycling-steps "${RECYCLING_STEPS:-1}" \
  --sampling-steps "${SAMPLING_STEPS:-1}" \
  --boltz-recycling-steps "${BOLTZ_RECYCLING_STEPS:-1}" \
  --step-size "${STEP_SIZE:-0.25}" \
  >"${LOG_PATH}" 2>&1 </dev/null &
PID="$!"

echo "pid=${PID}"
echo "log_path=${LOG_PATH}"
