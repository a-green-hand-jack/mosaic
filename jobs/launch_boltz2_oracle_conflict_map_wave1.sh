#!/usr/bin/env bash
set -euo pipefail

# Run a small Wave 1 cross-target Protenix/Boltz2 oracle conflict map.
# Run from the baseline worktree on an allocated idle Quest GPU node.

LOG_DIR="${LOG_DIR:-/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/manual-boltz2-oracle-conflict-map-wave1-$(hostname)-${RUN_STAMP}.out}"
TARGET_ROOT="${TARGET_ROOT:-/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb}"

export BOLTZ_CACHE="${BOLTZ_CACHE:-/projects/p32572/Jieke/.cache/boltz}"
export TARGET_ROOT
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_DIR}"

nohup bash -lc '
set -euo pipefail
echo "run_label=phase0_conflict_map_wave1"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "node=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true

run_target() {
  local target_id="$1"
  local structure="$2"
  local chain="$3"
  local length="$4"
  echo "===TARGET ${target_id} structure=${structure} chain=${chain} length=${length} start=$(date -u +%Y-%m-%dT%H:%M:%SZ)==="
  uv run python -u scripts/run_boltz2_oracle_gradient_conflict.py \
    --target-structure "${structure}" \
    --target-chain "${chain}" \
    --target-length "${length}" \
    --steps "${STEPS:-1}" \
    --num-seeds "${NUM_SEEDS:-4}" \
    --method-ids "${METHOD_IDS:-M3,M4,M7c,M8a}" \
    --binder-length "${BINDER_LENGTH:-24}" \
    --recycling-steps "${RECYCLING_STEPS:-1}" \
    --sampling-steps "${SAMPLING_STEPS:-1}" \
    --boltz-recycling-steps "${BOLTZ_RECYCLING_STEPS:-1}" \
    --step-size "${STEP_SIZE:-0.25}" \
    --report-dir docs/reports \
    --output-dir docs/results
  echo "===TARGET ${target_id} done=$(date -u +%Y-%m-%dT%H:%M:%SZ)==="
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
}

run_target il7ra IL7RA.cif A 48
run_target pdl1 "${TARGET_ROOT}/4ZQK.cif" A 48
run_target pd1 "${TARGET_ROOT}/4ZQK.cif" B 48
run_target cd47 "${TARGET_ROOT}/2JJS.cif" C 48
run_target il6 "${TARGET_ROOT}/1ALU.cif" A 48

echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
' >"${LOG_PATH}" 2>&1 </dev/null &
PID="$!"

echo "pid=${PID}"
echo "log_path=${LOG_PATH}"
