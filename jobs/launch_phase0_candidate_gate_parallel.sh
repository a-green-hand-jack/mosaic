#!/usr/bin/env bash
set -euo pipefail

# Run a small Phase 0 candidate-level gate for the promoted update rule.
# The launcher shards by target and supports multiple target processes on one
# GPU or across a comma-separated GPU list.

LOG_DIR="${LOG_DIR:-/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LABEL="${RUN_LABEL:-phase0_candidate_gate_m3_m4_m11a}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/manual-${RUN_LABEL}-$(hostname)-${RUN_STAMP}.out}"
MONITOR_LOG_PATH="${MONITOR_LOG_PATH:-${LOG_DIR}/resource-${RUN_LABEL}-$(hostname)-${RUN_STAMP}.out}"
TARGET_ROOT="${TARGET_ROOT:-/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb}"
RUN_GIT_COMMIT="${RUN_GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo nogit)}"

export BOLTZ_CACHE="${BOLTZ_CACHE:-/projects/p32572/Jieke/.cache/boltz}"
export PROTENIX_CACHE="${PROTENIX_CACHE:-/projects/p32572/Jieke/.cache/protenix}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TARGET_ROOT
export RUN_STAMP
export RUN_LABEL
export RUN_GIT_COMMIT

mkdir -p "${LOG_DIR}"

nohup bash -lc '
set -euo pipefail

IFS="," read -r -a REQUESTED_TARGETS <<< "${TARGET_IDS:-il7ra,pdl1}"
IFS="," read -r -a GPU_IDS <<< "${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_PARALLEL_PROCESSES="${MAX_PARALLEL_PROCESSES:-2}"

echo "run_label=${RUN_LABEL}"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "node=$(hostname)"
echo "git_commit=${RUN_GIT_COMMIT}"
echo "target_ids=${TARGET_IDS:-il7ra,pdl1}"
echo "gpu_ids=${GPU_IDS[*]}"
echo "max_parallel_processes=${MAX_PARALLEL_PROCESSES}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true

target_spec() {
  case "$1" in
    il7ra) echo "IL7RA.cif|A|48" ;;
    pdl1) echo "${TARGET_ROOT}/4ZQK.cif|A|48" ;;
    pd1) echo "${TARGET_ROOT}/4ZQK.cif|B|48" ;;
    cd47) echo "${TARGET_ROOT}/2JJS.cif|C|48" ;;
    il6) echo "${TARGET_ROOT}/1ALU.cif|A|48" ;;
    1vpf_a) echo "${TARGET_ROOT}/1VPF.cif|A|48" ;;
    1vpf_b) echo "${TARGET_ROOT}/1VPF.cif|B|48" ;;
    1vpf_c) echo "${TARGET_ROOT}/1VPF.cif|C|48" ;;
    1vpf_d) echo "${TARGET_ROOT}/1VPF.cif|D|48" ;;
    2p4e_p) echo "${TARGET_ROOT}/2P4E.cif|P|48" ;;
    2p4e_a) echo "${TARGET_ROOT}/2P4E.cif|A|48" ;;
    7opb_a) echo "${TARGET_ROOT}/7OPB.cif|A|48" ;;
    7opb_b) echo "${TARGET_ROOT}/7OPB.cif|B|48" ;;
    7opb_c) echo "${TARGET_ROOT}/7OPB.cif|C|48" ;;
    9ilb_a) echo "${TARGET_ROOT}/9ILB.cif|A|48" ;;
    *)
      echo "unknown target id: $1" >&2
      return 2
      ;;
  esac
}

running_jobs=0
target_index=0

run_target() {
  local target_id="$1"
  local gpu_id="$2"
  local spec structure chain length out_dir protenix_dir boltz_dir candidates_csv
  spec="$(target_spec "${target_id}")"
  IFS="|" read -r structure chain length <<< "${spec}"
  out_dir="docs/results/${RUN_LABEL}_${RUN_STAMP}/${target_id}"
  protenix_dir="${out_dir}/protenix"
  boltz_dir="${out_dir}/boltz2"
  mkdir -p "${protenix_dir}" "${boltz_dir}" docs/reports

  echo "===TARGET ${target_id} gpu=${gpu_id} structure=${structure} chain=${chain} length=${length} start=$(date -u +%Y-%m-%dT%H:%M:%SZ)==="
  CUDA_VISIBLE_DEVICES="${gpu_id}" uv run python -u scripts/run_protenix_update_geometry_smoke.py \
    --target-structure "${structure}" \
    --target-chain "${chain}" \
    --target-length "${length}" \
    --binder-length "${BINDER_LENGTH:-24}" \
    --steps "${STEPS:-1}" \
    --num-seeds "${NUM_SEEDS:-2}" \
    --seed-start "${SEED_START:-0}" \
    --method-ids "${METHOD_IDS:-M3,M4,M11a}" \
    --candidate-score-mode "${CANDIDATE_SCORE_MODE:-both}" \
    --candidate-topk-samples "${CANDIDATE_TOPK_SAMPLES:-2}" \
    --candidate-sample-top-k "${CANDIDATE_SAMPLE_TOP_K:-4}" \
    --candidate-sample-temperature "${CANDIDATE_SAMPLE_TEMPERATURE:-0.5}" \
    --candidate-best-metric "${CANDIDATE_BEST_METRIC:-bt_pae}" \
    --recycling-steps "${RECYCLING_STEPS:-1}" \
    --sampling-steps "${SAMPLING_STEPS:-1}" \
    --step-size "${STEP_SIZE:-0.25}" \
    --protenix-cache "${PROTENIX_CACHE}" \
    --output-dir "${protenix_dir}" \
    --report-dir docs/reports

  candidates_csv="$(find "${protenix_dir}" -maxdepth 1 -name "*_candidates.csv" -print | sort | tail -1)"
  if [[ -z "${candidates_csv}" ]]; then
    echo "missing candidates csv for ${target_id}" >&2
    return 3
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" uv run python -u scripts/score_boltz2_candidate_holdout.py \
    --candidates-csv "${candidates_csv}" \
    --target-structure "${structure}" \
    --target-chain "${chain}" \
    --target-length "${length}" \
    --boltz-cache "${BOLTZ_CACHE}" \
    --max-candidates "${MAX_BOLTZ2_CANDIDATES:-12}" \
    --method-ids "${METHOD_IDS:-M3,M4,M11a}" \
    --score-modes "${BOLTZ2_SCORE_MODES:-argmax,topk_sample,soft}" \
    --balance-by "${BALANCE_BY:-method_score_mode}" \
    --deduplicate-sequences \
    --deduplicate-scope "${DEDUPLICATE_SCOPE:-group}" \
    --recycling-steps "${BOLTZ_RECYCLING_STEPS:-1}" \
    --sampling-steps "${BOLTZ_SAMPLING_STEPS:-1}" \
    --output-dir "${boltz_dir}" \
    --report-dir docs/reports \
    --run-label "${RUN_LABEL}_${target_id}"

  echo "===TARGET ${target_id} done=$(date -u +%Y-%m-%dT%H:%M:%SZ)==="
}

for target_id in "${REQUESTED_TARGETS[@]}"; do
  gpu_id="${GPU_IDS[$((target_index % ${#GPU_IDS[@]}))]}"
  run_target "${target_id}" "${gpu_id}" &
  running_jobs=$((running_jobs + 1))
  target_index=$((target_index + 1))
  if (( running_jobs >= MAX_PARALLEL_PROCESSES )); then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
done

wait
echo "finish_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
' >"${LOG_PATH}" 2>&1 </dev/null &
PID="$!"

(
  while kill -0 "${PID}" 2>/dev/null; do
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits || true
    ps -o pid,ppid,pcpu,pmem,rss,vsz,etime,cmd -p "${PID}" --forest || true
    ps -fu "${USER}" | grep -E "run_protenix_update_geometry_smoke|score_boltz2_candidate_holdout|uv run python" | grep -v grep || true
    sleep 60
  done
  echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
) >"${MONITOR_LOG_PATH}" 2>&1 </dev/null &
MONITOR_PID="$!"

echo "pid=${PID}"
echo "monitor_pid=${MONITOR_PID}"
echo "log_path=${LOG_PATH}"
echo "monitor_log_path=${MONITOR_LOG_PATH}"
