#!/usr/bin/env bash
set -euo pipefail

# Re-score an existing Phase 0 candidate-gate Protenix output with Boltz2.
# This is useful when candidate selection policy changes but Protenix candidates
# do not need to be regenerated.

LOG_DIR="${LOG_DIR:-/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LABEL="${RUN_LABEL:-phase0_candidate_gate_boltz2_rescore}"
SOURCE_RUN="${SOURCE_RUN:-phase0_candidate_gate_fair_m3_m4_m11a_20260513T135620Z}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/manual-${RUN_LABEL}-$(hostname)-${RUN_STAMP}.out}"
MONITOR_LOG_PATH="${MONITOR_LOG_PATH:-${LOG_DIR}/resource-${RUN_LABEL}-$(hostname)-${RUN_STAMP}.out}"
TARGET_ROOT="${TARGET_ROOT:-/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb}"
RUN_GIT_COMMIT="${RUN_GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo nogit)}"

export BOLTZ_CACHE="${BOLTZ_CACHE:-/projects/p32572/Jieke/.cache/boltz}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export RUN_STAMP
export RUN_LABEL
export SOURCE_RUN
export TARGET_ROOT
export RUN_GIT_COMMIT

mkdir -p "${LOG_DIR}"

nohup bash -lc '
set -euo pipefail

IFS="," read -r -a REQUESTED_TARGETS <<< "${TARGET_IDS:-il7ra,pdl1}"
MAX_PARALLEL_PROCESSES="${MAX_PARALLEL_PROCESSES:-2}"

echo "run_label=${RUN_LABEL}"
echo "source_run=${SOURCE_RUN}"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "node=$(hostname)"
echo "git_commit=${RUN_GIT_COMMIT}"
echo "target_ids=${TARGET_IDS:-il7ra,pdl1}"
echo "max_parallel_processes=${MAX_PARALLEL_PROCESSES}"
echo "max_boltz2_candidates=${MAX_BOLTZ2_CANDIDATES:-18}"
echo "balance_by=${BALANCE_BY:-method_score_mode}"
echo "deduplicate_scope=${DEDUPLICATE_SCOPE:-group}"
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

run_target() {
  local target_id="$1"
  local spec structure chain length source_dir candidates_csv boltz_dir
  spec="$(target_spec "${target_id}")"
  IFS="|" read -r structure chain length <<< "${spec}"
  source_dir="docs/results/${SOURCE_RUN}/${target_id}/protenix"
  candidates_csv="$(find "${source_dir}" -maxdepth 1 -name "*_candidates.csv" -print | sort | tail -1)"
  if [[ -z "${candidates_csv}" ]]; then
    echo "missing candidates csv for ${target_id} in ${source_dir}" >&2
    return 3
  fi

  boltz_dir="docs/results/${RUN_LABEL}_${RUN_STAMP}/${target_id}/boltz2"
  mkdir -p "${boltz_dir}" docs/reports

  echo "===TARGET ${target_id} boltz2_start=$(date -u +%Y-%m-%dT%H:%M:%SZ) candidates_csv=${candidates_csv}==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python -u scripts/score_boltz2_candidate_holdout.py \
    --candidates-csv "${candidates_csv}" \
    --target-structure "${structure}" \
    --target-chain "${chain}" \
    --target-length "${length}" \
    --boltz-cache "${BOLTZ_CACHE}" \
    --max-candidates "${MAX_BOLTZ2_CANDIDATES:-18}" \
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

running_jobs=0
for target_id in "${REQUESTED_TARGETS[@]}"; do
  run_target "${target_id}" &
  running_jobs=$((running_jobs + 1))
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
    ps -fu "${USER}" | grep -E "score_boltz2_candidate_holdout|uv run python" | grep -v grep || true
    sleep 60
  done
  echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
) >"${MONITOR_LOG_PATH}" 2>&1 </dev/null &
MONITOR_PID="$!"

echo "pid=${PID}"
echo "monitor_pid=${MONITOR_PID}"
echo "log_path=${LOG_PATH}"
echo "monitor_log_path=${MONITOR_LOG_PATH}"
