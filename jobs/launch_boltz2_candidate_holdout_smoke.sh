#!/usr/bin/env bash
set -euo pipefail

# Launch a detached smoke on an already allocated Quest GPU node.
LOG_DIR="${LOG_DIR:-/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/manual-boltz2-candidate-holdout-$(hostname)-${RUN_STAMP}.out}"

mkdir -p "${LOG_DIR}"
nohup bash jobs/boltz2_candidate_holdout_smoke.sh >"${LOG_PATH}" 2>&1 </dev/null &
PID="$!"

echo "pid=${PID}"
echo "log_path=${LOG_PATH}"
