#!/usr/bin/env bash
set -euo pipefail

# Balanced 24-candidate Boltz2 holdout over the latest ACT-021 CEM table.
export CANDIDATES_CSV="${CANDIDATES_CSV:-docs/results/phase0_protenix_cem_23ff9f1_20260507T232223Z_candidates.csv}"
export MAX_CANDIDATES="${MAX_CANDIDATES:-24}"
export METHOD_IDS="${METHOD_IDS:-M3,M7c,M8a}"
export SCORE_MODES="${SCORE_MODES:-argmax,topk_sample,soft}"
export BALANCE_BY="${BALANCE_BY:-method_id}"
export RUN_LABEL="${RUN_LABEL:-act021_m3_m7c_m8a_24}"

exec bash jobs/launch_boltz2_candidate_holdout_smoke.sh
