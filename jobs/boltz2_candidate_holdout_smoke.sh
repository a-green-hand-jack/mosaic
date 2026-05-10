#!/usr/bin/env bash
set -euo pipefail

# Run from the baseline worktree on an allocated Quest GPU node.
export BOLTZ_CACHE="${BOLTZ_CACHE:-/projects/p32572/Jieke/.cache/boltz}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv run python -u scripts/score_boltz2_candidate_holdout.py \
  --candidates-csv "${CANDIDATES_CSV:-docs/results/phase0_protenix_cem_2dd3965_20260507T120914Z_candidates.csv}" \
  --target-structure "${TARGET_STRUCTURE:-IL7RA.cif}" \
  --target-chain "${TARGET_CHAIN:-A}" \
  --target-length "${TARGET_LENGTH:-48}" \
  --boltz-cache "${BOLTZ_CACHE}" \
  --max-candidates "${MAX_CANDIDATES:-5}" \
  --deduplicate-sequences \
  --score-modes "${SCORE_MODES:-argmax,topk_sample,soft}" \
  --recycling-steps "${RECYCLING_STEPS:-1}" \
  --sampling-steps "${SAMPLING_STEPS:-1}" \
  --run-label "${RUN_LABEL:-cem2dd3965_top5}"
