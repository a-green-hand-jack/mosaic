#!/usr/bin/env bash
# Submit with:
#   sbatch jobs/quest_protenix_act020_warm_cem_h100.slurm
# Or run on an idle H100 placeholder node:
#   bash jobs/quest_protenix_act020_warm_cem_h100.slurm
# Default purpose:
#   Run ACT-020 warm-started CEM / hard-candidate optimizer variants.
#SBATCH --account=p32572
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sch-act020-h100
#SBATCH --output=/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/slurm-act020-h100-%j.out

set -euo pipefail

export UV_CACHE_DIR=/projects/p32572/Jieke/.cache/uv
export UV_PYTHON_INSTALL_DIR=/projects/p32572/Jieke/Envs/uv-python
export HF_HOME=/projects/p32572/Jieke/.cache/huggingface
export XDG_CACHE_HOME=/projects/p32572/Jieke/.cache
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /projects/p32572/Jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines

echo "node=$(hostname)"
if command -v git >/dev/null 2>&1; then
  echo "commit=$(git rev-parse --short HEAD)"
else
  echo "commit=unavailable-git-not-on-path"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

/usr/bin/time -v /home/zpt6685/.local/bin/uv run python scripts/run_protenix_cem_optimizer.py \
  --target-structure IL7RA.cif \
  --target-length "${TARGET_LENGTH:-48}" \
  --binder-length "${BINDER_LENGTH:-24}" \
  --steps "${STEPS:-3}" \
  --num-seeds "${NUM_SEEDS:-2}" \
  --step-size "${STEP_SIZE:-0.25}" \
  --recycling-steps "${RECYCLING_STEPS:-1}" \
  --sampling-steps "${SAMPLING_STEPS:-1}" \
  --candidate-sample-top-k "${CANDIDATE_SAMPLE_TOP_K:-4}" \
  --candidate-sample-temperature "${CANDIDATE_SAMPLE_TEMPERATURE:-0.25}" \
  --candidate-best-metric "${CANDIDATE_BEST_METRIC:-bt_pae}" \
  --baseline-method-ids "${BASELINE_METHOD_IDS:-M3,M7c,M8a}" \
  --baseline-topk-samples "${BASELINE_TOPK_SAMPLES:-24}" \
  --cem-method-ids "${CEM_METHOD_IDS:-WCEMp,WCEMc}" \
  --cem-rounds "${CEM_ROUNDS:-3}" \
  --cem-samples-per-round "${CEM_SAMPLES_PER_ROUND:-8}" \
  --cem-elite-count "${CEM_ELITE_COUNT:-2}" \
  --cem-update-rate "${CEM_UPDATE_RATE:-0.6}" \
  --cem-min-uniform-mix "${CEM_MIN_UNIFORM_MIX:-0.05}" \
  --warm-start-uniform-mix "${WARM_START_UNIFORM_MIX:-0.02}" \
  --protenix-cache /projects/p32572/Jieke/.cache/protenix \
  --output-dir docs/results \
  --report-dir docs/reports

latest_candidates="$(ls -t docs/results/phase0_protenix_cem_*_candidates.csv | head -1)"
run_stem="${latest_candidates##*/}"
run_stem="${run_stem%_candidates.csv}"

/home/zpt6685/.local/bin/uv run python scripts/summarize_topk_handoff_sensitivity.py \
  --candidates-csv "${latest_candidates}" \
  --budgets "${TOPK_BUDGETS:-1,8,24}" \
  --metrics "${RERANK_METRICS:-bt_pae,bt_iptm,contact}" \
  --output-prefix "docs/results/${run_stem}_topk_sensitivity"
