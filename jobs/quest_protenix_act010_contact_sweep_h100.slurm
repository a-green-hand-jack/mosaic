#!/usr/bin/env bash
# Submit with:
#   sbatch jobs/quest_protenix_act010_contact_sweep_h100.slurm
# Or run on an idle H100 placeholder node:
#   bash jobs/quest_protenix_act010_contact_sweep_h100.slurm
#SBATCH --account=p32572
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sch-act010-h100
#SBATCH --output=/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/slurm-act010-h100-%j.out

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

/usr/bin/time -v /home/zpt6685/.local/bin/uv run python scripts/run_protenix_update_geometry_smoke.py \
  --target-structure IL7RA.cif \
  --target-length "${TARGET_LENGTH:-48}" \
  --binder-length "${BINDER_LENGTH:-24}" \
  --steps "${STEPS:-3}" \
  --num-seeds "${NUM_SEEDS:-2}" \
  --step-size "${STEP_SIZE:-0.25}" \
  --recycling-steps "${RECYCLING_STEPS:-1}" \
  --sampling-steps "${SAMPLING_STEPS:-1}" \
  --candidate-score-mode "${CANDIDATE_SCORE_MODE:-both}" \
  --protenix-cache /projects/p32572/Jieke/.cache/protenix \
  --output-dir docs/results \
  --report-dir docs/reports
