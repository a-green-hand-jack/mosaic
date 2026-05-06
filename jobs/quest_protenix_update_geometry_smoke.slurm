#!/usr/bin/env bash
# Submit with:
#   sbatch --gres=gpu:a100:1 jobs/quest_protenix_update_geometry_smoke.slurm
# Or run on an idle placeholder node:
#   bash jobs/quest_protenix_update_geometry_smoke.slurm
#SBATCH --account=p32572
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sch-protenix-geo
#SBATCH --output=/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/slurm-protenix-geo-%j.out

set -euo pipefail

export UV_CACHE_DIR=/projects/p32572/Jieke/.cache/uv
export UV_PYTHON_INSTALL_DIR=/projects/p32572/Jieke/Envs/uv-python
export HF_HOME=/projects/p32572/Jieke/.cache/huggingface
export XDG_CACHE_HOME=/projects/p32572/Jieke/.cache
export PYTHONNOUSERSITE=1

cd /projects/p32572/Jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines

echo "node=$(hostname)"
if command -v git >/dev/null 2>&1; then
  echo "commit=$(git rev-parse --short HEAD)"
else
  echo "commit=unavailable-git-not-on-path"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

/home/zpt6685/.local/bin/uv run python scripts/run_protenix_update_geometry_smoke.py \
  --target-structure IL7RA.cif \
  --target-length 48 \
  --binder-length 24 \
  --steps 4 \
  --num-seeds 2 \
  --step-size 0.25 \
  --recycling-steps 1 \
  --sampling-steps 1 \
  --protenix-cache /projects/p32572/Jieke/.cache/protenix \
  --output-dir docs/results \
  --report-dir docs/reports
