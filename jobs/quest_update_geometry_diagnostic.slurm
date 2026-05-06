#!/usr/bin/env bash
# Submit with an explicit GPU class when desired:
#   sbatch --gres=gpu:a100:1 jobs/quest_update_geometry_diagnostic.slurm
#   sbatch --gres=gpu:h100:1 jobs/quest_update_geometry_diagnostic.slurm
# This job is lightweight enough to run on a placeholder node via:
#   bash jobs/quest_update_geometry_diagnostic.slurm
#SBATCH --account=p32572
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sch-update-geo
#SBATCH --output=/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/slurm-update-geo-%j.out

set -euo pipefail

export UV_CACHE_DIR=/projects/p32572/Jieke/.cache/uv
export UV_PYTHON_INSTALL_DIR=/projects/p32572/Jieke/Envs/uv-python
export BOLTZ_CACHE=/projects/p32572/Jieke/.cache/boltz
export PYTHONNOUSERSITE=1

cd /projects/p32572/Jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines

echo "node=$(hostname)"
if command -v git >/dev/null 2>&1; then
  echo "commit=$(git rev-parse --short HEAD)"
else
  echo "commit=unavailable-git-not-on-path"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

/home/zpt6685/.local/bin/uv run python scripts/run_update_geometry_diagnostic.py \
  --binder-length 80 \
  --steps 24 \
  --num-seeds 8 \
  --step-size 0.45 \
  --output-dir docs/results \
  --report-dir docs/reports
