#!/usr/bin/env bash
# Submit with an explicit GPU class when desired:
#   sbatch --gres=gpu:a100:1 jobs/quest_baseline_pilot.slurm
#   sbatch --gres=gpu:h100:1 jobs/quest_baseline_pilot.slurm
# The checked-in config defaults to dry-run provenance only.
# Flip runtime.dry_run in configs/baselines/phase0_il7ra.yaml only after ACT-005/ACT-007 are satisfied.
#SBATCH --account=p32572
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sch-baseline-pilot
#SBATCH --output=/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/slurm-baseline-pilot-%j.out

set -euo pipefail

export UV_CACHE_DIR=/projects/p32572/Jieke/.cache/uv
export UV_PYTHON_INSTALL_DIR=/projects/p32572/Jieke/Envs/uv-python
export BOLTZ_CACHE=/projects/p32572/Jieke/.cache/boltz
export PYTHONNOUSERSITE=1

cd /projects/p32572/Jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines

echo "node=$(hostname)"
echo "commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

/home/zpt6685/.local/bin/uv run python scripts/run_baseline_pilot.py \
  --config configs/baselines/phase0_il7ra.yaml \
  --dry-run
