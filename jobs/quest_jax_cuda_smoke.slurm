#!/usr/bin/env bash
# Submit with an explicit GPU class, for example:
#   sbatch --gres=gpu:a100:1 jobs/quest_jax_cuda_smoke.slurm
#   sbatch --gres=gpu:h100:1 jobs/quest_jax_cuda_smoke.slurm
#SBATCH --account=p32572
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sch-jax-smoke
#SBATCH --output=/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/slurm-jax-smoke-%j.out

set -euo pipefail

export UV_CACHE_DIR=/projects/p32572/Jieke/.cache/uv
export UV_PYTHON_INSTALL_DIR=/projects/p32572/Jieke/Envs/uv-python
export PYTHONNOUSERSITE=1

cd /projects/p32572/Jieke/Projects/SCH-BinderDesign/code

echo "node=$(hostname)"
echo "commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

/home/zpt6685/.local/bin/uv run python - <<'PY'
import jax
import jax.numpy as jnp
import torch

print("jax", jax.__version__)
print("jax devices", jax.devices())
x = jnp.ones((1024, 1024), dtype=jnp.float32)
y = (x @ x).block_until_ready()
print("jax matmul sum", float(y.sum()))
print("torch", torch.__version__)
print("torch cuda available", torch.cuda.is_available())
PY
