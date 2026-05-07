#!/usr/bin/env bash
# Submit with an explicit GPU class, for example:
#   sbatch --gres=gpu:a100:1 jobs/quest_boltz2_load_smoke.slurm
#   sbatch --gres=gpu:h100:1 jobs/quest_boltz2_load_smoke.slurm
# This smoke verifies that the shared Boltz2 cache is complete enough for
# no-download model initialization on a compute node. It does not run diffusion.
#SBATCH --account=p32572
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sch-boltz2-load
#SBATCH --output=/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/slurm-boltz2-load-%j.out

set -euo pipefail

export UV_CACHE_DIR=/projects/p32572/Jieke/.cache/uv
export UV_PYTHON_INSTALL_DIR=/projects/p32572/Jieke/Envs/uv-python
export PYTHONNOUSERSITE=1
export BOLTZ_CACHE=/projects/p32572/Jieke/.cache/boltz
export JAX_PLATFORMS=cuda,cpu

cd /projects/p32572/Jieke/Projects/SCH-BinderDesign/code

echo "node=$(hostname)"
if command -v git >/dev/null 2>&1; then
  echo "commit=$(git rev-parse --short HEAD)"
else
  echo "commit=unavailable-git-not-on-path"
fi
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader

/home/zpt6685/.local/bin/uv run python - <<'PY'
from pathlib import Path
import os
import time

import equinox as eqx
import jax
import torch

from mosaic.losses.boltz2 import load_boltz2

cache = Path(os.environ["BOLTZ_CACHE"])
checkpoint = cache / "boltz2_conf.ckpt"
required = [
    checkpoint,
    cache / "boltz2_aff.ckpt",
    cache / "mols",
]

print("jax", jax.__version__)
print("jax devices", jax.devices())
print("torch", torch.__version__, "torch_cuda_available", torch.cuda.is_available())
for path in required:
    if not path.exists():
        raise FileNotFoundError(f"missing required Boltz2 cache asset: {path}")
    if path.is_file():
        print(f"asset {path} size_bytes={path.stat().st_size}")
    else:
        print(f"asset {path} dir_entries={sum(1 for _ in path.iterdir())}")

start = time.time()
model = load_boltz2(checkpoint)
jax.block_until_ready(model)
elapsed = time.time() - start

array_tree = eqx.filter(model, eqx.is_array)
leaves = [x for x in jax.tree_util.tree_leaves(array_tree) if hasattr(x, "shape")]
num_params = sum(x.size for x in leaves)
num_bytes = sum(x.size * x.dtype.itemsize for x in leaves)
devices = sorted({str(x.device) for x in leaves[:20]})

print(f"load_ok elapsed_sec={elapsed:.2f}")
print(f"array_leaves={len(leaves)} params={num_params} bytes={num_bytes}")
print(f"sample_devices={devices}")
print("boltz2_no_download_load_smoke=PASS")
PY
