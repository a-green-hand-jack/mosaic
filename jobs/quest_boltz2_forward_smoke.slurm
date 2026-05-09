#!/usr/bin/env bash
# Submit with an explicit GPU class, for example:
#   sbatch --gres=gpu:a100:1 jobs/quest_boltz2_forward_smoke.slurm
#   sbatch --gres=gpu:h100:1 jobs/quest_boltz2_forward_smoke.slurm
# This smoke verifies raw mmCIF -> Boltz2 features -> one recycling trunk forward.
# It does not run diffusion sampling.
#SBATCH --account=p32572
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sch-boltz2-fwd
#SBATCH --output=/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/slurm-boltz2-fwd-%j.out

set -euo pipefail

export UV_CACHE_DIR=/projects/p32572/Jieke/.cache/uv
export UV_PYTHON_INSTALL_DIR=/projects/p32572/Jieke/Envs/uv-python
export PYTHONNOUSERSITE=1
export BOLTZ_CACHE=/projects/p32572/Jieke/.cache/boltz
export RAW_TARGET_DIR=/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import torch

from mosaic.losses.boltz2 import boltz2_trunk, load_boltz2, load_features_and_structure_writer
from mosaic.models.boltz2 import _prefix, build_template_yaml, chain_yaml
from mosaic.structure_prediction import TargetChain

AA3 = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE SEC PYL ASX GLX UNK".split()
)


def clean_chain(path: Path, chain_id: str) -> gemmi.Chain:
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    source_chain = structure[0][chain_id]
    chain = gemmi.Chain(chain_id)
    for residue in source_chain:
        if residue.entity_type == gemmi.EntityType.Polymer and residue.name in AA3:
            chain.add_residue(residue.clone())
    if len(chain) == 0:
        raise ValueError(f"{path.name}:{chain_id} has no polymer protein residues")
    return chain


def make_target_chain(path: Path, chain_id: str) -> TargetChain:
    chain = clean_chain(path, chain_id)
    sequence = gemmi.one_letter_code([residue.name for residue in chain])
    return TargetChain(sequence=sequence, use_msa=False, template_chain=chain)


def build_target_features(target: TargetChain, cache: Path):
    yaml = "\n".join([_prefix(), chain_yaml("A", target)])
    template_handle, template_yaml = build_template_yaml("A", [target])
    if template_handle is not None:
        yaml += template_yaml
    try:
        features, writer = load_features_and_structure_writer(yaml, cache=cache)
    finally:
        if template_handle is not None:
            template_handle.close()
    return features, writer


cache = Path(os.environ["BOLTZ_CACHE"])
target_path = Path(os.environ["RAW_TARGET_DIR"]) / "4ZQK.cif"
checkpoint = cache / "boltz2_conf.ckpt"

print("jax", jax.__version__)
print("jax devices", jax.devices())
print("torch", torch.__version__, "torch_cuda_available", torch.cuda.is_available())
print("target", target_path, "chain=A")

target = make_target_chain(target_path, "A")
print("sequence_len", len(target.sequence))

feature_start = time.time()
features, _ = build_target_features(target, cache)
jax.block_until_ready(features)
print(f"features_ok elapsed_sec={time.time() - feature_start:.2f}")
print("feature_token_index_shape", tuple(np.asarray(features["token_index"]).shape))
print("feature_atom_pad_mask_shape", tuple(np.asarray(features["atom_pad_mask"]).shape))

load_start = time.time()
model = load_boltz2(checkpoint)
jax.block_until_ready(model)
print(f"load_ok elapsed_sec={time.time() - load_start:.2f}")

forward_start = time.time()
initial_embedding, trunk_state = boltz2_trunk(
    model,
    features,
    recycling_steps=1,
    deterministic=True,
    key=jax.random.key(0),
)
jax.block_until_ready((initial_embedding, trunk_state))
print(f"trunk_forward_ok elapsed_sec={time.time() - forward_start:.2f}")
print("s_shape", tuple(trunk_state.s.shape), "s_dtype", trunk_state.s.dtype)
print("z_shape", tuple(trunk_state.z.shape), "z_dtype", trunk_state.z.dtype)
print("s_finite", bool(jnp.isfinite(trunk_state.s).all()))
print("z_finite", bool(jnp.isfinite(trunk_state.z).all()))
print("boltz2_feature_trunk_smoke=PASS")
PY
