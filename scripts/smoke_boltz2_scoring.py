from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import torch

from mosaic.losses.boltz2 import (
    boltz2_forward_from_trunk,
    boltz2_trunk,
    load_boltz2,
    load_features_and_structure_writer,
)
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


def build_features(chains: list[TargetChain], cache: Path):
    yaml = "\n".join(
        [_prefix()]
        + [chain_yaml(chain_id, chain) for chain_id, chain in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)]
    )
    template_handle, template_yaml = build_template_yaml("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
    if template_handle is not None:
        yaml += template_yaml
    try:
        features, writer = load_features_and_structure_writer(yaml, cache=cache)
    finally:
        if template_handle is not None:
            template_handle.close()
    return features, writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test Boltz2 binder+target scoring through tiny diffusion/confidence."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--pdb-file", default="4ZQK.cif")
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-length", type=int, default=32)
    parser.add_argument("--binder-residue", default="G")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.cache / "boltz2_conf.ckpt"
    target_path = args.raw_dir / args.pdb_file

    print("node", os.uname().nodename, flush=True)
    print("jax", jax.__version__, flush=True)
    print("jax devices", jax.devices(), flush=True)
    print("torch", torch.__version__, "torch_cuda_available", torch.cuda.is_available(), flush=True)

    binder = TargetChain(
        sequence=args.binder_residue * args.binder_length,
        use_msa=False,
    )
    target = make_target_chain(target_path, args.target_chain)
    chains = [binder, target]
    print(
        "binder_len",
        args.binder_length,
        "target_len",
        len(target.sequence),
        "total_len",
        args.binder_length + len(target.sequence),
        flush=True,
    )

    feature_start = time.time()
    features, _ = build_features(chains, args.cache)
    jax.block_until_ready(features)
    print(f"features_ok elapsed_sec={time.time() - feature_start:.2f}", flush=True)
    print("feature_token_index_shape", tuple(np.asarray(features["token_index"]).shape), flush=True)
    print("feature_atom_pad_mask_shape", tuple(np.asarray(features["atom_pad_mask"]).shape), flush=True)
    print("unique_asym_id", np.unique(np.asarray(features["asym_id"])).tolist(), flush=True)

    load_start = time.time()
    model = load_boltz2(checkpoint)
    jax.block_until_ready(model)
    print(f"load_ok elapsed_sec={time.time() - load_start:.2f}", flush=True)

    trunk_start = time.time()
    initial_embedding, trunk_state = boltz2_trunk(
        model,
        features,
        recycling_steps=args.recycling_steps,
        deterministic=True,
        key=jax.random.key(args.seed),
    )
    jax.block_until_ready((initial_embedding, trunk_state))
    print(f"trunk_ok elapsed_sec={time.time() - trunk_start:.2f}", flush=True)

    scoring_start = time.time()
    output = boltz2_forward_from_trunk(
        model,
        features,
        initial_embedding,
        trunk_state,
        num_sampling_steps=args.sampling_steps,
        deterministic=True,
        key=jax.random.key(args.seed + 100),
    )
    jax.block_until_ready(output)
    print(f"scoring_ok elapsed_sec={time.time() - scoring_start:.2f}", flush=True)

    binder_len = args.binder_length
    inter_pae = output.pae[:binder_len, binder_len:]
    inter_dist_logits = output.distogram_logits[:binder_len, binder_len:]
    dist_probs = jax.nn.softmax(inter_dist_logits, axis=-1)
    contact_prob_12a = dist_probs[..., output.distogram_bins < 12.0].sum(-1)

    print("distogram_logits_shape", tuple(output.distogram_logits.shape), flush=True)
    print("plddt_shape", tuple(output.plddt.shape), "plddt_mean", float(jnp.mean(output.plddt)), flush=True)
    print("pae_shape", tuple(output.pae.shape), "pae_mean", float(jnp.mean(output.pae)), flush=True)
    print("inter_pae_mean", float(jnp.mean(inter_pae)), flush=True)
    print("inter_contact_prob_12a_mean", float(jnp.mean(contact_prob_12a)), flush=True)
    print("structure_coordinates_shape", tuple(output.structure_coordinates.shape), flush=True)
    print("backbone_coordinates_shape", tuple(output.backbone_coordinates.shape), flush=True)
    print("distogram_finite", bool(jnp.isfinite(output.distogram_logits).all()), flush=True)
    print("plddt_finite", bool(jnp.isfinite(output.plddt).all()), flush=True)
    print("pae_finite", bool(jnp.isfinite(output.pae).all()), flush=True)
    print("structure_finite", bool(jnp.isfinite(output.structure_coordinates).all()), flush=True)
    print("boltz2_scoring_smoke=PASS", flush=True)


if __name__ == "__main__":
    main()
