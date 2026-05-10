from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import torch

from mosaic.common import TOKENS
from mosaic.losses.boltz2 import (
    boltz2_forward_from_trunk,
    boltz2_trunk,
    load_boltz2,
    load_features_and_structure_writer,
    set_binder_sequence,
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


def sequence_to_one_hot(sequence: str) -> jax.Array:
    indices = np.array([TOKENS.index(residue) for residue in sequence], dtype=np.int32)
    return jax.nn.one_hot(indices, len(TOKENS))


def candidate_sequences(length: int, seed: int) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    candidates = [
        ("poly_g", "G" * length),
        ("poly_a", "A" * length),
        ("poly_s", "S" * length),
    ]
    for idx in range(2):
        sequence = "".join(rng.choice(list(TOKENS), size=length))
        candidates.append((f"random_{idx + 1}", sequence))
    return candidates


def finite_metrics(output, binder_len: int) -> dict[str, float | bool | list[int]]:
    inter_pae = output.pae[:binder_len, binder_len:]
    inter_dist_logits = output.distogram_logits[:binder_len, binder_len:]
    dist_probs = jax.nn.softmax(inter_dist_logits, axis=-1)
    contact_prob_8a = dist_probs[..., output.distogram_bins < 8.0].sum(-1)
    contact_prob_12a = dist_probs[..., output.distogram_bins < 12.0].sum(-1)

    return {
        "distogram_shape": list(output.distogram_logits.shape),
        "plddt_mean": float(jnp.mean(output.plddt)),
        "plddt_binder_mean": float(jnp.mean(output.plddt[:binder_len])),
        "plddt_target_mean": float(jnp.mean(output.plddt[binder_len:])),
        "pae_mean": float(jnp.mean(output.pae)),
        "inter_pae_mean": float(jnp.mean(inter_pae)),
        "inter_pae_min": float(jnp.min(inter_pae)),
        "inter_contact_prob_8a_mean": float(jnp.mean(contact_prob_8a)),
        "inter_contact_prob_12a_mean": float(jnp.mean(contact_prob_12a)),
        "distogram_finite": bool(jnp.isfinite(output.distogram_logits).all()),
        "plddt_finite": bool(jnp.isfinite(output.plddt).all()),
        "pae_finite": bool(jnp.isfinite(output.pae).all()),
        "structure_finite": bool(jnp.isfinite(output.structure_coordinates).all()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a small set of binder sequence candidates with finite Boltz2 smoke metrics."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--pdb-file", default="4ZQK.cif")
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-length", type=int, default=32)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.cache / "boltz2_conf.ckpt"
    target_path = args.raw_dir / args.pdb_file

    print("node", os.uname().nodename, flush=True)
    print("jax", jax.__version__, flush=True)
    print("jax devices", jax.devices(), flush=True)
    print("torch", torch.__version__, "torch_cuda_available", torch.cuda.is_available(), flush=True)

    binder = TargetChain(sequence="G" * args.binder_length, use_msa=False)
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
    base_features, _ = build_features(chains, args.cache)
    jax.block_until_ready(base_features)
    print(f"features_ok elapsed_sec={time.time() - feature_start:.2f}", flush=True)
    print("feature_token_index_shape", tuple(np.asarray(base_features["token_index"]).shape), flush=True)
    print("feature_atom_pad_mask_shape", tuple(np.asarray(base_features["atom_pad_mask"]).shape), flush=True)

    load_start = time.time()
    model = load_boltz2(checkpoint)
    jax.block_until_ready(model)
    print(f"load_ok elapsed_sec={time.time() - load_start:.2f}", flush=True)

    results = []
    for candidate_idx, (name, sequence) in enumerate(candidate_sequences(args.binder_length, args.seed)):
        candidate_start = time.time()
        one_hot = sequence_to_one_hot(sequence)
        features = set_binder_sequence(one_hot, base_features)

        initial_embedding, trunk_state = boltz2_trunk(
            model,
            features,
            recycling_steps=args.recycling_steps,
            deterministic=True,
            key=jax.random.key(args.seed + candidate_idx),
        )
        output = boltz2_forward_from_trunk(
            model,
            features,
            initial_embedding,
            trunk_state,
            num_sampling_steps=args.sampling_steps,
            deterministic=True,
            key=jax.random.key(args.seed + 1000 + candidate_idx),
        )
        jax.block_until_ready(output)

        row = {
            "name": name,
            "sequence": sequence,
            "elapsed_sec": round(time.time() - candidate_start, 2),
            **finite_metrics(output, args.binder_length),
        }
        row["finite_scoring_ok"] = bool(row["distogram_finite"] and row["plddt_finite"] and row["pae_finite"])
        results.append(row)
        print("RESULT_JSON", json.dumps(row, sort_keys=True), flush=True)

    if not all(row["finite_scoring_ok"] for row in results):
        raise RuntimeError("At least one candidate produced non-finite finite scoring outputs")

    ranked = sorted(
        results,
        key=lambda row: (row["inter_pae_mean"], -row["inter_contact_prob_12a_mean"]),
    )
    print("RANKED_JSON", json.dumps(ranked, sort_keys=True), flush=True)
    print("boltz2_candidate_scoring_smoke=PASS", flush=True)


if __name__ == "__main__":
    main()
