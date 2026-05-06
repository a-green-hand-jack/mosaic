#!/usr/bin/env python3
"""Run a minimal Protenix-backed update-geometry smoke experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import gemmi
import jax
import jax.numpy as jnp
import numpy as np

import protenix.backend as protenix_backend

from mosaic.common import LossTerm, TOKENS
from mosaic.losses import structure_prediction as sp
from mosaic.losses.trigram import TrigramLL
from mosaic.structure_prediction import TargetChain

from run_update_geometry_diagnostic import (
    ChargeTarget,
    HYDROPHOBIC,
    MethodSpec,
    POSITIVE,
    NEGATIVE,
    SolubilityHydrophobicLimit,
    entropy,
    git_metadata,
    run_single_method_with_terminal,
    token_indicator,
    write_csv,
)


METHODS = [
    MethodSpec("M1", "single_protenix_contact"),
    MethodSpec("M3", "naive_weighted"),
    MethodSpec("M4", "normalized_weighted"),
    MethodSpec("M6", "soft_cone_correction"),
    MethodSpec("M7", "contact_preserving_soft_cone"),
]


def configure_protenix_cache(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    protenix_backend._CACHE_DIR = str(path)
    protenix_backend.download_data()

    from protenix.configs.configs_data import data_configs
    from protenix.data import ccd

    components_file = path / "components.v20240608.cif"
    rdkit_file = path / "components.v20240608.cif.rdkit_mol.pkl"
    cluster_file = path / "clusters-by-entity-40.txt"
    data_configs["ccd_components_file"] = str(components_file)
    data_configs["ccd_components_rdkit_mol_file"] = str(rdkit_file)
    data_configs["pdb_cluster_file"] = str(cluster_file)
    ccd.COMPONENTS_FILE = str(components_file)
    ccd.RKDIT_MOL_PKL = rdkit_file
    ccd.biotite_load_ccd_cif.cache_clear()
    ccd.get_component_atom_array.cache_clear()


def read_target_sequence(path: Path, target_length: int) -> str:
    structure = gemmi.read_structure(str(path))
    structure.remove_ligands_and_waters()
    structure.remove_empty_chains()
    sequence = gemmi.one_letter_code([res.name for res in structure[0][0]])
    return sequence[:target_length]


def build_loss_context(args: argparse.Namespace):
    configure_protenix_cache(args.protenix_cache)

    from mosaic.models.protenix import ProtenixMini

    target_sequence = read_target_sequence(args.target_structure, args.target_length)
    model = ProtenixMini()
    features, _writer = model.binder_features(
        binder_length=args.binder_length,
        chains=[TargetChain(sequence=target_sequence, use_msa=False)],
    )
    features = jax.tree.map(
        lambda value: jnp.asarray(value) if isinstance(value, np.ndarray) else value,
        features,
    )
    protenix_contact = model.build_loss(
        loss=sp.BinderTargetContact(contact_distance=args.contact_distance),
        features=features,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
    )

    hydrophobic_weights = token_indicator(HYDROPHOBIC)
    charge_weights = token_indicator(POSITIVE) - token_indicator(NEGATIVE)
    sequence_losses = {
        "solubility_limit": SolubilityHydrophobicLimit(
            hydrophobic_weights,
            target=0.30,
            sharpness=24.0,
        ),
        "charge_target": ChargeTarget(charge_weights, target_per_residue=0.06),
        "trigram_naturalness": TrigramLL.from_pkl(args.trigram_path),
    }
    design_losses = {
        "protenix_contact": protenix_contact,
        **sequence_losses,
    }
    return model, features, design_losses, sequence_losses


def build_losses(args: argparse.Namespace) -> dict[str, LossTerm]:
    _model, _features, design_losses, _sequence_losses = build_loss_context(args)
    return design_losses


def scalar_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(()))


def one_hot_argmax(sequence: jax.Array) -> jax.Array:
    return jax.nn.one_hot(jnp.argmax(sequence, axis=-1), len(TOKENS))


def sequence_string(sequence: jax.Array) -> str:
    indices = np.asarray(jnp.argmax(sequence, axis=-1))
    return "".join(TOKENS[int(index)] for index in indices)


def eval_sequence_loss(loss: LossTerm, sequence: jax.Array, key: jax.Array) -> tuple[float, dict[str, float]]:
    value, aux = loss(sequence, key=key)
    metrics = {}
    for key_path, aux_value in jax.tree_util.tree_leaves_with_path(aux):
        if hasattr(aux_value, "shape") and aux_value.shape == ():
            name = "_".join(str(part.key) for part in key_path if hasattr(part, "key"))
            metrics[name] = scalar_float(aux_value)
    return scalar_float(value), metrics


def score_candidate(
    *,
    model,
    features,
    sequence_losses: dict[str, LossTerm],
    terminal_sequence: jax.Array,
    method: MethodSpec,
    seed: int,
    score_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if score_mode == "argmax":
        candidate = one_hot_argmax(terminal_sequence)
    elif score_mode == "soft":
        candidate = terminal_sequence
    else:
        raise ValueError(score_mode)

    key = jax.random.key(seed + 1_000_000)
    output = model.model_output(
        PSSM=candidate,
        features=features,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        key=key,
    )

    row: dict[str, Any] = {
        "method_id": method.method_id,
        "method": method.name,
        "seed": seed,
        "score_mode": score_mode,
        "sequence": sequence_string(candidate),
        "sequence_entropy": entropy(candidate),
    }
    structure_metrics = {
        "protenix_contact": sp.BinderTargetContact(contact_distance=args.contact_distance),
        "plddt": sp.PLDDTLoss(),
        "within_binder_pae": sp.WithinBinderPAE(),
        "binder_target_pae": sp.BinderTargetPAE(),
        "target_binder_pae": sp.TargetBinderPAE(),
        "iptm": sp.IPTMLoss(),
        "binder_target_iptm": sp.BinderTargetIPTM(),
        "ipsae_min": sp.IPSAE_min(),
        "binder_ptm": sp.BinderPTMLoss(),
    }
    for name, loss in structure_metrics.items():
        value, aux = loss(sequence=candidate, output=output, key=jax.random.fold_in(key, len(row)))
        row[f"candidate_loss_{name}"] = scalar_float(value)
        for aux_name, aux_value in aux.items():
            row[f"candidate_{aux_name}"] = scalar_float(aux_value)

    for idx, (name, loss) in enumerate(sequence_losses.items()):
        value, aux = eval_sequence_loss(loss, candidate, jax.random.fold_in(key, idx + 100))
        row[f"candidate_loss_{name}"] = value
        for aux_name, aux_value in aux.items():
            row[f"candidate_{name}_{aux_name}"] = aux_value

    return row


def summarize_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    summaries = []
    metric_keys = sorted(
        key
        for key in rows[0]
        if key.startswith("candidate_")
        and key not in {"candidate_sequence"}
        and isinstance(rows[0][key], (int, float))
    )
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary: dict[str, Any] = {
            "method": method,
            "num_candidates": len(method_rows),
        }
        for key in metric_keys:
            summary[f"mean_{key}"] = float(np.mean([row[key] for row in method_rows]))
        summaries.append(summary)
    return sorted(summaries, key=lambda row: row.get("mean_candidate_bt_pae", float("inf")))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        final_rows = []
        for seed in sorted({row["seed"] for row in method_rows}):
            seed_rows = [row for row in method_rows if row["seed"] == seed]
            final_rows.append(max(seed_rows, key=lambda row: row["step"]))
        summary = {
            "method": method,
            "num_rows": len(method_rows),
            "mean_oracle_harm_rate": float(np.mean([r["oracle_harm_rate"] for r in method_rows])),
            "mean_worst_oracle_directional_derivative": float(
                np.mean([r["worst_oracle_directional_derivative"] for r in method_rows])
            ),
            "mean_step_norm": float(np.mean([r["step_norm"] for r in method_rows])),
            "mean_final_entropy": float(np.mean([r["sequence_entropy"] for r in final_rows])),
        }
        for key in sorted(k for k in final_rows[0] if k.startswith("loss_")):
            summary[f"mean_final_{key}"] = float(np.mean([r[key] for r in final_rows]))
        summaries.append(summary)
    return sorted(summaries, key=lambda row: row["mean_oracle_harm_rate"])


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 0 Protenix Update-Geometry Smoke",
        "",
        f"Run ID: `{payload['run_id']}`",
        "",
        "## Scope",
        "",
        "This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.",
        "",
        "## Summary",
        "",
        "| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {method} | {mean_oracle_harm_rate:.3f} | {mean_worst_oracle_directional_derivative:.4f} | {mean_step_norm:.4f} | {mean_final_entropy:.3f} | {mean_final_loss_protenix_contact:.4f} | {mean_final_loss_solubility_limit:.4f} | {mean_final_loss_charge_target:.4f} | {mean_final_loss_trigram_naturalness:.4f} |".format(
                **row
            )
        )
    if payload.get("candidate_summary"):
        lines.extend(
            [
                "",
                "## Candidate Holdout Scores",
                "",
                "| Method | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["candidate_summary"]:
            lines.append(
                "| {method} | {num_candidates} | {mean_candidate_plddt:.4f} | {mean_candidate_bt_pae:.4f} | {mean_candidate_bt_iptm:.4f} | {mean_candidate_ipsae_min:.4f} | {mean_candidate_loss_protenix_contact:.4f} | {mean_candidate_loss_trigram_naturalness:.4f} |".format(
                    **row
                )
            )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-structure", type=Path, default=Path("IL7RA.cif"))
    parser.add_argument("--target-length", type=int, default=48)
    parser.add_argument("--binder-length", type=int, default=24)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--init-temperature", type=float, default=0.5)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=1)
    parser.add_argument("--contact-distance", type=float, default=20.0)
    parser.add_argument("--trigram-path", type=Path, default=Path("trigram_seg.pkl"))
    parser.add_argument("--protenix-cache", type=Path, default=Path("/projects/p32572/Jieke/.cache/protenix"))
    parser.add_argument("--candidate-score-mode", choices=["argmax", "soft", "both"], default="argmax")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/reports"))
    args = parser.parse_args()

    model, features, losses, sequence_losses = build_loss_context(args)
    weights = {name: 1.0 for name in losses}
    rows = []
    candidate_rows = []
    score_modes = ["argmax", "soft"] if args.candidate_score_mode == "both" else [args.candidate_score_mode]
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for method in METHODS:
            method_rows, terminal_sequence = run_single_method_with_terminal(
                method=method,
                losses=losses,
                weights=weights,
                seed=seed,
                binder_length=args.binder_length,
                steps=args.steps,
                step_size=args.step_size,
                init_temperature=args.init_temperature,
            )
            rows.extend(method_rows)
            for score_mode in score_modes:
                candidate_rows.append(
                    score_candidate(
                        model=model,
                        features=features,
                        sequence_losses=sequence_losses,
                        terminal_sequence=terminal_sequence,
                        method=method,
                        seed=seed,
                        score_mode=score_mode,
                        args=args,
                    )
                )

    metadata = git_metadata()
    run_id = "phase0_protenix_update_geometry_{}_{}".format(
        metadata["short_commit"] or "nogit",
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    payload = {
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": metadata,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "oracles": list(losses),
        "methods": [method.__dict__ for method in METHODS],
        "summary": summarize(rows),
        "candidate_summary": summarize_candidates(candidate_rows),
    }

    json_path = args.output_dir / f"{run_id}.json"
    csv_path = args.output_dir / f"{run_id}_steps.csv"
    candidates_csv_path = args.output_dir / f"{run_id}_candidates.csv"
    report_path = args.report_dir / f"{run_id}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_csv(candidates_csv_path, candidate_rows)
    write_report(report_path, payload)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "json": str(json_path),
                "csv": str(csv_path),
                "candidates_csv": str(candidates_csv_path),
                "report": str(report_path),
                "summary": payload["summary"],
                "candidate_summary": payload["candidate_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
