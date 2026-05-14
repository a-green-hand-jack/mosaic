#!/usr/bin/env python3
"""Run a reduced Protenix-vs-Boltz2 oracle-gradient conflict experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import gemmi
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import LossTerm
from mosaic.losses.boltz2 import (
    BOLTZ2_DISTOGRAM_BINS,
    boltz2_trunk,
    load_boltz2,
    load_features_and_structure_writer,
    set_binder_sequence,
)
from mosaic.models.boltz2 import _prefix, build_template_yaml, chain_yaml
from mosaic.structure_prediction import TargetChain

from run_protenix_update_geometry_smoke import build_loss_context
from run_update_geometry_diagnostic import (
    MethodSpec,
    git_metadata,
    run_single_method,
    write_csv,
)


AA3 = set(
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER "
        "THR TRP TYR VAL MSE SEC PYL ASX GLX UNK"
    ).split()
)


METHODS = [
    MethodSpec("M3", "naive_weighted"),
    MethodSpec("M4", "normalized_weighted"),
    MethodSpec(
        "M12a",
        "active_constraint_qp_grid",
        aux_slack=0.04,
        min_primary_descent_ratio=0.75,
        cone_denominator=10,
        active_constraint_loss_thresholds={
            "solubility_limit": 0.04,
            "charge_target": 0.01,
        },
    ),
    MethodSpec(
        "M13a",
        "active_constraint_contact_warmup",
        aux_slack=0.04,
        min_primary_descent_ratio=0.75,
        cone_denominator=10,
        active_constraint_loss_thresholds={
            "solubility_limit": 0.04,
            "charge_target": 0.01,
        },
        contact_warmup_steps=1,
    ),
    MethodSpec(
        "M12b",
        "active_constraint_qp_grid",
        aux_slack=0.04,
        min_primary_descent_ratio=0.85,
        cone_denominator=10,
        active_constraint_loss_thresholds={
            "solubility_limit": 0.08,
            "charge_target": 0.01,
        },
    ),
    MethodSpec(
        "M12c",
        "active_constraint_qp_grid",
        aux_slack=0.04,
        min_primary_descent_ratio=0.90,
        cone_denominator=10,
        active_constraint_loss_thresholds={
            "solubility_limit": 0.12,
            "charge_target": 0.01,
        },
    ),
    MethodSpec(
        "M12d",
        "active_constraint_qp_grid",
        aux_slack=0.04,
        min_primary_descent_ratio=0.90,
        cone_denominator=10,
        active_constraint_loss_thresholds={
            "solubility_limit": 1.0,
            "charge_target": 0.01,
        },
    ),
    MethodSpec(
        "M7c",
        "contact_preserving_soft_cone",
        aux_slack=0.08,
        max_aux_harms=1,
        min_primary_descent_ratio=0.60,
        cone_denominator=10,
    ),
    MethodSpec(
        "M8a",
        "contact_qp_grid",
        aux_slack=0.08,
        max_aux_harms=0,
        min_primary_descent_ratio=0.60,
        cone_denominator=10,
    ),
    MethodSpec(
        "M10a",
        "balanced_zero_harm_cone",
        aux_slack=0.0,
        max_aux_harms=0,
        cone_denominator=10,
    ),
    MethodSpec("M11a", "pcgrad_normalized"),
]


class Boltz2DistogramContactLoss(LossTerm):
    joltz2: Any
    features: Any
    binder_length: int = eqx.field(converter=int)
    contact_distance: float = eqx.field(converter=float)
    recycling_steps: int = eqx.field(converter=int)
    deterministic: bool = True

    def __call__(self, sequence, *, key):
        features = set_binder_sequence(sequence, self.features)
        _initial_embedding, trunk_state = boltz2_trunk(
            self.joltz2,
            features,
            recycling_steps=self.recycling_steps,
            deterministic=self.deterministic,
            key=key,
        )
        logits = self.joltz2.distogram_module(trunk_state.z)[0, :, :, 0, :]
        inter_logits = logits[: self.binder_length, self.binder_length :]
        dist_probs = jax.nn.softmax(inter_logits, axis=-1)
        contact_prob = dist_probs[..., BOLTZ2_DISTOGRAM_BINS < self.contact_distance].sum(-1)
        mean_contact = jnp.mean(contact_prob)
        return -mean_contact, {
            "contact_prob_mean": mean_contact,
            "contact_prob_max": jnp.max(contact_prob),
            "distogram_logits_finite": jnp.all(jnp.isfinite(logits)),
        }


def clean_chain(path: Path, chain_id: str, target_length: int | None) -> gemmi.Chain:
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    source_chain = structure[0][chain_id]
    chain = gemmi.Chain(chain_id)
    for residue in source_chain:
        if residue.entity_type == gemmi.EntityType.Polymer and residue.name in AA3:
            chain.add_residue(residue.clone())
            if target_length is not None and len(chain) >= target_length:
                break
    if len(chain) == 0:
        raise ValueError(f"{path}:{chain_id} has no polymer protein residues")
    return chain


def make_target_chain(path: Path, chain_id: str, target_length: int | None) -> TargetChain:
    chain = clean_chain(path, chain_id, target_length)
    sequence = gemmi.one_letter_code([residue.name for residue in chain])
    return TargetChain(sequence=sequence, use_msa=False, template_chain=chain)


def build_boltz2_features(chains: list[TargetChain], cache: Path):
    yaml = "\n".join(
        [_prefix()]
        + [
            chain_yaml(chain_id, chain)
            for chain_id, chain in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
        ]
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


def method_allowlist(raw: str) -> list[MethodSpec]:
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    methods = [method for method in METHODS if method.method_id in wanted]
    missing = wanted - {method.method_id for method in methods}
    if missing:
        raise ValueError(f"Unknown method ids: {sorted(missing)}")
    return methods


def build_losses(args: argparse.Namespace) -> dict[str, LossTerm]:
    _model, _features, protenix_losses, sequence_losses = build_loss_context(args)
    checkpoint = args.boltz_cache / "boltz2_conf.ckpt"
    binder = TargetChain(sequence="G" * args.binder_length, use_msa=False)
    target = make_target_chain(args.target_structure, args.target_chain, args.target_length)
    boltz_features, _writer = build_boltz2_features([binder, target], args.boltz_cache)
    boltz_model = load_boltz2(checkpoint)

    return {
        "protenix_contact": protenix_losses["protenix_contact"],
        "boltz2_distogram_contact": Boltz2DistogramContactLoss(
            joltz2=boltz_model,
            features=boltz_features,
            binder_length=args.binder_length,
            contact_distance=args.boltz_contact_distance,
            recycling_steps=args.boltz_recycling_steps,
        ),
        **sequence_losses,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "method_id": method_rows[0]["method_id"],
                "num_rows": len(method_rows),
                "mean_oracle_harm_rate": float(
                    np.mean([r["oracle_harm_rate"] for r in method_rows])
                ),
                "mean_worst_oracle_directional_derivative": float(
                    np.mean([r["worst_oracle_directional_derivative"] for r in method_rows])
                ),
                "mean_protenix_directional_derivative": float(
                    np.mean([r["dir_protenix_contact"] for r in method_rows])
                ),
                "mean_boltz2_directional_derivative": float(
                    np.mean([r["dir_boltz2_distogram_contact"] for r in method_rows])
                ),
                "mean_protenix_boltz2_cosine": float(
                    np.mean(
                        [
                            r["cos_protenix_contact__boltz2_distogram_contact"]
                            for r in method_rows
                        ]
                    )
                ),
                "mean_step_norm": float(np.mean([r["step_norm"] for r in method_rows])),
            }
        )
    return sorted(summaries, key=lambda row: row["mean_oracle_harm_rate"])


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke",
        "",
        f"Run ID: `{payload['run_id']}`",
        "",
        "## Scope",
        "",
        "This reduced Phase 0 experiment directly measures gradient conflict "
        "between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. "
        "It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` "
        "are currently non-finite.",
        "",
        "## Summary",
        "",
        "| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | "
        "Protenix/Boltz2 cosine | Step norm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {method_id} / {method} | {mean_oracle_harm_rate:.3f} | "
            "{mean_worst_oracle_directional_derivative:.4f} | "
            "{mean_protenix_directional_derivative:.4f} | "
            "{mean_boltz2_directional_derivative:.4f} | "
            "{mean_protenix_boltz2_cosine:.4f} | {mean_step_norm:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.",
            "- A positive directional derivative means the projected update locally "
            "harms that oracle.",
            "- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{payload['json_path']}`",
            f"- CSV: `{payload['csv_path']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binder-length", type=int, default=24)
    parser.add_argument("--target-structure", type=Path, default=Path("IL7RA.cif"))
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--target-length", type=int, default=48)
    parser.add_argument("--trigram-path", type=Path, default=Path("trigram_seg.pkl"))
    parser.add_argument(
        "--protenix-cache",
        type=Path,
        default=Path("/projects/p32572/Jieke/.cache/protenix"),
    )
    parser.add_argument(
        "--boltz-cache",
        type=Path,
        default=Path("/projects/p32572/Jieke/.cache/boltz"),
    )
    parser.add_argument("--contact-distance", type=float, default=20.0)
    parser.add_argument("--boltz-contact-distance", type=float, default=12.0)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=1)
    parser.add_argument("--boltz-recycling-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--init-temperature", type=float, default=0.5)
    parser.add_argument("--method-ids", default="M3,M4,M7c,M8a")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/reports"))
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = method_allowlist(args.method_ids)
    losses = build_losses(args)
    weights = {name: 1.0 for name in losses}

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for method in methods:
            rows.extend(
                run_single_method(
                    method=method,
                    losses=losses,
                    weights=weights,
                    seed=seed,
                    binder_length=args.binder_length,
                    steps=args.steps,
                    step_size=args.step_size,
                    init_temperature=args.init_temperature,
                    snapshot_dir=args.snapshot_dir,
                    snapshot_context={
                        "label": f"{args.target_structure.stem}_{args.target_chain}",
                        "target_structure": str(args.target_structure),
                        "target_chain": args.target_chain,
                        "target_length": args.target_length,
                        "binder_length": args.binder_length,
                    },
                )
            )

    run_id = "phase0_boltz2_oracle_gradient_conflict_{}_{}".format(
        git_metadata()["short_commit"] or "nogit",
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    json_path = args.output_dir / f"{run_id}.json"
    csv_path = args.output_dir / f"{run_id}_steps.csv"
    report_path = args.report_dir / f"{run_id}.md"
    payload = {
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_metadata(),
        "args": {key: str(value) for key, value in vars(args).items()},
        "oracles": list(losses),
        "methods": [asdict(method) for method in methods],
        "summary": summarize(rows),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_report(report_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
