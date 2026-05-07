#!/usr/bin/env python3
"""Run a reduced Protenix-backed CEM optimizer diagnostic."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import TOKENS

from run_update_geometry_diagnostic import MethodSpec, entropy, git_metadata, run_single_method_with_terminal, write_csv
from run_protenix_update_geometry_smoke import (
    METHODS,
    best_candidate_rows,
    build_loss_context,
    score_candidate,
    summarize,
    summarize_candidates,
    topk_sample_one_hot,
)


METRIC_DIRECTIONS = {
    "bt_pae": ("candidate_bt_pae", "min"),
    "bt_iptm": ("candidate_bt_iptm", "max"),
    "contact": ("candidate_loss_protenix_contact", "min"),
}

CEM_METHODS = [
    MethodSpec("CEMp", "cem_bt_pae"),
    MethodSpec("CEMc", "cem_contact"),
]


def metric_value(row: dict[str, Any], metric: str) -> float:
    column, _direction = METRIC_DIRECTIONS[metric]
    return float(row[column])


def sort_key_for_metric(row: dict[str, Any], metric: str) -> float:
    _column, direction = METRIC_DIRECTIONS[metric]
    value = metric_value(row, metric)
    return value if direction == "min" else -value


def cem_metric(method: MethodSpec) -> str:
    if method.method_id == "CEMc":
        return "contact"
    return "bt_pae"


def smooth_elite_distribution(elites: list[jax.Array], min_uniform_mix: float) -> jax.Array:
    elite_mean = jnp.mean(jnp.stack(elites, axis=0), axis=0)
    uniform = jnp.full_like(elite_mean, 1.0 / len(TOKENS))
    mixed = (1.0 - min_uniform_mix) * elite_mean + min_uniform_mix * uniform
    return mixed / jnp.sum(mixed, axis=-1, keepdims=True)


def score_hard_candidate(
    *,
    model,
    features,
    sequence_losses,
    candidate: jax.Array,
    method: MethodSpec,
    seed: int,
    sample_index: int,
    round_index: int,
    source: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = score_candidate(
        model=model,
        features=features,
        sequence_losses=sequence_losses,
        terminal_sequence=candidate,
        method=method,
        seed=seed,
        score_mode="soft",
        sample_index=sample_index,
        args=args,
    )
    row["score_mode"] = "topk_sample"
    row["candidate_sample_index"] = sample_index
    row["candidate_sample_top_k"] = args.candidate_sample_top_k
    row["candidate_sample_temperature"] = args.candidate_sample_temperature
    row["cem_round"] = round_index
    row["candidate_source"] = source
    return row


def run_cem_method(
    *,
    model,
    features,
    sequence_losses,
    method: MethodSpec,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    key = jax.random.key(seed + 50_000)
    distribution = jax.nn.softmax(
        args.init_temperature * jax.random.gumbel(key, shape=(args.binder_length, len(TOKENS))),
        axis=-1,
    )
    candidate_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    ranking_metric = cem_metric(method)

    for round_index in range(args.cem_rounds):
        round_start_entropy = entropy(distribution)
        scored: list[tuple[dict[str, Any], jax.Array]] = []
        for local_index in range(args.cem_samples_per_round):
            sample_index = round_index * args.cem_samples_per_round + local_index
            candidate = topk_sample_one_hot(
                distribution,
                key=jax.random.fold_in(key, sample_index + 1),
                top_k=args.candidate_sample_top_k,
                temperature=args.candidate_sample_temperature,
            )
            row = score_hard_candidate(
                model=model,
                features=features,
                sequence_losses=sequence_losses,
                candidate=candidate,
                method=method,
                seed=seed,
                sample_index=sample_index,
                round_index=round_index,
                source="cem_sample",
                args=args,
            )
            row["cem_ranking_metric"] = ranking_metric
            scored.append((row, candidate))
            candidate_rows.append(row)

        elite = sorted(scored, key=lambda item: sort_key_for_metric(item[0], ranking_metric))[: args.cem_elite_count]
        elite_distribution = smooth_elite_distribution(
            [candidate for _row, candidate in elite],
            args.cem_min_uniform_mix,
        )
        distribution = (1.0 - args.cem_update_rate) * distribution + args.cem_update_rate * elite_distribution
        distribution = distribution / jnp.sum(distribution, axis=-1, keepdims=True)

        best_row = elite[0][0]
        round_rows.append(
            {
                "method_id": method.method_id,
                "method": method.name,
                "seed": seed,
                "cem_round": round_index,
                "cem_ranking_metric": ranking_metric,
                "round_start_entropy": round_start_entropy,
                "round_end_entropy": entropy(distribution),
                "elite_count": args.cem_elite_count,
                "samples_per_round": args.cem_samples_per_round,
                "best_candidate_bt_pae": best_row["candidate_bt_pae"],
                "best_candidate_bt_iptm": best_row["candidate_bt_iptm"],
                "best_candidate_plddt": best_row["candidate_plddt"],
                "best_candidate_contact": best_row["candidate_loss_protenix_contact"],
            }
        )

    final_argmax = jax.nn.one_hot(jnp.argmax(distribution, axis=-1), len(TOKENS))
    final_row = score_hard_candidate(
        model=model,
        features=features,
        sequence_losses=sequence_losses,
        candidate=final_argmax,
        method=method,
        seed=seed,
        sample_index=args.cem_rounds * args.cem_samples_per_round,
        round_index=args.cem_rounds,
        source="cem_final_argmax",
        args=args,
    )
    final_row["score_mode"] = "argmax"
    final_row["cem_ranking_metric"] = ranking_metric
    candidate_rows.append(final_row)
    return candidate_rows, round_rows


def selected_methods(method_ids: str) -> list[MethodSpec]:
    requested = {item.strip() for item in method_ids.split(",") if item.strip()}
    methods = [method for method in METHODS if method.method_id in requested]
    if not methods:
        raise ValueError(f"No baseline methods selected by --baseline-method-ids={method_ids!r}")
    return methods


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 0 ACT-015A CEM Optimizer",
        "",
        f"Run ID: `{payload['run_id']}`",
        "",
        "## Scope",
        "",
        "This run tests whether cold top-k candidate selection can be internalized as a small CEM / elite-sampling optimizer. It compares CEM variants against matched-budget M3/M7c top-k baselines in the reduced ProtenixMini setting.",
        "",
        "## Candidate Summary",
        "",
        "| Method ID | Method | Score mode | Candidates | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["candidate_summary"]:
        lines.append(
            "| {method_id} | {method} | {score_mode} | {num_candidates} | {mean_candidate_plddt:.4f} | {mean_candidate_bt_pae:.4f} | {mean_candidate_bt_iptm:.4f} | {mean_candidate_ipsae_min:.4f} | {mean_candidate_loss_protenix_contact:.4f} | {mean_candidate_loss_trigram_naturalness:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Round Summary",
            "",
            "| Method ID | Seed | Round | Metric | Start entropy | End entropy | Best BT PAE | Best BT ipTM | Best pLDDT | Best contact |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["cem_round_summary"]:
        lines.append(
            "| {method_id} | {seed} | {cem_round} | {cem_ranking_metric} | {round_start_entropy:.4f} | {round_end_entropy:.4f} | {best_candidate_bt_pae:.4f} | {best_candidate_bt_iptm:.4f} | {best_candidate_plddt:.4f} | {best_candidate_contact:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "CEM uses hard-candidate oracle calls inside optimization. Comparisons should be made against M3/M7c top-k baselines with the same candidate scoring budget.",
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
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--init-temperature", type=float, default=0.5)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=1)
    parser.add_argument("--contact-distance", type=float, default=20.0)
    parser.add_argument("--trigram-path", type=Path, default=Path("trigram_seg.pkl"))
    parser.add_argument("--protenix-cache", type=Path, default=Path("/projects/p32572/Jieke/.cache/protenix"))
    parser.add_argument("--candidate-sample-top-k", type=int, default=4)
    parser.add_argument("--candidate-sample-temperature", type=float, default=0.25)
    parser.add_argument("--candidate-best-metric", choices=["bt_pae", "bt_iptm", "contact"], default="bt_pae")
    parser.add_argument("--baseline-method-ids", type=str, default="M3,M7c")
    parser.add_argument("--baseline-topk-samples", type=int, default=24)
    parser.add_argument("--cem-method-ids", type=str, default="CEMp,CEMc")
    parser.add_argument("--cem-rounds", type=int, default=3)
    parser.add_argument("--cem-samples-per-round", type=int, default=8)
    parser.add_argument("--cem-elite-count", type=int, default=2)
    parser.add_argument("--cem-update-rate", type=float, default=0.7)
    parser.add_argument("--cem-min-uniform-mix", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/reports"))
    args = parser.parse_args()

    model, features, losses, sequence_losses = build_loss_context(args)
    weights = {name: 1.0 for name in losses}
    step_rows = []
    candidate_rows = []
    round_rows = []

    baseline_methods = selected_methods(args.baseline_method_ids)
    requested_cem_ids = {item.strip() for item in args.cem_method_ids.split(",") if item.strip()}
    cem_methods = [method for method in CEM_METHODS if method.method_id in requested_cem_ids]
    if not cem_methods:
        raise ValueError(f"No CEM methods selected by --cem-method-ids={args.cem_method_ids!r}")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for method in baseline_methods:
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
            step_rows.extend(method_rows)
            for score_mode in ("soft", "argmax"):
                candidate_rows.append(
                    score_candidate(
                        model=model,
                        features=features,
                        sequence_losses=sequence_losses,
                        terminal_sequence=terminal_sequence,
                        method=method,
                        seed=seed,
                        score_mode=score_mode,
                        sample_index=0,
                        args=args,
                    )
                )
            for sample_index in range(args.baseline_topk_samples):
                candidate_rows.append(
                    score_candidate(
                        model=model,
                        features=features,
                        sequence_losses=sequence_losses,
                        terminal_sequence=terminal_sequence,
                        method=method,
                        seed=seed,
                        score_mode="topk_sample",
                        sample_index=sample_index,
                        args=args,
                    )
                )

        for method in cem_methods:
            rows, cem_rounds = run_cem_method(
                model=model,
                features=features,
                sequence_losses=sequence_losses,
                method=method,
                seed=seed,
                args=args,
            )
            candidate_rows.extend(rows)
            round_rows.extend(cem_rounds)

    metadata = git_metadata()
    run_id = "phase0_protenix_cem_{}_{}".format(
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
        "baseline_methods": [method.__dict__ for method in baseline_methods],
        "cem_methods": [method.__dict__ for method in cem_methods],
        "baseline_summary": summarize(step_rows),
        "candidate_summary": summarize_candidates(candidate_rows),
        "candidate_best_summary": summarize_candidates(best_candidate_rows(candidate_rows, args.candidate_best_metric)),
        "cem_round_summary": round_rows,
    }

    json_path = args.output_dir / f"{run_id}.json"
    steps_csv_path = args.output_dir / f"{run_id}_steps.csv"
    candidates_csv_path = args.output_dir / f"{run_id}_candidates.csv"
    rounds_csv_path = args.output_dir / f"{run_id}_cem_rounds.csv"
    report_path = args.report_dir / f"{run_id}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(steps_csv_path, step_rows)
    write_csv(candidates_csv_path, candidate_rows)
    write_csv(rounds_csv_path, round_rows)
    write_report(report_path, payload)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "json": str(json_path),
                "steps_csv": str(steps_csv_path),
                "candidates_csv": str(candidates_csv_path),
                "rounds_csv": str(rounds_csv_path),
                "report": str(report_path),
                "candidate_summary": payload["candidate_summary"],
                "candidate_best_summary": payload["candidate_best_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
