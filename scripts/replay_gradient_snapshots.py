#!/usr/bin/env python3
"""Replay saved oracle-gradient snapshots with alternative update rules."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from run_update_geometry_diagnostic import (
    MethodSpec,
    balanced_zero_harm_diagnostics,
    choose_raw_direction,
    contact_qp_grid_diagnostics,
    entropy,
    flatten_dot,
    grad_norm,
    offdiag_cosines,
    projected_update,
    write_csv,
)


METHODS = [
    MethodSpec("M3", "naive_weighted"),
    MethodSpec("M4", "normalized_weighted"),
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


def method_allowlist(raw: str) -> list[MethodSpec]:
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    methods = [method for method in METHODS if method.method_id in wanted]
    missing = wanted - {method.method_id for method in methods}
    if missing:
        raise ValueError(f"Unknown method ids: {sorted(missing)}")
    return methods


def load_snapshot(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        oracle_names = [str(name) for name in data["oracle_names"]]
        grad_stack = data["grad_stack"]
        grads = {
            name: jnp.asarray(grad_stack[idx])
            for idx, name in enumerate(oracle_names)
        }
        return {
            "path": path,
            "metadata": metadata,
            "x": jnp.asarray(data["x"]),
            "grads": grads,
            "weights": json.loads(str(data["weights_json"])),
        }


def replay_one(
    *,
    snapshot: dict[str, Any],
    method: MethodSpec,
    step_size: float,
) -> dict[str, Any]:
    x = snapshot["x"]
    grads = snapshot["grads"]
    weights = {name: float(snapshot["weights"].get(name, 1.0)) for name in grads}
    metadata = snapshot["metadata"]

    raw_direction = choose_raw_direction(
        method,
        grads,
        weights,
        x=x,
        step_size=step_size,
    )
    x_new, actual_update = projected_update(x, raw_direction, step_size)
    directional = {f"dir_{name}": flatten_dot(grad, actual_update) for name, grad in grads.items()}
    oracle_names = list(grads)
    primary_name = oracle_names[0]
    aux_directional = [directional[f"dir_{name}"] for name in oracle_names[1:]]
    harms = [value > 0 for value in directional.values()]
    row: dict[str, Any] = {
        "snapshot_path": str(snapshot["path"]),
        "snapshot_label": metadata.get("label", ""),
        "snapshot_source_method_id": metadata.get("method_id", ""),
        "snapshot_source_method": metadata.get("method", ""),
        "seed": metadata.get("seed", ""),
        "step": metadata.get("step", ""),
        "method_id": method.method_id,
        "method": method.name,
        "method_aux_slack": method.aux_slack,
        "method_max_aux_harms": method.max_aux_harms,
        "method_min_primary_descent_ratio": method.min_primary_descent_ratio,
        "method_cone_denominator": method.cone_denominator,
        "oracle_harm_rate": float(np.mean(harms)),
        "worst_oracle_directional_derivative": max(directional.values()),
        "primary_oracle": primary_name,
        "primary_directional_derivative": directional[f"dir_{primary_name}"],
        "aux_harm_count": sum(value > 0 for value in aux_directional),
        "aux_worst_directional_derivative": max(aux_directional) if aux_directional else 0.0,
        "mean_oracle_descent": float(np.mean(list(directional.values()))),
        "step_norm": grad_norm(actual_update),
        "sequence_entropy": entropy(x),
        "sequence_entropy_after": entropy(x_new),
        "sequence_entropy_delta": entropy(x_new) - entropy(x),
    }
    row.update(directional)
    row.update(offdiag_cosines(grads))
    if method.name in {"contact_qp_grid", "contact_qp_grid_contact_first"}:
        row.update(
            contact_qp_grid_diagnostics(
                grads=grads,
                weights=weights,
                x=x,
                actual_update=actual_update,
                step_size=step_size,
                aux_slack=method.aux_slack,
                min_primary_descent_ratio=method.min_primary_descent_ratio,
                cone_denominator=method.cone_denominator,
            )
        )
    if method.name == "balanced_zero_harm_cone":
        row.update(
            balanced_zero_harm_diagnostics(
                grads=grads,
                weights=weights,
                x=x,
                actual_update=actual_update,
                step_size=step_size,
                aux_slack=method.aux_slack,
                max_harms=method.max_aux_harms,
                cone_denominator=method.cone_denominator,
            )
        )
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for method_id in sorted({row["method_id"] for row in rows}):
        method_rows = [row for row in rows if row["method_id"] == method_id]
        summaries.append(
            {
                "method_id": method_id,
                "method": method_rows[0]["method"],
                "num_rows": len(method_rows),
                "mean_oracle_harm_rate": float(
                    np.mean([row["oracle_harm_rate"] for row in method_rows])
                ),
                "mean_worst_oracle_directional_derivative": float(
                    np.mean(
                        [row["worst_oracle_directional_derivative"] for row in method_rows]
                    )
                ),
                "mean_step_norm": float(np.mean([row["step_norm"] for row in method_rows])),
                "mean_oracle_descent": float(
                    np.mean([row["mean_oracle_descent"] for row in method_rows])
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["mean_oracle_harm_rate"],
            row["mean_worst_oracle_directional_derivative"],
        ),
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Gradient Snapshot Replay",
        "",
        f"Run ID: `{payload['run_id']}`",
        "",
        "## Summary",
        "",
        "| Method | Harm rate | Worst derivative | Mean descent | Step norm | Rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {method_id} / {method} | {mean_oracle_harm_rate:.3f} | "
            "{mean_worst_oracle_directional_derivative:.4f} | "
            "{mean_oracle_descent:.4f} | {mean_step_norm:.4f} | {num_rows} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This replay uses saved `x_t` and oracle gradients only; "
            "it does not rerun Protenix or Boltz2.",
            "- Passing replay is a cheap precondition for a real oracle run, "
            "not paper evidence by itself.",
            "",
            "## Inputs",
            "",
        ]
    )
    for item in payload["snapshots"]:
        lines.append(f"- `{item}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--method-ids", default="M3,M4,M7c,M8a,M10a,M11a")
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = method_allowlist(args.method_ids)
    snapshots = [load_snapshot(path) for path in args.snapshots]
    rows = [
        replay_one(snapshot=snapshot, method=method, step_size=args.step_size)
        for snapshot in snapshots
        for method in methods
    ]
    run_id = "phase0_gradient_snapshot_replay_{}".format(
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    payload = {
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "args": {
            "method_ids": args.method_ids,
            "step_size": args.step_size,
            "output_dir": str(args.output_dir),
            "report_dir": str(args.report_dir),
        },
        "snapshots": [str(path) for path in args.snapshots],
        "methods": [asdict(method) for method in methods],
        "summary": summarize(rows),
    }

    json_path = args.output_dir / f"{run_id}.json"
    csv_path = args.output_dir / f"{run_id}_steps.csv"
    report_path = args.report_dir / f"{run_id}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_report(report_path, payload)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "report": str(report_path),
                "summary": payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
