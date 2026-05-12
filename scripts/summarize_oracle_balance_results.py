#!/usr/bin/env python3
"""Aggregate Phase 0 oracle-balance runs and apply the update-rule gate."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = {
    "oracle_harm_rate": "mean_oracle_harm_rate",
    "worst_oracle_directional_derivative": "mean_worst_oracle_directional_derivative",
    "dir_protenix_contact": "mean_protenix_directional_derivative",
    "dir_boltz2_distogram_contact": "mean_boltz2_directional_derivative",
    "step_norm": "mean_step_norm",
}


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_csv_path(json_path: Path, payload: dict[str, Any]) -> Path:
    csv_path = payload.get("csv_path")
    if csv_path:
        candidate = Path(csv_path)
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            candidate = json_path.parent / candidate
            if candidate.exists():
                return candidate
    candidate = json_path.with_name(f"{json_path.stem}_steps.csv")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find step CSV for {json_path}")


def target_label_from_payload(json_path: Path, payload: dict[str, Any]) -> str:
    args = payload.get("args", {})
    structure = str(args.get("target_structure", json_path.stem))
    chain = str(args.get("target_chain", ""))
    stem = Path(structure).stem
    return f"{stem}:{chain}" if chain else stem


def load_run(path: Path) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    if path.suffix == ".json":
        payload = load_json(path)
        csv_path = infer_csv_path(path, payload)
        rows = read_rows(csv_path)
        return target_label_from_payload(path, payload), payload, rows
    if path.suffix == ".csv":
        rows = read_rows(path)
        payload = {"run_id": path.stem.removesuffix("_steps"), "args": {}}
        return path.stem.removesuffix("_steps"), payload, rows
    raise ValueError(f"Unsupported input file: {path}")


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_rows": len(rows),
        "num_seeds": len({row.get("seed") for row in rows if row.get("seed") not in {None, ""}}),
    }
    for source, output in METRICS.items():
        values = [value for row in rows if (value := parse_float(row.get(source))) is not None]
        result[output] = mean(values)
    harm_values = [
        value for row in rows if (value := parse_float(row.get("oracle_harm_rate"))) is not None
    ]
    result["zero_harm"] = bool(harm_values) and all(value <= 0.0 for value in harm_values)
    return result


def aggregate_runs(
    runs: list[tuple[str, dict[str, Any], list[dict[str, Any]]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_target_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    method_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_names: dict[str, str] = {}

    for target, _payload, rows in runs:
        for row in rows:
            method_id = str(row.get("method_id", row.get("method", "unknown")))
            method_names.setdefault(method_id, str(row.get("method", method_id)))
            per_target_groups[(target, method_id)].append(row)
            method_groups[method_id].append(row | {"target_label": target})

    per_target = []
    for (target, method_id), rows in sorted(per_target_groups.items()):
        per_target.append(
            {
                "target": target,
                "method_id": method_id,
                "method": method_names.get(method_id, method_id),
                **aggregate_group(rows),
            }
        )

    overall = []
    for method_id, rows in sorted(method_groups.items()):
        target_zero_harm = {
            row["target"]
            for row in per_target
            if row["method_id"] == method_id and row["zero_harm"]
        }
        target_count = len({row.get("target_label") for row in rows})
        overall.append(
            {
                "method_id": method_id,
                "method": method_names.get(method_id, method_id),
                "num_targets": target_count,
                "zero_harm_targets": len(target_zero_harm),
                **aggregate_group(rows),
            }
        )
    overall.sort(
        key=lambda row: (
            row["mean_oracle_harm_rate"],
            row["mean_worst_oracle_directional_derivative"],
        )
    )
    return per_target, overall


def gate_methods(
    overall: list[dict[str, Any]],
    *,
    baseline_method_id: str,
    harm_tolerance: float,
    worst_tolerance: float,
    min_step_norm_ratio: float,
) -> list[dict[str, Any]]:
    by_method = {row["method_id"]: row for row in overall}
    baseline = by_method.get(baseline_method_id)
    if baseline is None:
        raise ValueError(f"Baseline method `{baseline_method_id}` not found")

    gated = []
    for row in overall:
        if row["method_id"] == baseline_method_id:
            decision = "baseline"
            checks = {
                "harm_no_worse": True,
                "worst_no_worse": True,
                "protenix_descent": row["mean_protenix_directional_derivative"] <= 0.0,
                "boltz2_descent": row["mean_boltz2_directional_derivative"] <= 0.0,
                "step_nontrivial": True,
            }
        else:
            checks = {
                "harm_no_worse": row["mean_oracle_harm_rate"]
                <= baseline["mean_oracle_harm_rate"] + harm_tolerance,
                "worst_no_worse": row["mean_worst_oracle_directional_derivative"]
                <= baseline["mean_worst_oracle_directional_derivative"] + worst_tolerance,
                "protenix_descent": row["mean_protenix_directional_derivative"] <= 0.0,
                "boltz2_descent": row["mean_boltz2_directional_derivative"] <= 0.0,
                "step_nontrivial": row["mean_step_norm"]
                >= baseline["mean_step_norm"] * min_step_norm_ratio,
            }
            decision = "pass" if all(checks.values()) else "fail"
        gated.append(row | {"gate_decision": decision, "gate_checks": checks})
    return gated


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row if key != "gate_checks"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 0 Oracle Balance Gate Summary",
        "",
        f"Created: `{payload['created_at_utc']}`",
        "",
        "## Gate",
        "",
        f"- Baseline: `{payload['gate']['baseline_method_id']}`",
        f"- Harm tolerance: `{payload['gate']['harm_tolerance']}`",
        f"- Worst-derivative tolerance: `{payload['gate']['worst_tolerance']}`",
        f"- Minimum step-norm ratio: `{payload['gate']['min_step_norm_ratio']}`",
        "",
        "## Overall",
        "",
        "| Method | Gate | Targets | Zero-harm targets | Harm | Worst dir | Protenix dir | Boltz2 dir | Step norm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["overall"]:
        lines.append(
            "| {method_id} / {method} | {gate_decision} | {num_targets} | "
            "{zero_harm_targets} | {mean_oracle_harm_rate:.3f} | "
            "{mean_worst_oracle_directional_derivative:.4f} | "
            "{mean_protenix_directional_derivative:.4f} | "
            "{mean_boltz2_directional_derivative:.4f} | "
            "{mean_step_norm:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Per-Target",
            "",
            "| Target | Method | Zero harm | Harm | Worst dir | Protenix dir | Boltz2 dir | Step norm |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["per_target"]:
        lines.append(
            "| {target} | {method_id} / {method} | {zero_harm} | "
            "{mean_oracle_harm_rate:.3f} | "
            "{mean_worst_oracle_directional_derivative:.4f} | "
            "{mean_protenix_directional_derivative:.4f} | "
            "{mean_boltz2_directional_derivative:.4f} | "
            "{mean_step_norm:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `pass` means the method matches the configured update-level gate "
            "against the baseline.",
            "- This gate is intentionally update-level only; candidate-level "
            "Boltz2 holdout remains a separate decision.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="Per-target JSON or *_steps.csv files")
    parser.add_argument("--baseline-method-id", default="M4")
    parser.add_argument("--harm-tolerance", type=float, default=0.0)
    parser.add_argument("--worst-tolerance", type=float, default=0.0)
    parser.add_argument("--min-step-norm-ratio", type=float, default=0.50)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("docs/reports/phase0_oracle_balance_gate_summary"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [load_run(path) for path in args.inputs]
    per_target, overall = aggregate_runs(runs)
    gated_overall = gate_methods(
        overall,
        baseline_method_id=args.baseline_method_id,
        harm_tolerance=args.harm_tolerance,
        worst_tolerance=args.worst_tolerance,
        min_step_norm_ratio=args.min_step_norm_ratio,
    )
    payload = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": [str(path) for path in args.inputs],
        "gate": {
            "baseline_method_id": args.baseline_method_id,
            "harm_tolerance": args.harm_tolerance,
            "worst_tolerance": args.worst_tolerance,
            "min_step_norm_ratio": args.min_step_norm_ratio,
        },
        "per_target": per_target,
        "overall": gated_overall,
    }

    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    write_json(json_path, payload)
    write_csv(csv_path, gated_overall)
    write_markdown(md_path, payload)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
