#!/usr/bin/env python3
"""Summarize top-k handoff sensitivity from a candidate CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = {
    "bt_pae": ("candidate_bt_pae", "min"),
    "bt_iptm": ("candidate_bt_iptm", "max"),
    "ipsae_min": ("candidate_ipsae_min", "max"),
    "contact": ("candidate_loss_protenix_contact", "min"),
}

SUMMARY_COLUMNS = [
    "candidate_plddt",
    "candidate_bt_pae",
    "candidate_bt_iptm",
    "candidate_ipsae_min",
    "candidate_loss_protenix_contact",
    "candidate_loss_trigram_naturalness",
    "candidate_solubility_limit_hydrophobic_mean",
    "candidate_charge_target_charge_per_residue",
]


def parse_float(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            for key, value in row.items():
                if key.startswith("candidate_") or key.startswith("method_") or key == "seed":
                    try:
                        parsed[key] = parse_float(value)
                    except ValueError:
                        parsed[key] = value
            parsed["seed"] = int(float(parsed["seed"]))
            parsed["candidate_sample_index"] = int(float(parsed.get("candidate_sample_index", 0)))
            rows.append(parsed)
    return rows


def better(left: dict[str, Any], right: dict[str, Any], metric: str) -> bool:
    column, direction = METRICS[metric]
    left_value = float(left[column])
    right_value = float(right[column])
    if direction == "min":
        return left_value < right_value
    return left_value > right_value


def select_best(
    rows: list[dict[str, Any]],
    *,
    budget: int,
    metric: str,
    score_mode: str,
) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row["score_mode"] != score_mode:
            continue
        if score_mode == "topk_sample" and row["candidate_sample_index"] >= budget:
            continue
        key = (row["method_id"], row["method"], row["seed"])
        if key not in best or better(row, best[key], metric):
            best[key] = row
    return list(best.values())


def summarize_rows(rows: list[dict[str, Any]], *, budget: int, metric: str, score_mode: str) -> list[dict[str, Any]]:
    selected = select_best(rows, budget=budget, metric=metric, score_mode=score_mode)
    by_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_method[(row["method_id"], row["method"])].append(row)

    summaries = []
    for (method_id, method), method_rows in sorted(by_method.items()):
        summary: dict[str, Any] = {
            "selection_metric": metric,
            "budget": budget if score_mode == "topk_sample" else 1,
            "score_mode": score_mode,
            "method_id": method_id,
            "method": method,
            "num_selected": len(method_rows),
        }
        for column in SUMMARY_COLUMNS:
            values = [float(row[column]) for row in method_rows if column in row and row[column] != ""]
            finite_values = [value for value in values if not math.isnan(value)]
            summary[f"mean_{column}"] = sum(finite_values) / len(finite_values) if finite_values else float("nan")
        summaries.append(summary)
    return sorted(summaries, key=lambda row: row["mean_candidate_bt_pae"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 0 ACT-012 Top-K Handoff Sensitivity",
        "",
        f"Source candidates: `{payload['source_candidates']}`",
        "",
        "## Scope",
        "",
        "This analysis reuses one candidate pool and recomputes best-per-seed top-k selection under different sample budgets and reranking metrics. It tests whether the ACT-011 M7c advantage is robust to the decoding budget and selection rule.",
        "",
        "## Controls",
        "",
        "| Metric | Score mode | Method | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["control_summary"]:
        lines.append(
            "| {selection_metric} | {score_mode} | {method_id} | {plddt} | {bt_pae} | {bt_iptm} | {ipsae} | {contact} | {trigram} |".format(
                selection_metric=row["selection_metric"],
                score_mode=row["score_mode"],
                method_id=row["method_id"],
                plddt=format_float(row["mean_candidate_plddt"]),
                bt_pae=format_float(row["mean_candidate_bt_pae"]),
                bt_iptm=format_float(row["mean_candidate_bt_iptm"]),
                ipsae=format_float(row["mean_candidate_ipsae_min"]),
                contact=format_float(row["mean_candidate_loss_protenix_contact"]),
                trigram=format_float(row["mean_candidate_loss_trigram_naturalness"]),
            )
        )

    for metric in payload["metrics"]:
        lines.extend(
            [
                "",
                f"## Top-K Selection: `{metric}`",
                "",
                "| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        rows = [
            row
            for row in payload["topk_summary"]
            if row["selection_metric"] == metric
        ]
        for row in rows:
            lines.append(
                "| {budget} | {method_id} | {num_selected} | {plddt} | {bt_pae} | {bt_iptm} | {ipsae} | {contact} | {trigram} |".format(
                    budget=row["budget"],
                    method_id=row["method_id"],
                    num_selected=row["num_selected"],
                    plddt=format_float(row["mean_candidate_plddt"]),
                    bt_pae=format_float(row["mean_candidate_bt_pae"]),
                    bt_iptm=format_float(row["mean_candidate_bt_iptm"]),
                    ipsae=format_float(row["mean_candidate_ipsae_min"]),
                    contact=format_float(row["mean_candidate_loss_protenix_contact"]),
                    trigram=format_float(row["mean_candidate_loss_trigram_naturalness"]),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- A robust positive signal means M7c stays near the top across budgets and rerank metrics under matched candidate scoring budget.",
            "- If gains appear only at high budget, report the quality/cost tradeoff rather than claiming a free decoding improvement.",
            "- If metric choice changes the winner, the next experiment should pre-register the paper-facing selection metric before scale-up.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--budgets", type=str, default="1,4,8")
    parser.add_argument("--metrics", type=str, default="bt_pae,bt_iptm,contact")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    budgets = [int(item.strip()) for item in args.budgets.split(",") if item.strip()]
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    for metric in metrics:
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")

    rows = read_rows(args.candidates_csv)
    topk_summary = []
    control_summary = []
    for metric in metrics:
        for score_mode in ("soft", "argmax"):
            control_summary.extend(summarize_rows(rows, budget=1, metric=metric, score_mode=score_mode))
        for budget in budgets:
            topk_summary.extend(summarize_rows(rows, budget=budget, metric=metric, score_mode="topk_sample"))

    payload = {
        "source_candidates": str(args.candidates_csv),
        "budgets": budgets,
        "metrics": metrics,
        "control_summary": control_summary,
        "topk_summary": topk_summary,
    }

    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    report_path = args.output_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, control_summary + topk_summary)
    write_report(report_path, payload)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
