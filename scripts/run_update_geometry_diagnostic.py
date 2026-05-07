#!/usr/bin/env python3
"""Run a lightweight Mosaic update-geometry diagnostic.

This is a smoke-scale experiment for the project hypothesis: when several
Mosaic-style oracle losses disagree, naive weighted updates can locally harm
some oracles, and a simple geometry-aware correction should reduce that harm.

The first version uses sequence-level proxy oracles so the diagnostic can run
without downloading structure-model checkpoints. It still exercises the same
relaxed-sequence, LossTerm, gradient, simplex-projection, and update-metric
machinery needed for the structure-oracle experiment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.common import LossTerm, TOKENS
from mosaic.losses.trigram import TrigramLL
from mosaic.optimizers import projection_simplex


HYDROPHOBIC = frozenset("AILMFWVY")
POSITIVE = frozenset("KR")
NEGATIVE = frozenset("DE")


def git_metadata() -> dict[str, str | None]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, text=True).strip()
        except Exception:
            return None

    file_git = resolve_git_metadata_from_files(Path.cwd())
    return {
        "commit": run(["git", "rev-parse", "HEAD"]) or file_git["commit"],
        "short_commit": run(["git", "rev-parse", "--short", "HEAD"]) or file_git["short_commit"],
        "branch": run(["git", "branch", "--show-current"]) or file_git["branch"],
        "dirty": run(["git", "status", "--short"]),
    }


def resolve_git_metadata_from_files(repo: Path) -> dict[str, str | None]:
    git_path = repo / ".git"
    if git_path.is_file():
        content = git_path.read_text(encoding="utf-8").strip()
        if content.startswith("gitdir:"):
            git_dir = (repo / content.split(":", 1)[1].strip()).resolve()
        else:
            return {"commit": None, "short_commit": None, "branch": None}
    elif git_path.is_dir():
        git_dir = git_path
    else:
        return {"commit": None, "short_commit": None, "branch": None}

    common_dir = git_dir
    common_dir_file = git_dir / "commondir"
    if common_dir_file.exists():
        common_dir = (
            git_dir / common_dir_file.read_text(encoding="utf-8").strip()
        ).resolve()

    head_file = git_dir / "HEAD"
    if not head_file.exists():
        return {"commit": None, "short_commit": None, "branch": None}

    head = head_file.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return {"commit": head, "short_commit": head[:7], "branch": "detached"}

    ref = head.split(" ", 1)[1]
    branch = ref.removeprefix("refs/heads/")
    commit = None
    ref_file = common_dir / ref
    if ref_file.exists():
        commit = ref_file.read_text(encoding="utf-8").strip()
    else:
        packed_refs = common_dir / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2 and parts[1] == ref:
                    commit = parts[0]
                    break

    return {
        "commit": commit,
        "short_commit": commit[:7] if commit else None,
        "branch": branch,
    }


def token_indicator(chars: frozenset[str]) -> jax.Array:
    return jnp.asarray([1.0 if token in chars else 0.0 for token in TOKENS])


class HydrophobicContactProxy(LossTerm):
    weights: jax.Array

    def __call__(self, soft_sequence, *, key):
        hydrophobic_mean = jnp.mean(soft_sequence @ self.weights)
        return -hydrophobic_mean, {"hydrophobic_mean": hydrophobic_mean}


class SolubilityHydrophobicLimit(LossTerm):
    weights: jax.Array
    target: float = eqx.field(converter=float)
    sharpness: float = eqx.field(converter=float)

    def __call__(self, soft_sequence, *, key):
        hydrophobic_mean = jnp.mean(soft_sequence @ self.weights)
        excess = hydrophobic_mean - self.target
        penalty = jax.nn.softplus(self.sharpness * excess) / self.sharpness
        return penalty, {
            "hydrophobic_mean": hydrophobic_mean,
            "hydrophobic_excess": excess,
        }


class ChargeTarget(LossTerm):
    charges: jax.Array
    target_per_residue: float = eqx.field(converter=float)

    def __call__(self, soft_sequence, *, key):
        charge_per_residue = jnp.mean(soft_sequence @ self.charges)
        loss = (charge_per_residue - self.target_per_residue) ** 2
        return loss, {"charge_per_residue": charge_per_residue}


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    name: str
    aux_slack: float = 0.02
    max_aux_harms: int = 1
    min_primary_descent_ratio: float = 0.0
    cone_denominator: int = 8


METHODS = [
    MethodSpec("M1", "single_hydrophobic_proxy"),
    MethodSpec("M3", "naive_weighted"),
    MethodSpec("M4", "normalized_weighted"),
    MethodSpec("M6", "soft_cone_correction"),
]


def eval_loss_and_grad(loss: LossTerm, x: jax.Array, key: jax.Array):
    (value, aux), grad = eqx.filter_value_and_grad(loss, has_aux=True)(x, key=key)
    grad = jnp.nan_to_num(grad - grad.mean(axis=-1, keepdims=True))
    value = jnp.nan_to_num(value, nan=1e6)
    return value, aux, grad


def flatten_dot(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.sum(a * b))


def grad_norm(g: jax.Array) -> float:
    return float(jnp.sqrt(jnp.sum(g * g) + 1e-12))


def normalized(g: jax.Array) -> jax.Array:
    return g / (jnp.sqrt(jnp.sum(g * g)) + 1e-8)


def entropy(x: jax.Array) -> float:
    x = jnp.clip(x, 1e-8, 1.0)
    return float(-jnp.mean(jnp.sum(x * jnp.log(x), axis=-1)))


def projected_update(x: jax.Array, raw_direction: jax.Array, step_size: float) -> tuple[jax.Array, jax.Array]:
    x_new = jnp.asarray(projection_simplex(np.asarray(x + step_size * raw_direction)))
    return x_new, x_new - x


def choose_raw_direction(
    method: MethodSpec,
    grads: dict[str, jax.Array],
    weights: dict[str, float],
    *,
    x: jax.Array | None = None,
    step_size: float | None = None,
) -> jax.Array:
    ordered = list(grads)

    if method.name.startswith("single_"):
        return -grads[ordered[0]]

    if method.name == "naive_weighted":
        return -sum(weights[name] * grads[name] for name in ordered)

    normed_sum = sum(weights[name] * normalized(grads[name]) for name in ordered)
    raw = -normed_sum

    if method.name == "normalized_weighted":
        return raw

    if method.name == "soft_cone_correction":
        if x is None or step_size is None:
            raise ValueError("soft_cone_correction needs x and step_size")
        return choose_soft_cone_direction(
            grads=grads,
            weights=weights,
            x=x,
            step_size=step_size,
        )

    if method.name == "contact_preserving_soft_cone":
        if x is None or step_size is None:
            raise ValueError("contact_preserving_soft_cone needs x and step_size")
        return choose_contact_preserving_direction(
            grads=grads,
            weights=weights,
            x=x,
            step_size=step_size,
            aux_slack=method.aux_slack,
            max_aux_harms=method.max_aux_harms,
            min_primary_descent_ratio=method.min_primary_descent_ratio,
            cone_denominator=method.cone_denominator,
        )

    raise ValueError(method.name)


def simplex_weight_grid(n: int, denominator: int = 4):
    for counts in itertools.product(range(denominator + 1), repeat=n):
        if sum(counts) == denominator:
            yield np.asarray(counts, dtype=np.float32) / float(denominator)


def choose_soft_cone_direction(
    *,
    grads: dict[str, jax.Array],
    weights: dict[str, float],
    x: jax.Array,
    step_size: float,
) -> jax.Array:
    names = list(grads)
    normed_grads = {name: normalized(grads[name]) for name in names}
    candidates = []

    # Include normalized weighted and single-oracle directions.
    candidates.append(-sum(weights[name] * normed_grads[name] for name in names))
    candidates.extend([-normed_grads[name] for name in names])

    # Search the convex cone spanned by negative normalized gradients. The score
    # is computed on the actual projected simplex update, not on the raw vector.
    for alpha in simplex_weight_grid(len(names), denominator=4):
        if np.count_nonzero(alpha) == 0:
            continue
        raw = -sum(float(a) * normed_grads[name] for a, name in zip(alpha, names))
        candidates.append(raw)

    best_score = None
    best_raw = candidates[0]
    for raw in candidates:
        _, actual_update = projected_update(x, raw, step_size)
        derivatives = [flatten_dot(grads[name], actual_update) for name in names]
        harm_count = sum(value > 0 for value in derivatives)
        worst = max(derivatives)
        mean_derivative = float(np.mean(derivatives))
        step_norm_value = grad_norm(actual_update)
        # Primary objective: avoid harming oracles. Secondary: reduce the worst
        # harm. Tertiary: keep meaningful average descent and avoid zero steps.
        score = (
            harm_count,
            worst,
            mean_derivative,
            -step_norm_value,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_raw = raw
    return best_raw


def cone_candidates(
    *,
    grads: dict[str, jax.Array],
    weights: dict[str, float],
    denominator: int,
) -> list[jax.Array]:
    names = list(grads)
    normed_grads = {name: normalized(grads[name]) for name in names}
    candidates = [
        -sum(weights[name] * normed_grads[name] for name in names),
        -sum(weights[name] * grads[name] for name in names),
    ]
    candidates.extend([-normed_grads[name] for name in names])
    candidates.extend([-grads[name] for name in names])
    for alpha in simplex_weight_grid(len(names), denominator=denominator):
        if np.count_nonzero(alpha) == 0:
            continue
        raw = -sum(float(a) * normed_grads[name] for a, name in zip(alpha, names))
        candidates.append(raw)
    return candidates


def choose_contact_preserving_direction(
    *,
    grads: dict[str, jax.Array],
    weights: dict[str, float],
    x: jax.Array,
    step_size: float,
    aux_slack: float = 0.02,
    max_aux_harms: int = 1,
    min_primary_descent_ratio: float = 0.0,
    cone_denominator: int = 8,
) -> jax.Array:
    names = list(grads)
    primary = names[0]
    aux_names = names[1:]
    candidates = cone_candidates(grads=grads, weights=weights, denominator=cone_denominator)

    evaluated = []
    fallback = []
    for raw in candidates:
        _, actual_update = projected_update(x, raw, step_size)
        derivatives = {name: flatten_dot(grads[name], actual_update) for name in names}
        aux_derivatives = [derivatives[name] for name in aux_names]
        aux_harms = sum(value > 0 for value in aux_derivatives)
        aux_worst = max(aux_derivatives) if aux_derivatives else derivatives[primary]
        primary_derivative = derivatives[primary]
        step_norm_value = grad_norm(actual_update)
        total_harms = sum(value > 0 for value in derivatives.values())

        fallback.append(
            (
                total_harms,
                max(derivatives.values()),
                primary_derivative,
                -step_norm_value,
                raw,
            )
        )
        evaluated.append((primary_derivative, aux_harms, aux_worst, -step_norm_value, raw))

    best_primary_derivative = min(item[0] for item in evaluated)
    min_allowed_primary = min_primary_descent_ratio * best_primary_derivative
    feasible = [
        item
        for item in evaluated
        if item[1] <= max_aux_harms
        and item[2] <= aux_slack
        and item[0] < 0
        and item[0] <= min_allowed_primary
    ]
    if feasible:
        return min(feasible, key=lambda item: item[:-1])[-1]
    return min(fallback, key=lambda item: item[:-1])[-1]


def offdiag_cosines(grads: dict[str, jax.Array]) -> dict[str, float]:
    names = list(grads)
    out = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            denom = grad_norm(grads[left]) * grad_norm(grads[right]) + 1e-12
            out[f"cos_{left}__{right}"] = flatten_dot(grads[left], grads[right]) / denom
    return out


def run_single_method_with_terminal(
    *,
    method: MethodSpec,
    losses: dict[str, LossTerm],
    weights: dict[str, float],
    seed: int,
    binder_length: int,
    steps: int,
    step_size: float,
    init_temperature: float,
) -> tuple[list[dict[str, Any]], jax.Array]:
    key = jax.random.key(seed)
    x = jax.nn.softmax(
        init_temperature * jax.random.gumbel(key, shape=(binder_length, len(TOKENS))),
        axis=-1,
    )
    rows = []

    for step in range(steps):
        step_key = jax.random.fold_in(key, step)
        values = {}
        aux_values = {}
        grads = {}
        for idx, (name, loss) in enumerate(losses.items()):
            value, aux, grad = eval_loss_and_grad(loss, x, jax.random.fold_in(step_key, idx))
            values[name] = float(value)
            aux_values[name] = {
                aux_name: float(aux_value)
                for aux_name, aux_value in jax.tree_util.tree_leaves_with_path(aux)
                if hasattr(aux_value, "shape") and aux_value.shape == ()
            }
            grads[name] = grad

        raw_direction = choose_raw_direction(
            method,
            grads,
            weights,
            x=x,
            step_size=step_size,
        )
        x_new, actual_update = projected_update(x, raw_direction, step_size)

        directional = {
            f"dir_{name}": flatten_dot(grad, actual_update)
            for name, grad in grads.items()
        }
        harms = [value > 0 for value in directional.values()]
        cosines = offdiag_cosines(grads)
        row = {
            "method_id": method.method_id,
            "method": method.name,
            "method_aux_slack": method.aux_slack,
            "method_max_aux_harms": method.max_aux_harms,
            "method_min_primary_descent_ratio": method.min_primary_descent_ratio,
            "method_cone_denominator": method.cone_denominator,
            "seed": seed,
            "step": step,
            "oracle_harm_rate": float(np.mean(harms)),
            "worst_oracle_directional_derivative": max(directional.values()),
            "mean_oracle_descent": float(np.mean(list(directional.values()))),
            "step_norm": grad_norm(actual_update),
            "sequence_entropy": entropy(x),
            "sequence_entropy_delta": entropy(x_new) - entropy(x),
        }
        row.update({f"loss_{name}": value for name, value in values.items()})
        row.update(directional)
        row.update(cosines)
        rows.append(row)
        x = x_new

    return rows, x


def run_single_method(
    *,
    method: MethodSpec,
    losses: dict[str, LossTerm],
    weights: dict[str, float],
    seed: int,
    binder_length: int,
    steps: int,
    step_size: float,
    init_temperature: float,
) -> list[dict[str, Any]]:
    rows, _terminal = run_single_method_with_terminal(
        method=method,
        losses=losses,
        weights=weights,
        seed=seed,
        binder_length=binder_length,
        steps=steps,
        step_size=step_size,
        init_temperature=init_temperature,
    )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summaries = []
    for method, method_rows in grouped.items():
        final_rows = [row for row in method_rows if row["step"] == max(r["step"] for r in method_rows if r["seed"] == row["seed"])]
        summaries.append(
            {
                "method": method,
                "num_rows": len(method_rows),
                "mean_oracle_harm_rate": float(np.mean([r["oracle_harm_rate"] for r in method_rows])),
                "mean_worst_oracle_directional_derivative": float(
                    np.mean([r["worst_oracle_directional_derivative"] for r in method_rows])
                ),
                "mean_step_norm": float(np.mean([r["step_norm"] for r in method_rows])),
                "mean_final_entropy": float(np.mean([r["sequence_entropy"] for r in final_rows])),
                "mean_final_hydrophobic_loss": float(np.mean([r["loss_hydrophobic_contact"] for r in final_rows])),
                "mean_final_solubility_loss": float(np.mean([r["loss_solubility_limit"] for r in final_rows])),
                "mean_final_charge_loss": float(np.mean([r["loss_charge_target"] for r in final_rows])),
                "mean_final_trigram_loss": float(np.mean([r["loss_trigram_naturalness"] for r in final_rows])),
            }
        )
    return sorted(summaries, key=lambda row: row["mean_oracle_harm_rate"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 0 Update-Geometry Diagnostic",
        "",
        f"Run ID: `{payload['run_id']}`",
        "",
        "## Scope",
        "",
        "This is a lightweight Mosaic-internal smoke experiment using sequence-level proxy oracles. It validates the update-geometry logging and gives an initial read on whether geometry-aware updates reduce oracle harm versus scalarized updates.",
        "",
        "## Summary",
        "",
        "| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final hydrophobic loss | Final solubility loss | Final charge loss | Final trigram loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {method} | {mean_oracle_harm_rate:.3f} | {mean_worst_oracle_directional_derivative:.4f} | {mean_step_norm:.4f} | {mean_final_entropy:.3f} | {mean_final_hydrophobic_loss:.4f} | {mean_final_solubility_loss:.4f} | {mean_final_charge_loss:.4f} | {mean_final_trigram_loss:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `oracle_harm_rate` is the fraction of oracle gradients for which the actual projected update has positive directional derivative.",
            "- Lower harm rate and lower worst-oracle directional derivative indicate safer multi-oracle updates.",
            "- This run does not yet include structure oracles; it is a fast mechanism check before spending structure-model compute.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binder-length", type=int, default=80)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=0.45)
    parser.add_argument("--init-temperature", type=float, default=0.5)
    parser.add_argument("--trigram-path", type=Path, default=Path("trigram_seg.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/reports"))
    args = parser.parse_args()

    hydrophobic_weights = token_indicator(HYDROPHOBIC)
    charge_weights = token_indicator(POSITIVE) - token_indicator(NEGATIVE)
    losses: dict[str, LossTerm] = {
        "hydrophobic_contact": HydrophobicContactProxy(hydrophobic_weights),
        "solubility_limit": SolubilityHydrophobicLimit(
            hydrophobic_weights,
            target=0.30,
            sharpness=24.0,
        ),
        "charge_target": ChargeTarget(charge_weights, target_per_residue=0.06),
        "trigram_naturalness": TrigramLL.from_pkl(args.trigram_path),
    }
    weights = {name: 1.0 for name in losses}

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for method in METHODS:
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
                )
            )

    run_id = "phase0_update_geometry_{}_{}".format(
        git_metadata()["short_commit"] or "nogit",
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    summary = summarize(rows)
    payload = {
        "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_metadata(),
        "args": vars(args) | {"trigram_path": str(args.trigram_path), "output_dir": str(args.output_dir), "report_dir": str(args.report_dir)},
        "oracles": list(losses),
        "methods": [asdict(method) for method in METHODS],
        "summary": summary,
    }

    json_path = args.output_dir / f"{run_id}.json"
    csv_path = args.output_dir / f"{run_id}_steps.csv"
    report_path = args.report_dir / f"{run_id}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_summary_markdown(report_path, payload)

    print(json.dumps({"run_id": run_id, "json": str(json_path), "csv": str(csv_path), "report": str(report_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
