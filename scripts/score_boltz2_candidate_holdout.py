#!/usr/bin/env python3
"""Score Phase 0 binder candidates with finite Boltz2 second-oracle metrics."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

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
    (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER "
        "THR TRP TYR VAL MSE SEC PYL ASX GLX UNK"
    ).split()
)


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
        if not content.startswith("gitdir:"):
            return {"commit": None, "short_commit": None, "branch": None}
        git_dir = (repo / content.split(":", 1)[1].strip()).resolve()
    elif git_path.is_dir():
        git_dir = git_path
    else:
        return {"commit": None, "short_commit": None, "branch": None}

    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return {"commit": None, "short_commit": None, "branch": None}

    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        commit_path = git_dir / ref
        commit = commit_path.read_text(encoding="utf-8").strip() if commit_path.exists() else None
        branch = ref.removeprefix("refs/heads/")
    else:
        commit = head
        branch = None

    return {
        "commit": commit,
        "short_commit": commit[:7] if commit else None,
        "branch": branch,
    }


def parse_csv_list(raw: str | None) -> set[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


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
    if target_length is not None and len(chain) < target_length:
        raise ValueError(
            f"{path}:{chain_id} has only {len(chain)} residues, "
            f"fewer than target_length={target_length}"
        )
    return chain


def make_target_chain(path: Path, chain_id: str, target_length: int | None) -> TargetChain:
    chain = clean_chain(path, chain_id, target_length)
    sequence = gemmi.one_letter_code([residue.name for residue in chain])
    return TargetChain(sequence=sequence, use_msa=False, template_chain=chain)


def build_features(chains: list[TargetChain], cache: Path):
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


def sequence_to_one_hot(sequence: str) -> jax.Array:
    invalid = sorted(set(sequence) - set(TOKENS))
    if invalid:
        raise ValueError(f"Sequence contains unsupported residues {invalid}: {sequence}")
    indices = np.array([TOKENS.index(residue) for residue in sequence], dtype=np.int32)
    return jax.nn.one_hot(indices, len(TOKENS))


def read_candidates(
    path: Path,
    *,
    max_candidates: int,
    method_ids: set[str] | None,
    score_modes: set[str] | None,
    deduplicate_sequences: bool,
    deduplicate_scope: str,
    balance_by: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    filtered: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        sequence = (row.get("sequence") or "").strip().upper()
        if not sequence:
            continue
        if method_ids is not None and row.get("method_id") not in method_ids:
            continue
        if score_modes is not None and row.get("score_mode") not in score_modes:
            continue
        row = dict(row)
        row["source_row_index"] = row_index
        row["sequence"] = sequence
        row["selection_group"] = format_balance_key(balance_key(row, balance_by))
        filtered.append(row)

    candidates = deduplicate_candidate_rows(
        filtered,
        deduplicate_sequences=deduplicate_sequences,
        deduplicate_scope=deduplicate_scope,
        balance_by=balance_by,
    )

    selected = select_candidates(candidates, max_candidates=max_candidates, balance_by=balance_by)
    selection_summary = summarize_selection(
        raw_rows=rows,
        filtered_rows=filtered,
        deduplicated_rows=candidates,
        selected_rows=selected,
        max_candidates=max_candidates,
        balance_by=balance_by,
        deduplicate_sequences=deduplicate_sequences,
        deduplicate_scope=deduplicate_scope,
    )

    if not selected:
        raise ValueError(f"No candidates selected from {path}")

    lengths = {len(row["sequence"]) for row in selected}
    if len(lengths) != 1:
        raise ValueError(f"Selected candidates have mixed lengths: {sorted(lengths)}")
    return selected, selection_summary


def balance_key(row: dict[str, Any], balance_by: str) -> tuple[str, ...]:
    if balance_by == "none":
        return ("all",)
    if balance_by == "method_id":
        return (str(row.get("method_id") or ""),)
    if balance_by == "method_score_mode":
        return (str(row.get("method_id") or ""), str(row.get("score_mode") or ""))
    raise ValueError(f"Unsupported balance_by={balance_by}")


def format_balance_key(key: tuple[str, ...]) -> str:
    return "::".join(key)


def deduplicate_candidate_rows(
    candidates: list[dict[str, Any]],
    *,
    deduplicate_sequences: bool,
    deduplicate_scope: str,
    balance_by: str,
) -> list[dict[str, Any]]:
    if not deduplicate_sequences:
        return list(candidates)

    if deduplicate_scope == "global":
        deduplicated: list[dict[str, Any]] = []
        seen_sequences: set[str] = set()
        for row in candidates:
            sequence = row["sequence"]
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            deduplicated.append(row)
        return deduplicated

    if deduplicate_scope == "group":
        deduplicated = []
        seen_by_group: dict[tuple[str, ...], set[str]] = {}
        for row in candidates:
            key = balance_key(row, balance_by)
            sequence = row["sequence"]
            seen_sequences = seen_by_group.setdefault(key, set())
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            deduplicated.append(row)
        return deduplicated

    raise ValueError(f"Unsupported deduplicate_scope={deduplicate_scope}")


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
    balance_by: str,
) -> list[dict[str, Any]]:
    if balance_by == "none":
        return candidates[:max_candidates]

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    group_order: list[tuple[str, ...]] = []
    for row in candidates:
        key = balance_key(row, balance_by)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(row)

    selected: list[dict[str, Any]] = []
    while len(selected) < max_candidates:
        made_progress = False
        for key in group_order:
            if groups[key]:
                selected.append(groups[key].pop(0))
                made_progress = True
                if len(selected) >= max_candidates:
                    break
        if not made_progress:
            break
    return selected


def count_by_group(rows: list[dict[str, Any]], balance_by: str) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = balance_key(row, balance_by)
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_selection(
    *,
    raw_rows: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    deduplicated_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    max_candidates: int,
    balance_by: str,
    deduplicate_sequences: bool,
    deduplicate_scope: str,
) -> dict[str, Any]:
    filtered_counts = count_by_group(filtered_rows, balance_by)
    deduplicated_counts = count_by_group(deduplicated_rows, balance_by)
    selected_counts = count_by_group(selected_rows, balance_by)
    group_keys = sorted(set(filtered_counts) | set(deduplicated_counts) | set(selected_counts))
    return {
        "input_rows": len(raw_rows),
        "filtered_rows": len(filtered_rows),
        "deduplicated_rows": len(deduplicated_rows),
        "selected_rows": len(selected_rows),
        "max_candidates": max_candidates,
        "balance_by": balance_by,
        "deduplicate_sequences": deduplicate_sequences,
        "deduplicate_scope": deduplicate_scope if deduplicate_sequences else "none",
        "deduplicated_removed_rows": len(filtered_rows) - len(deduplicated_rows),
        "unselected_rows": len(deduplicated_rows) - len(selected_rows),
        "groups": [
            {
                "key": list(key),
                "group": format_balance_key(key),
                "available_rows": filtered_counts.get(key, 0),
                "deduplicated_rows": deduplicated_counts.get(key, 0),
                "selected_rows": selected_counts.get(key, 0),
            }
            for key in group_keys
        ],
    }


def finite_metrics(output, binder_len: int) -> dict[str, float | bool | list[int]]:
    inter_pae = output.pae[:binder_len, binder_len:]
    inter_dist_logits = output.distogram_logits[:binder_len, binder_len:]
    dist_probs = jax.nn.softmax(inter_dist_logits, axis=-1)
    contact_prob_8a = dist_probs[..., output.distogram_bins < 8.0].sum(-1)
    contact_prob_12a = dist_probs[..., output.distogram_bins < 12.0].sum(-1)

    return {
        "boltz2_distogram_shape": list(output.distogram_logits.shape),
        "boltz2_plddt_mean": float(jnp.mean(output.plddt)),
        "boltz2_plddt_binder_mean": float(jnp.mean(output.plddt[:binder_len])),
        "boltz2_plddt_target_mean": float(jnp.mean(output.plddt[binder_len:])),
        "boltz2_pae_mean": float(jnp.mean(output.pae)),
        "boltz2_inter_pae_mean": float(jnp.mean(inter_pae)),
        "boltz2_inter_pae_min": float(jnp.min(inter_pae)),
        "boltz2_inter_contact_prob_8a_mean": float(jnp.mean(contact_prob_8a)),
        "boltz2_inter_contact_prob_12a_mean": float(jnp.mean(contact_prob_12a)),
        "boltz2_distogram_finite": bool(jnp.isfinite(output.distogram_logits).all()),
        "boltz2_plddt_finite": bool(jnp.isfinite(output.plddt).all()),
        "boltz2_pae_finite": bool(jnp.isfinite(output.pae).all()),
        "boltz2_structure_finite": bool(jnp.isfinite(output.structure_coordinates).all()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "boltz2_rank",
        "source_row_index",
        "method_id",
        "method",
        "score_mode",
        "seed",
        "sequence",
        "candidate_bt_pae",
        "candidate_target_contact",
        "candidate_plddt",
        "boltz2_inter_pae_mean",
        "boltz2_inter_pae_min",
        "boltz2_inter_contact_prob_12a_mean",
        "boltz2_plddt_mean",
        "boltz2_plddt_binder_mean",
        "boltz2_finite_scoring_ok",
        "boltz2_structure_finite",
        "boltz2_elapsed_sec",
    ]
    all_keys = sorted({key for row in rows for key in row})
    fieldnames = preferred + [key for key in all_keys if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def pearson(x_values: list[float | None], y_values: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x = np.array([pair[0] for pair in pairs], dtype=np.float64)
    y = np.array([pair[1] for pair in pairs], dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_counts: dict[str, int] = {}
    method_score_mode_counts: dict[str, int] = {}
    for row in rows:
        method_id = str(row.get("method_id") or "")
        score_mode = str(row.get("score_mode") or "")
        method_counts[method_id] = method_counts.get(method_id, 0) + 1
        key = f"{method_id}::{score_mode}"
        method_score_mode_counts[key] = method_score_mode_counts.get(key, 0) + 1
    return {
        "num_candidates": len(rows),
        "all_finite_scoring_ok": all(bool(row["boltz2_finite_scoring_ok"]) for row in rows),
        "any_structure_finite": any(bool(row["boltz2_structure_finite"]) for row in rows),
        "mean_boltz2_inter_pae": float(np.mean([row["boltz2_inter_pae_mean"] for row in rows])),
        "best_boltz2_sequence": min(rows, key=lambda row: row["boltz2_inter_pae_mean"])["sequence"],
        "selected_by_method_id": method_counts,
        "selected_by_method_score_mode": method_score_mode_counts,
        "pearson_candidate_bt_pae_vs_boltz2_inter_pae": pearson(
            [safe_float(row.get("candidate_bt_pae")) for row in rows],
            [safe_float(row.get("boltz2_inter_pae_mean")) for row in rows],
        ),
        "pearson_candidate_contact_vs_boltz2_contact12": pearson(
            [safe_float(row.get("candidate_target_contact")) for row in rows],
            [safe_float(row.get("boltz2_inter_contact_prob_12a_mean")) for row in rows],
        ),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    selection = payload["candidate_selection"]
    lines = [
        "# Boltz2 Candidate Holdout Scoring",
        "",
        f"Run ID: `{payload['run_id']}`",
        "",
        "## Scope",
        "",
        "This post-hoc check scores Phase 0 Protenix candidates with Boltz2 "
        "finite outputs. It intentionally uses distogram, pLDDT, and PAE "
        "metrics only; coordinate-level structure output is recorded but not "
        "treated as a pass condition.",
        "",
        "## Summary",
        "",
        f"- Candidates scored: {summary['num_candidates']}",
        f"- Finite distogram/pLDDT/PAE for all candidates: {summary['all_finite_scoring_ok']}",
        f"- Any finite structure coordinates: {summary['any_structure_finite']}",
        f"- Mean Boltz2 inter-PAE: {summary['mean_boltz2_inter_pae']:.4f}",
        f"- Best Boltz2 sequence: `{summary['best_boltz2_sequence']}`",
        "- Pearson Protenix bt_PAE vs Boltz2 inter-PAE: "
        f"{summary['pearson_candidate_bt_pae_vs_boltz2_inter_pae']}",
        "- Pearson Protenix contact vs Boltz2 contact@12A: "
        f"{summary['pearson_candidate_contact_vs_boltz2_contact12']}",
        "",
        "## Candidate Selection",
        "",
        f"- Balance key: `{selection['balance_by']}`",
        f"- Deduplicate sequences: {selection['deduplicate_sequences']}",
        f"- Deduplicate scope: `{selection['deduplicate_scope']}`",
        f"- Input rows: {selection['input_rows']}",
        f"- Filtered rows: {selection['filtered_rows']}",
        f"- Deduplicated rows: {selection['deduplicated_rows']}",
        f"- Selected rows: {selection['selected_rows']}",
        f"- Rows removed by deduplication: {selection['deduplicated_removed_rows']}",
        f"- Rows left unselected after balancing: {selection['unselected_rows']}",
        "",
        "| Group | Available | After Dedup | Selected |",
        "|---|---:|---:|---:|",
    ]
    for group in selection["groups"]:
        lines.append(
            "| `{group}` | {available_rows} | {deduplicated_rows} | {selected_rows} |".format(
                **group
            )
        )
    lines += [
        "",
        "## Top Boltz2 Candidates",
        "",
        "| Rank | Method | Mode | Sequence | Protenix bt PAE | "
        "Boltz2 inter PAE | Boltz2 contact@12A | Boltz2 pLDDT |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"][:10]:
        lines.append(
            (
                "| {boltz2_rank} | {method_id} | {score_mode} | `{sequence}` | "
                "{candidate_bt_pae} | {boltz2_inter_pae_mean:.4f} | "
                "{boltz2_inter_contact_prob_12a_mean:.4f} | "
                "{boltz2_plddt_mean:.4f} |"
            ).format(
                **row
            )
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- CSV: `{payload['csv_path']}`",
        f"- JSON: `{payload['json_path']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a Phase 0 candidate CSV with finite Boltz2 second-oracle metrics."
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--target-structure", type=Path, default=Path("IL7RA.cif"))
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--target-length", type=int, default=48)
    parser.add_argument(
        "--boltz-cache",
        type=Path,
        default=Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser(),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument(
        "--method-ids",
        default=None,
        help="Comma-separated method_id allowlist, e.g. M3,M7c,M8a.",
    )
    parser.add_argument(
        "--score-modes",
        default=None,
        help="Comma-separated score_mode allowlist, e.g. argmax,topk_sample.",
    )
    parser.add_argument(
        "--balance-by",
        choices=("none", "method_id", "method_score_mode"),
        default="none",
        help="Round-robin selected candidates across this key after filtering.",
    )
    parser.add_argument("--deduplicate-sequences", action="store_true")
    parser.add_argument(
        "--deduplicate-scope",
        choices=("global", "group"),
        default="global",
        help=(
            "Scope for --deduplicate-sequences. The default preserves legacy "
            "global sequence deduplication; group deduplicates within the "
            "selected --balance-by group before round-robin selection."
        ),
    )
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/reports"))
    parser.add_argument("--run-label", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_label = args.run_label or args.candidates_csv.stem
    run_id = f"boltz2_candidate_holdout_{run_label}_{run_stamp}"
    checkpoint = args.checkpoint or (args.boltz_cache / "boltz2_conf.ckpt")

    print("run_id", run_id, flush=True)
    print("node", os.uname().nodename, flush=True)
    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    print("torch", torch.__version__, "torch_cuda_available", torch.cuda.is_available(), flush=True)
    print("checkpoint", checkpoint, "exists", checkpoint.exists(), flush=True)

    source_rows, selection_summary = read_candidates(
        args.candidates_csv,
        max_candidates=args.max_candidates,
        method_ids=parse_csv_list(args.method_ids),
        score_modes=parse_csv_list(args.score_modes),
        deduplicate_sequences=args.deduplicate_sequences,
        deduplicate_scope=args.deduplicate_scope,
        balance_by=args.balance_by,
    )
    binder_len = len(source_rows[0]["sequence"])
    target = make_target_chain(args.target_structure, args.target_chain, args.target_length)
    binder = TargetChain(sequence="G" * binder_len, use_msa=False)
    print(
        "selected_candidates",
        len(source_rows),
        "binder_len",
        binder_len,
        "target_len",
        len(target.sequence),
        "total_len",
        binder_len + len(target.sequence),
        flush=True,
    )

    feature_start = time.time()
    base_features, _ = build_features([binder, target], args.boltz_cache)
    jax.block_until_ready(base_features)
    print(f"features_ok elapsed_sec={time.time() - feature_start:.2f}", flush=True)
    print(
        "feature_token_index_shape",
        tuple(np.asarray(base_features["token_index"]).shape),
        flush=True,
    )

    load_start = time.time()
    model = load_boltz2(checkpoint)
    jax.block_until_ready(model)
    print(f"load_ok elapsed_sec={time.time() - load_start:.2f}", flush=True)

    scored_rows: list[dict[str, Any]] = []
    for candidate_idx, source_row in enumerate(source_rows):
        candidate_start = time.time()
        sequence = source_row["sequence"]
        features = set_binder_sequence(sequence_to_one_hot(sequence), base_features)
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
            **source_row,
            **finite_metrics(output, binder_len),
            "boltz2_elapsed_sec": round(time.time() - candidate_start, 2),
        }
        row["boltz2_finite_scoring_ok"] = bool(
            row["boltz2_distogram_finite"]
            and row["boltz2_plddt_finite"]
            and row["boltz2_pae_finite"]
        )
        scored_rows.append(row)
        print("RESULT_JSON", json.dumps(row, sort_keys=True), flush=True)

    if not all(row["boltz2_finite_scoring_ok"] for row in scored_rows):
        raise RuntimeError(
            "At least one candidate produced non-finite Boltz2 distogram/pLDDT/PAE outputs"
        )

    ranked_rows = sorted(
        scored_rows,
        key=lambda row: (row["boltz2_inter_pae_mean"], -row["boltz2_inter_contact_prob_12a_mean"]),
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["boltz2_rank"] = rank

    csv_path = args.output_dir / f"{run_id}.csv"
    json_path = args.output_dir / f"{run_id}.json"
    report_path = args.report_dir / f"{run_id}.md"
    payload = {
        "run_id": run_id,
        "args": {key: str(value) for key, value in vars(args).items()},
        "git": git_metadata(),
        "candidate_selection": selection_summary,
        "summary": summarize(ranked_rows),
        "rows": ranked_rows,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "report_path": str(report_path),
    }

    write_csv(csv_path, ranked_rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, payload)

    print("RANKED_JSON", json.dumps(ranked_rows, sort_keys=True), flush=True)
    print("csv_path", csv_path, flush=True)
    print("json_path", json_path, flush=True)
    print("report_path", report_path, flush=True)
    print("boltz2_candidate_holdout_scoring=PASS", flush=True)


if __name__ == "__main__":
    main()
