#!/usr/bin/env python3
"""Record provenance for the Phase 0 baseline pilot.

This script is deliberately dry-run first. It fixes the run schema and Quest
entrypoint before we spend GPU time on expensive structure-prediction oracles.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any


def run_command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def try_run_command(args: list[str], default: str | None = None) -> str | None:
    try:
        return run_command(args)
    except Exception:
        return default


def read_config_snapshot(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def maybe_module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed")


def collect_gpu_info() -> list[dict[str, str]]:
    try:
        output = run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
    except Exception:
        return []

    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total": parts[2],
                }
            )
    return devices


def collect_baseline_status() -> list[dict[str, Any]]:
    return [
        {
            "id": "B0",
            "name": "random_sequence_scoring",
            "status": "planned",
            "runner": None,
        },
        {
            "id": "B1",
            "name": "protenix_single_oracle_hallucination",
            "status": "planned",
            "source_example": "examples/batched_protenix.py",
        },
        {
            "id": "B2",
            "name": "boltzgen_boltz2_ranking",
            "status": "blocked_on_boltz_cache",
            "source_example": "examples/boltzgen_pipeline.py",
        },
        {
            "id": "B3",
            "name": "mosaic_weighted_composite",
            "status": "planned_after_B1_B2",
            "runner": None,
        },
    ]


def build_report(config_path: Path, dry_run: bool) -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    commit = try_run_command(["git", "rev-parse", "HEAD"], default="unavailable")
    short_commit = try_run_command(["git", "rev-parse", "--short", "HEAD"], default="nogit")
    branch = try_run_command(["git", "branch", "--show-current"], default="unavailable")
    dirty_status = try_run_command(["git", "status", "--short"], default=None)

    return {
        "schema_version": 1,
        "run_id": f"phase0_il7ra_{short_commit}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at_utc": now,
        "dry_run": dry_run,
        "config_path": str(config_path),
        "config_snapshot": read_config_snapshot(config_path),
        "git": {
            "commit": commit,
            "short_commit": short_commit,
            "branch": branch,
            "dirty": None if dirty_status is None else bool(dirty_status),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "environment": {
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR"),
            "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR"),
            "BOLTZ_CACHE": os.environ.get("BOLTZ_CACHE"),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        },
        "versions": {
            "jax": maybe_module_version("jax"),
            "torch": maybe_module_version("torch"),
            "mosaic": maybe_module_version("mosaic"),
            "gemmi": maybe_module_version("gemmi"),
        },
        "gpu": collect_gpu_info(),
        "baselines": collect_baseline_status(),
        "next_required_actions": [
            "Complete A100/H100 JAX CUDA smoke checks.",
            "Hydrate and verify /projects/p32572/Jieke/.cache/boltz.",
            "Implement B0 candidate generation and shared scoring schema.",
            "Adapt examples/batched_protenix.py into a non-notebook B1 runner.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/phase0_il7ra.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/runs"))
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    report = build_report(config_path=config_path, dry_run=args.dry_run)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{report['run_id']}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"run_id": report["run_id"], "output_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
