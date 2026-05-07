# Code Ops Status

Last updated: 2026-05-07

## State

Official Mosaic cloned locally. Upstream remote is `upstream`. Writable fork remote `origin` is `git@github.com:a-green-hand-jack/mosaic.git`.

Quest clone exists at `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code`. Verify the exact synchronized commit with `git rev-parse --short HEAD` before running jobs.

Quest environment uses uv:

- uv: `/home/zpt6685/.local/bin/uv`
- cache: `/projects/p32572/Jieke/.cache/uv`
- uv-managed Python root: `/projects/p32572/Jieke/Envs/uv-python`
- venv: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code/.venv`
- Python: 3.12.12
- sync command: `uv sync --group jax-cuda`
- GPU policy: use one uv environment across A100 and H100, then smoke-test each GPU family.
- smoke script: `jobs/quest_jax_cuda_smoke.slurm`

## Current Focus

Phase 0 ACT-010: diagnose why safer Protenix update geometry improves update-level metrics but does not yet consistently improve final argmax candidates.

## Latest Run

- Commit: `f092264`
- Run ID: `phase0_protenix_update_geometry_f092264_20260507T042412Z`
- Server/node: Quest H100 `qgpu3019`
- Runtime: 16:24 wall-clock, exit status 0
- Report: `docs/reports/phase0_act010_contact_sweep_2026-05-07.md`
- Raw artifacts:
  - `docs/reports/phase0_protenix_update_geometry_f092264_20260507T042412Z.md`
  - `docs/results/phase0_protenix_update_geometry_f092264_20260507T042412Z.json`
  - `docs/results/phase0_protenix_update_geometry_f092264_20260507T042412Z_steps.csv`
  - `docs/results/phase0_protenix_update_geometry_f092264_20260507T042412Z_candidates.csv`

Result summary: aggressive contact-preserving update `M7c` beats naive/normalized baselines under soft terminal scoring, but the advantage is mostly lost after argmax discretization. The next method step should target discretization-aware candidate handoff rather than only tuning cone slack.

## Blockers

- Torch is currently CPU-only because Mosaic's `pyproject.toml` routes torch through the PyTorch CPU wheel index.
