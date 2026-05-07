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

Continue Phase 0 reduced Protenix update-geometry experiments. Latest baseline worktree result is ACT-020 warm-started CEM at pushed commit `6013a6f`.

## Blockers

- Torch is currently CPU-only because Mosaic's `pyproject.toml` routes torch through the PyTorch CPU wheel index.

## Latest Experiment State

- Baseline branch: `baseline/phase0-mosaic-baselines`
- Local worktree: `/Users/jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines`
- Quest worktree: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines`
- Latest pushed commit: `6013a6f`
- Latest report: `docs/reports/phase0_act020_warm_cem_2026-05-07.md`
- Result: warm-started CEM repairs final argmax quality, especially `WCEMp_M7c`, but source M8a/M7c top-k remains stronger at matched budget 24; next method should preserve source diversity during warm CEM.
