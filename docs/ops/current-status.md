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

Continue Phase 0 reduced Protenix update-geometry experiments. Latest baseline worktree result is ACT-019 gradual hardening at pushed commit `b92238c`.

## Blockers

- Torch is currently CPU-only because Mosaic's `pyproject.toml` routes torch through the PyTorch CPU wheel index.

## Latest Experiment State

- Baseline branch: `baseline/phase0-mosaic-baselines`
- Local worktree: `/Users/jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines`
- Quest worktree: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines`
- Latest pushed commit: `b92238c`
- Latest report: `docs/reports/phase0_act019_gradual_hardening_2026-05-07.md`
- Result: naive early position hardening lowers entropy but increases update harm and worsens soft/argmax quality; next method should be warm-started hard-candidate optimization.
