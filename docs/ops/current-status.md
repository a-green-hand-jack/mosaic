# Code Ops Status

Last updated: 2026-05-06

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

Prepare Phase 0 trajectory geometry instrumentation.

## Blockers

- JAX CUDA smoke check must run on both A100 and H100 compute nodes when possible because Quest exposes multiple GPU families.
- Torch is currently CPU-only because Mosaic's `pyproject.toml` routes torch through the PyTorch CPU wheel index.
