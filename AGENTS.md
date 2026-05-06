# Code Agent Guidance

This repo is a Mosaic-derived code component for SCH-BinderDesign.

## Remotes

- `upstream`: official `escalante-bio/mosaic`.
- `origin`: writable fork at `git@github.com:a-green-hand-jack/mosaic.git`.

Do not push to `upstream`.

## Current Priority

Implement Phase 0 trajectory geometry analysis before adding the full SCH optimizer.

Expected report path:

- `docs/reports/phase0-claim1-trajectory-geometry.md`

Expected run/provenance paths:

- `docs/runs/`
- `docs/results/`

## Checks

Default checks:

- `uv sync --group jax-cuda`
- `uv run ruff format --check src tests examples`
- `uv run ruff check src tests examples`
- `uv run pytest tests -v`

Run dependency-heavy and GPU checks on Quest unless the task explicitly only needs local static inspection.

## Quest

Use GitHub sync for Quest execution. Push a local commit to `origin`, then pull that commit under the project execution root recorded in the root control repo.

Do not store large weights, model caches, datasets, checkpoints, or raw outputs in this repo.
