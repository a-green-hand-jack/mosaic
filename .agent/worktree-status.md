# Worktree Status: baseline/phase0-mosaic-baselines

## Purpose

Prepare a runnable Phase 0 baseline pilot for SCH-BinderDesign on Quest before implementing the full SCH optimizer.

## Scope

- Convert Mosaic example workflows into a controlled baseline plan.
- Define baseline methods, target set, metrics, seeds, controls, and stop rules.
- Add a Quest SLURM entrypoint that uses the shared uv and Boltz cache layout.
- Keep this branch focused on baseline planning and pilot plumbing, not the SCH optimizer.

## Linked Memory

- ACT-005: Run compute-node JAX CUDA smoke check.
- ACT-006: Calibrate baseline filter thresholds.
- ACT-007: Prepare durable Boltz checkpoint cache.
- CLM-001: Holdout-gradient insensitivity signature.
- CLM-002: Cross-oracle robustness metric design.
- EVD-003: In silico filter metric taxonomy.
- EVD-004: Quest checkpoint inventory.

## Latest Reliable State

- Branch created from `main` at `ef16d87`.
- Local worktree path: `/Users/jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines`.
- Intended Quest worktree path: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines`.
- Initial deliverable is a dry-run-safe pilot runner that records provenance before any expensive oracle calls.

## Exit Condition

- Baseline pilot plan and fairness ledger exist under `docs/reports/`.
- Quest job wrapper can run from the worktree and emits a provenance JSON.
- Quest clone has the branch/worktree checked out at the pushed commit.
- A100/H100 smoke result is recorded or explicitly blocked by queue availability.
