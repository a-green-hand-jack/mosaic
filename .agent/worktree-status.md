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
- ACT-010: Tune contact-preserving updates and discretization diagnostics.
- ACT-011: Add discretization-aware candidate handoff.
- ACT-015: Test CEM or constrained discrete repair.
- ACT-016: Implement QP-style constrained update.
- CLM-001: Holdout-gradient insensitivity signature.
- CLM-002: Cross-oracle robustness metric design.
- EVD-003: In silico filter metric taxonomy.
- EVD-004: Quest checkpoint inventory.
- EVD-014: ACT-010 contact sweep and soft-vs-argmax diagnostic.

## Latest Reliable State

- Branch created from `main` at `ef16d87`.
- Local worktree path: `/Users/jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines`.
- Intended Quest worktree path: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines`.
- Initial deliverable is a dry-run-safe pilot runner that records provenance before any expensive oracle calls.
- Current experiment scope excludes BoltzGen generation/ranking and external BindCraft/BoltzDesign1 baselines; the branch should compare Mosaic-internal update rules.
- Latest pushed commit: `e76d8bd`.
- ACT-010 result: H100 run `phase0_protenix_update_geometry_f092264_20260507T042412Z` completed successfully and is summarized in `docs/reports/phase0_act010_contact_sweep_2026-05-07.md`. M7c aggressive contact-preserving update is best under soft terminal scoring, but argmax loses most of the gain.
- ACT-011 implementation commit: `09ed71d`.
- ACT-011 result: H100 run `phase0_protenix_update_geometry_09ed71d_20260507T045421Z` completed successfully and is summarized in `docs/reports/phase0_act011_topk_handoff_2026-05-07.md`. M7c top-k sample/rerank beats M7c argmax and naive weighted top-k/argmax on discrete interface metrics under matched sample budget.
- ACT-012 implementation commit: `cc6864e`.
- ACT-012 result: H100 run `phase0_protenix_update_geometry_cc6864e_20260507T051752Z` completed successfully and is summarized in `docs/reports/phase0_act012_topk_sensitivity_2026-05-07.md`. M7c improves with sample budget, but M3 naive weighted is best across tested budgets and rerank metrics.
- ACT-013 result: lower terminal sampling temperature, especially `0.25`, revives M7c top-k candidate quality and is summarized in `docs/reports/phase0_act013_terminal_temperature_2026-05-07.md`.
- ACT-014 result: naive post-update entropy annealing lowers entropy but worsens update safety and candidate quality; summarized in `docs/reports/phase0_act014_entropy_annealing_2026-05-07.md`.
- ACT-015A result: naive CEM / elite sampling lowers entropy but fails under matched 24-candidate budget; summarized in `docs/reports/phase0_act015a_cem_2026-05-07.md`.
- ACT-015B result: QP-grid M8a strongly improves soft terminal interface metrics and top-k BT ipTM, but M3 remains best on BT PAE/contact and M8a increases update harm; summarized in `docs/reports/phase0_act015b_qp_2026-05-07.md`.
- Interpretation: gradient-guided constrained updates remain promising, but the current QP-grid settings are too contact-aggressive. The next branch task is a small QP tuning sweep over auxiliary slack and contact descent ratio.

## Exit Condition

- Baseline pilot plan and fairness ledger exist under `docs/reports/`.
- Quest job wrapper can run from the worktree and emits a provenance JSON.
- Quest clone has the branch/worktree checked out at the pushed commit.
- A100/H100 smoke result is recorded or explicitly blocked by queue availability.
- M0/M1/M3 update-level logging is ready for a Phase 0 diagnostic run.
- Soft and argmax candidate scoring are separated in the output schema.
- ACT-015B blocks immediate QP scale-up; next exit gate is a tuned constrained update that beats M3 top-k under matched hard-candidate budget while not exceeding M3 update harm.
