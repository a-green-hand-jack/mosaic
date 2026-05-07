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
- ACT-017: Tune QP-style constrained update and diagnose feasibility.
- ACT-018: Revise QP fallback and grid search.
- ACT-019: Test gradual position-wise hardening.
- ACT-020: Test warm-started hard-candidate optimization.
- ACT-021: Test diversity-preserving warm CEM.
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
- Latest pushed commit: verify with `git rev-parse --short HEAD` before running new jobs. Latest result-bearing commit is `eb0ca34`; latest ops/status commits are after that in branch history.
- ACT-010 result: H100 run `phase0_protenix_update_geometry_f092264_20260507T042412Z` completed successfully and is summarized in `docs/reports/phase0_act010_contact_sweep_2026-05-07.md`. M7c aggressive contact-preserving update is best under soft terminal scoring, but argmax loses most of the gain.
- ACT-011 implementation commit: `09ed71d`.
- ACT-011 result: H100 run `phase0_protenix_update_geometry_09ed71d_20260507T045421Z` completed successfully and is summarized in `docs/reports/phase0_act011_topk_handoff_2026-05-07.md`. M7c top-k sample/rerank beats M7c argmax and naive weighted top-k/argmax on discrete interface metrics under matched sample budget.
- ACT-012 implementation commit: `cc6864e`.
- ACT-012 result: H100 run `phase0_protenix_update_geometry_cc6864e_20260507T051752Z` completed successfully and is summarized in `docs/reports/phase0_act012_topk_sensitivity_2026-05-07.md`. M7c improves with sample budget, but M3 naive weighted is best across tested budgets and rerank metrics.
- ACT-013 result: lower terminal sampling temperature, especially `0.25`, revives M7c top-k candidate quality and is summarized in `docs/reports/phase0_act013_terminal_temperature_2026-05-07.md`.
- ACT-014 result: naive post-update entropy annealing lowers entropy but worsens update safety and candidate quality; summarized in `docs/reports/phase0_act014_entropy_annealing_2026-05-07.md`.
- ACT-015A result: naive CEM / elite sampling lowers entropy but fails under matched 24-candidate budget; summarized in `docs/reports/phase0_act015a_cem_2026-05-07.md`.
- ACT-015B result: QP-grid M8a strongly improves soft terminal interface metrics and top-k BT ipTM, but M3 remains best on BT PAE/contact and M8a increases update harm; summarized in `docs/reports/phase0_act015b_qp_2026-05-07.md`.
- ACT-017 result: QP threshold tuning did not reduce harm; M8c/M8d/M8e collapse to the same trajectory. M8b unexpectedly gives the best matched-budget hard top-k BT PAE/ipTM in the reduced run, but it still exceeds M3 update harm and has an infeasible fallback step; summarized in `docs/reports/phase0_act017_qp_tuning_2026-05-07.md`.
- ACT-018 result: QP fallback and grid revisions did not pass the gate. Contact-first fallback trades M8b's contact violation for auxiliary violation and higher harm; denominator-20 M8h exactly matches M8b. M8a/M8g win hard top-k in the reduced run but have worse harm than M3; summarized in `docs/reports/phase0_act018_qp_fallback_2026-05-07.md`.
- ACT-019 implementation commit: `f9d49f8`.
- ACT-019 result: naive gradual position hardening lowers entropy and freezes positions, but increases update harm and degrades soft/argmax candidate quality. `M9b` has a narrow best-of-8 BT PAE signal, but it does not pass the combined safety-quality gate; summarized in `docs/reports/phase0_act019_gradual_hardening_2026-05-07.md`.
- ACT-020 implementation commit: `2dd3965`.
- ACT-020 result: warm-started CEM from M3/M7c/M8a terminals improves final argmax quality, especially `WCEMp_M7c` with BT PAE 9.8045 versus M7c argmax 13.1110 and M3 argmax 15.6512. It does not beat the strongest matched-budget source top-k baselines at budget 24; summarized in `docs/reports/phase0_act020_warm_cem_2026-05-07.md`.
- ACT-021 implementation commit: `82f0356`.
- ACT-021 status: diversity-preserving warm CEM (`DWCEMp`, `DWCEMc`) is implemented in `scripts/run_protenix_cem_optimizer.py`, documented in `docs/designs/phase0_act021_diverse_warm_cem.md`, and has a Quest job wrapper at `jobs/quest_protenix_act021_diverse_warm_cem_h100.slurm`. Quest worktree is synced to `82f0356`.
- ACT-021 queued job on Quest: `7430412` (`sch-act021-anygpu`, command-line `--gres=gpu:1` fallback) is PENDING with reason `Priority`. The duplicate H100-only pending job `7430374` was canceled to avoid double-running the same experiment.
- Interpretation: gradient-guided constrained updates remain promising, but QP-grid threshold/fallback/grid tuning and naive early hardening should be parked. Warm-started hard-candidate optimization is useful for argmax repair, but needs diversity-preserving updates before it can beat source top-k under matched budget. ACT-021 will test whether preserving source terminal diversity during elite updates closes that gap.

## Exit Condition

- Baseline pilot plan and fairness ledger exist under `docs/reports/`.
- Quest job wrapper can run from the worktree and emits a provenance JSON.
- Quest clone has the branch/worktree checked out at the pushed commit.
- A100/H100 smoke result is recorded or explicitly blocked by queue availability.
- M0/M1/M3 update-level logging is ready for a Phase 0 diagnostic run.
- Soft and argmax candidate scoring are separated in the output schema.
- ACT-018 blocks immediate QP-grid scale-up, ACT-019 blocks naive early hardening scale-up, and ACT-020 shows warm CEM repairs argmax but not best-of-budget quality. Next exit gate is a diversity-preserving hard-candidate optimizer or delayed gated hardening method that improves matched-budget BT PAE/contact without exceeding M3 update harm or is explicitly framed as post-hoc candidate selection.
