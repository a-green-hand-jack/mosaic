# Worktree Status

Created: 2026-05-13

## Identity

- Branch: `feature/constraint-aware-candidate-gate`
- Path: `/Users/jieke/Projects/SCH-BinderDesign/code-worktrees/feature-constraint-aware-candidate-gate`
- Parent branch: `baseline/phase0-mosaic-baselines`
- Parent commit: `3be9949`
- Component: Mosaic-based code repo

## Purpose

Practice the new project framing by implementing the first concrete bridge from
update-level oracle safety to constraint-aware candidate evaluation.

This worktree should focus on fair candidate-gate selection and reporting before
changing the optimizer itself. It is the implementation workspace for the
diagnosis recorded in:

- root `docs/designs/constraint-aware-multi-oracle-optimization.md`
- root `docs/experiments/phase0-exp-c-candidate-gate-diagnosis-2026-05-13.md`

## Expected Difference From Parent

- Add quota-aware Boltz2 candidate selection.
- Deduplicate candidates within balance groups rather than globally before
  balancing.
- Report requested, available, deduplicated, selected, and scored candidate
  counts by method and score mode.
- Add a small aggregation/report path for EXP-C candidate gates.

## Linked Claims And Risks

- Claim: update-level gradient arbitration can reduce oracle harm.
- Open risk: update-level safety did not automatically improve hard-candidate
  quality in EXP-C.
- Evaluation risk: EXP-C Boltz2 scored subset was imbalanced after global
  sequence deduplication (`M3/M4/M11a = 10/9/5`).

## Evidence Paths

- `docs/results/`
- `docs/reports/`
- `docs/runs/`

## Exit Condition

Continue if a fair two-target EXP-C rerun can be launched and produces a clean
method/score-mode candidate accounting table.

Merge back only after:

- scripts pass syntax checks;
- the scorer preserves backward-compatible defaults or documents any changed
  behavior;
- a smoke run verifies selected-count reporting.

Park if the selection logic requires larger scorer refactoring than expected.

## Next Verification Step

Sync this branch to Quest, pull into a matching remote worktree, and run the
two-target EXP-C smoke with `DEDUPLICATE_SCOPE=group`. Local non-GPU checks
already passed:

- `bash -n jobs/launch_phase0_candidate_gate_parallel.sh`
- `python3 -m py_compile scripts/score_boltz2_candidate_holdout.py`
- `git diff --check`
