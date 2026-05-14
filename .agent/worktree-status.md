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

This worktree first focused on fair candidate-gate selection and reporting. It
now also carries the first threshold-aware update-rule prototype after fair and
multi-step candidate gates showed that `M11a`/PCGrad did not improve hard
candidate quality. It is the implementation workspace for the diagnosis recorded
in:

- root `docs/designs/constraint-aware-multi-oracle-optimization.md`
- root `docs/experiments/phase0-exp-c-candidate-gate-diagnosis-2026-05-13.md`

## Expected Difference From Parent

- Add quota-aware Boltz2 candidate selection.
- Deduplicate candidates within balance groups rather than globally before
  balancing.
- Report requested, available, deduplicated, selected, and scored candidate
  counts by method and score mode.
- Add a small aggregation/report path for EXP-C candidate gates.
- Add `M12a`, an active-constraint update rule that optimizes the primary
  contact oracle while only enforcing auxiliary losses whose current values
  exceed configured thresholds.
- Add `M12b`/`M12c`/`M12d` threshold variants to test whether the `pdl1`
  failure mode comes from activating the solubility constraint too early.

## Linked Claims And Risks

- Claim: update-level gradient arbitration can reduce oracle harm.
- Open risk: update-level safety did not automatically improve hard-candidate
  quality in EXP-C.
- Evaluation risk: EXP-C Boltz2 scored subset was imbalanced after global
  sequence deduplication (`M3/M4/M11a = 10/9/5`).
- Current method-design risk: PCGrad protects every auxiliary objective at all
  times, which may over-constrain contact optimization; `M12a` tests the
  alternative threshold-aware framing.

## Evidence Paths

- `docs/results/`
- `docs/reports/`
- `docs/runs/`

## Exit Condition

Continue if a two-target candidate-gate rerun comparing `M3,M4,M12a` can be
launched and produces clean method/score-mode accounting plus Boltz2 holdout
scores.

Merge back only after:

- scripts pass syntax checks;
- the scorer preserves backward-compatible defaults or documents any changed
  behavior;
- a smoke run verifies selected-count reporting;
- `M12a` records active-constraint diagnostics and can run through Protenix
  candidate generation.

Park if the selection logic requires larger scorer refactoring than expected.

## Next Verification Step

Sync this branch to Quest, pull into a matching remote worktree, and run a
two-target threshold sweep with `METHOD_IDS=M3,M12a,M12b,M12c,M12d`,
`DEDUPLICATE_SCOPE=group`, and `MAX_BOLTZ2_CANDIDATES=30`. Local non-GPU checks
already passed:

- `bash -n jobs/launch_phase0_candidate_gate_parallel.sh`
- `python3 -m py_compile scripts/score_boltz2_candidate_holdout.py`
- `python3 -m py_compile scripts/run_update_geometry_diagnostic.py scripts/run_protenix_update_geometry_smoke.py scripts/run_boltz2_oracle_gradient_conflict.py`
- `git diff --check`

Local proxy execution was not run because `uv` resolves the CUDA dependency
group on macOS arm64 and cannot install `nvidia-cublas-cu12`; execution
verification should happen on Quest.
