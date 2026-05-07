# Phase 0 ACT-018 QP Fallback And Grid Report

## Summary

- Goal: test whether the useful ACT-017 `M8b` hard-candidate signal can be made cleaner by changing QP fallback order or increasing grid resolution.
- Setup: reduced IL7RA ProtenixMini smoke on Quest H100, 2 seeds, 3 update steps, 24-residue binder, 48-residue target crop, 8 cold top-k samples per seed at temperature 0.25.
- Main answer: no. Contact-first fallback does not pass the safety gate, and a higher-resolution grid does not change the selected M8b trajectory.
- Best candidate result: `M8a/M8g` win hard top-k BT PAE, BT ipTM, and contact at budget 8, but both have higher update harm than M3.
- Decision: stop spending runs on QP threshold/fallback/grid tuning. The next method should move to gradual hardening or a hard-candidate optimizer warm-started from a useful relaxed trajectory.

## 1. Experiment Motivation

ACT-017 left one ambiguous signal. Strict QP `M8b` gave the best hard top-k BT PAE/ipTM in that run, but had higher update harm than M3 and one infeasible fallback step with large contact violation.

ACT-018 asks two narrow questions:

1. Was `M8b` bad only because the fallback prioritized auxiliary violation too strongly over contact violation?
2. Was the QP candidate grid too coarse, so a higher-resolution grid could find a better feasible strict-QP direction?

This matters because a positive answer would keep the QP line alive as a clean constrained multi-objective optimizer. A negative answer means the project should move to a different hard-candidate mechanism.

## 2. Experiment Setup

| Item | Value |
|---|---|
| Run ID | `phase0_protenix_update_geometry_dcf9b7c_20260507T100940Z` |
| Code commit | `dcf9b7c` |
| Server | Quest `qgpu3019`, H100 80GB |
| Runtime | 17 min 58 sec, exit status 0 |
| Target | `IL7RA.cif` |
| Target length | 48 |
| Binder length | 24 |
| Seeds | 2 |
| Steps | 3 |
| Candidate top-k samples | 8 |
| Candidate sampling temperature | 0.25 |
| Compared methods | `M3,M7c,M8a,M8b,M8f,M8g,M8h` |

Primary artifacts:

- `docs/designs/phase0_act018_qp_fallback_and_grid.md`
- `docs/results/phase0_protenix_update_geometry_dcf9b7c_20260507T100940Z.json`
- `docs/results/phase0_protenix_update_geometry_dcf9b7c_20260507T100940Z_steps.csv`
- `docs/results/phase0_protenix_update_geometry_dcf9b7c_20260507T100940Z_candidates.csv`
- `docs/results/phase0_protenix_update_geometry_dcf9b7c_20260507T100940Z_topk_sensitivity.md`

## 3. Core Algorithm or Method

ACT-018 keeps the same QP-grid objective but adds two variants:

| Method | Selector | Aux tolerance | Contact ratio | Grid denominator | Purpose |
|---|---|---:|---:|---:|---|
| `M8b` | aux-first fallback | 0.04 | 0.60 | 10 | ACT-017 strict-QP reference. |
| `M8f` | contact-first fallback | 0.04 | 0.60 | 10 | Test whether protecting contact first fixes M8b fallback. |
| `M8g` | contact-first fallback | 0.06 | 0.60 | 10 | Slightly looser contact-first variant. |
| `M8h` | aux-first fallback | 0.04 | 0.60 | 20 | Test whether higher-resolution grid changes M8b. |

The contact-first fallback changes only infeasible selection:

```text
old fallback: minimize aux violation, then contact violation
new fallback: minimize contact violation, then aux violation
```

## 4. Metrics

| Metric | Direction | Role |
|---|---|---|
| Harm rate | lower better | Whether the local update worsens any oracle terms. |
| QP feasible rate | higher better | Whether selected QP updates satisfy contact and auxiliary constraints. |
| QP contact violation | lower better | Whether fallback sacrifices contact descent. |
| QP aux violation | lower better | Whether fallback sacrifices auxiliary constraints. |
| BT PAE | lower better | Main hard-candidate interface metric. |
| BT ipTM | higher better | Interface-confidence metric. |
| Contact loss | lower better | Protenix contact objective. |
| Trigram loss | lower better | Sequence naturalness sanity check. |

## 5. Results

### 5.1 Update-Level Geometry

| Method | Harm rate | Primary derivative | Aux harm count | Worst aux derivative | QP feasible | Aux violation | Contact violation | Primary ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M7c | 0.1250 | -3.3634 | 0.5000 | 0.0095 | n/a | n/a | n/a | n/a |
| M3 | 0.2083 | -5.3565 | 0.8333 | 0.0103 | n/a | n/a | n/a | n/a |
| M8b | 0.2917 | -6.2158 | 1.1667 | 0.0061 | 0.8333 | 0.0000 | 0.9731 | 0.8645 |
| M8h | 0.2917 | -6.2158 | 1.1667 | 0.0061 | 0.8333 | 0.0000 | 0.9731 | 0.8645 |
| M8f | 0.3333 | -5.3861 | 1.3333 | 0.0154 | 0.8333 | 0.0014 | 0.0000 | 0.9375 |
| M8a | 0.3750 | -5.4396 | 1.5000 | 0.0203 | 1.0000 | 0.0000 | 0.0000 | 0.9374 |
| M8g | 0.3750 | -5.4396 | 1.5000 | 0.0203 | 1.0000 | 0.0000 | 0.0000 | 0.9374 |

Answer for this subquestion:

- Goal: eliminate the M8b contact-violation fallback while preserving candidate quality and reducing harm.
- Did we get the answer? Yes. Contact-first fallback removes the average contact violation in `M8f`, but introduces auxiliary violation and increases harm to 0.3333.
- How did we find it? By adding QP diagnostics and aggregating selected feasibility, contact violation, aux violation, and directional derivatives.
- Are the results correct? Internally yes: the same seeds, target crop, candidate budget, and oracle setup were used. The caveat is reduced scale.

### 5.2 Soft And Argmax Controls

| Method | Mode | pLDDT | BT PAE | BT ipTM | Contact loss | Trigram loss |
|---|---|---:|---:|---:|---:|---:|
| M8a | soft | 0.7317 | 8.7733 | 0.5189 | 2.2505 | 3.3112 |
| M8g | soft | 0.7317 | 8.7733 | 0.5189 | 2.2505 | 3.3112 |
| M3 | soft | 0.7061 | 9.5202 | 0.4521 | 2.3085 | 3.2164 |
| M8f | soft | 0.6275 | 12.0293 | 0.3605 | 2.7618 | 3.3061 |
| M8b | soft | 0.6045 | 12.7037 | 0.3013 | 2.8432 | 3.3066 |
| M8h | soft | 0.6045 | 12.7037 | 0.3013 | 2.8432 | 3.3066 |
| M7c | soft | 0.5114 | 15.1401 | 0.2151 | 2.9642 | 3.1569 |
| M8a | argmax | 0.5601 | 13.5433 | 0.3320 | 2.8391 | 3.1758 |
| M8g | argmax | 0.5601 | 13.5433 | 0.3320 | 2.8391 | 3.1758 |
| M7c | argmax | 0.5286 | 14.6620 | 0.2724 | 3.2752 | 2.8716 |
| M3 | argmax | 0.4935 | 16.2362 | 0.2374 | 3.1624 | 3.3002 |
| M8f | argmax | 0.4862 | 16.7880 | 0.2241 | 3.1296 | 3.3409 |
| M8b | argmax | 0.4090 | 19.0436 | 0.0837 | 3.5486 | 3.3275 |
| M8h | argmax | 0.4090 | 19.0436 | 0.0837 | 3.5486 | 3.3275 |

Answer for this subquestion:

- Goal: test whether the QP variants improve terminal relaxed quality or simple hard decoding.
- Did we get the answer? Yes. `M8a/M8g` are strongest in soft and argmax controls, but they are also the least safe QP variants by update harm.
- How did we find it? By scoring soft and argmax terminal candidates with the same Protenix holdout metrics.
- Are the results correct? Yes for this reduced run. They should not be treated as final binder-design pass-rate evidence.

### 5.3 Best-of-Budget Top-K

BT PAE reranking:

| Budget | Best | M8a/M8g | M8f | M8b/M8h | M3 | M7c |
|---:|---|---:|---:|---:|---:|---:|
| 1 | M8a/M8g | 12.4553 | 15.3120 | 15.9536 | 14.5224 | 14.2548 |
| 4 | M8a/M8g | 11.7196 | 14.5763 | 15.3099 | 13.5906 | 14.2548 |
| 8 | M8a/M8g | 11.3190 | 12.8555 | 12.9064 | 13.5906 | 13.6600 |

BT ipTM reranking:

| Budget | Best | M8a/M8g | M8f | M8b/M8h | M3 | M7c |
|---:|---|---:|---:|---:|---:|---:|
| 1 | M8a/M8g | 0.4363 | 0.3049 | 0.2576 | 0.3736 | 0.2892 |
| 4 | M8a/M8g | 0.4493 | 0.3447 | 0.2946 | 0.3850 | 0.2892 |
| 8 | M8a/M8g | 0.4493 | 0.3629 | 0.3129 | 0.3850 | 0.3376 |

Contact reranking:

| Budget | Best | M8a/M8g | M8f | M8b/M8h | M3 | M7c |
|---:|---|---:|---:|---:|---:|---:|
| 1 | M8a/M8g | 2.8403 | 3.0789 | 3.2119 | 3.0689 | 3.0319 |
| 4 | M8a/M8g | 2.6901 | 2.9596 | 3.2119 | 2.8988 | 2.9908 |
| 8 | M8a/M8g | 2.6839 | 2.7247 | 2.6947 | 2.7850 | 2.9037 |

Answer for this subquestion:

- Goal: decide whether any ACT-018 variant beats M3 under matched hard-candidate budget without worse update harm.
- Did we get the answer? Yes. `M8a/M8g` beat M3 on hard top-k metrics, but both have worse update harm. `M8f/M8b/M8h` do not satisfy the combined safety-quality gate.
- How did we find it? By reusing the same candidate pool and recomputing best-per-seed top-k selection for budgets 1, 4, and 8 under BT PAE, BT ipTM, and contact reranking.
- Are the results correct? The matched-budget comparison is fair inside this reduced run. It remains provisional until repeated with more seeds/targets.

## 6. How to Read the Tables

The update table answers whether the local optimizer is safe. The top-k tables answer whether the terminal distribution gives good hard candidates under matched candidate scoring budget.

The important comparison is not simply who wins top-k. `M8a/M8g` win the reduced hard-candidate table, but they fail the safety side of the gate. `M8f` fixes contact violation but raises harm. `M8h` tests whether a finer grid helps; it does not, because it exactly matches `M8b` in all reported aggregate metrics.

## 7. Interpretation

ACT-018 rejects two QP rescue hypotheses:

1. Fallback order is not the main problem. Contact-first fallback trades contact violation for auxiliary violation and higher harm.
2. Grid coarseness at denominator 10 is not the main problem for strict QP. Denominator 20 selects the same trajectory as `M8b`.

The positive part is narrower: `M8a/M8g` again show that QP-like contact-aggressive directions can produce strong soft and hard candidates in reduced runs. The negative part is decisive for method planning: this version of QP does not produce a clean constrained optimizer because quality and update safety still trade off too strongly.

## 8. Conclusion and Discussion

### What was the goal?

The goal was to see whether ACT-017's strict-QP signal could be made cleaner by fixing fallback behavior or increasing grid resolution.

### Did we get the answer finally?

Yes. This route is exhausted for now. Contact-first fallback and a higher-resolution grid do not pass the acceptance gate.

### How did we find the answer?

We implemented `M8f/M8g/M8h`, ran them against M3, M7c, M8a, and M8b on the same reduced ProtenixMini IL7RA setup, then compared local QP diagnostics and matched-budget top-k candidate quality.

### Are the results correct?

The run completed with exit status 0. The result is internally consistent: diagnostics show the exact safety tradeoff, and top-k tables show that candidate quality still comes from the more contact-aggressive, higher-harm variants.

## 9. Limitations and Caveats

- Reduced ProtenixMini setup only: 2 seeds, 3 steps, 24-residue binder, 48-residue target crop.
- No cross-oracle holdout or final filter pass-rate evaluation.
- `M8h` increases CPU-side candidate search and took 17:58 wall-clock, but did not alter the selected trajectory.
- The report rejects only this QP-grid/fallback line, not all constrained optimization formulations.

## 10. Next Steps

1. Park QP-grid threshold/fallback/grid tuning.
2. Start gradual position-wise hardening: commit high-confidence positions progressively rather than using terminal argmax.
3. Alternatively, test warm-started CEM-lite from M3/M8a/M8g terminal distributions, not from a random relaxed distribution.
4. Keep M3 and the best contact-aggressive QP variant as controls, but require non-worse update harm before any scale-up claim.

## Reproducibility Notes

Run command on Quest:

```bash
bash jobs/quest_protenix_act018_qp_fallback_h100.slurm
```

Local validation before submission:

```bash
python3 -m py_compile scripts/run_update_geometry_diagnostic.py scripts/run_protenix_update_geometry_smoke.py scripts/summarize_topk_handoff_sensitivity.py
bash -n jobs/quest_protenix_act018_qp_fallback_h100.slurm
git diff --check
```

The run completed on Quest `qgpu3019` with exit status 0.
