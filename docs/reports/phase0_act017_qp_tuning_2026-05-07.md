# Phase 0 ACT-017 QP Tuning Sweep Report

## Summary

- Goal: test whether QP-grid tuning can keep the strong M8a relaxed-interface signal while reducing auxiliary harm enough to beat M3 under matched hard-candidate top-k selection.
- Setup: reduced IL7RA ProtenixMini smoke on Quest H100, 2 seeds, 3 update steps, 24-residue binder, 48-residue target crop, 8 cold top-k samples per seed at temperature 0.25.
- Main answer: no. The tuned variants `M8c/M8d/M8e` do not reduce harm relative to `M8a`; all three converge to the same trajectory and remain above M3's update harm.
- New useful finding: `M8b`, the strict auxiliary QP reference, is the best hard top-k method in this run across BT PAE, BT ipTM, and contact reranking at budgets 4 and 8.
- Interpretation: QP-style geometry is still worth keeping, but threshold tuning alone is not enough. The next QP version should change the candidate search or fallback policy, while the next optimization direction should prioritize hard-candidate mechanisms such as CEM-lite or gradual hardening.

## 1. Experiment Motivation

ACT-015B showed a tension:

- `M8a` gave the best relaxed soft candidate, but had too much auxiliary harm and did not beat M3 on BT PAE/contact hard top-k.
- `M8b` was safer locally, but looked weaker as a terminal candidate generator.

ACT-017 asks whether this is just a tuning problem. If a middle QP setting can preserve contact descent, reduce harm, and improve hard candidates, then the project has a clean path toward a constrained multi-objective optimizer.

The broader research question is:

> Can Mosaic use oracle geometry more precisely than scalar averaging by choosing updates that preserve contact descent while bounding auxiliary damage?

## 2. Experiment Setup

| Item | Value |
|---|---|
| Run ID | `phase0_protenix_update_geometry_65947b0_20260507T091217Z` |
| Code commit | `65947b0` |
| Server | Quest `qgpu3019`, H100 80GB |
| Runtime | 16 min 50 sec, exit status 0 |
| Target | `IL7RA.cif` |
| Target length | 48 |
| Binder length | 24 |
| Seeds | 2 |
| Steps | 3 |
| Candidate top-k samples | 8 |
| Candidate sampling temperature | 0.25 |
| Compared methods | `M3,M7c,M8a,M8b,M8c,M8d,M8e` |

Primary artifacts:

- `docs/designs/phase0_act017_qp_tuning_sweep.md`
- `docs/results/phase0_protenix_update_geometry_65947b0_20260507T091217Z.json`
- `docs/results/phase0_protenix_update_geometry_65947b0_20260507T091217Z_steps.csv`
- `docs/results/phase0_protenix_update_geometry_65947b0_20260507T091217Z_candidates.csv`
- `docs/results/phase0_protenix_update_geometry_65947b0_20260507T091217Z_topk_sensitivity.md`

## 3. Core Algorithm or Method

The QP-grid approximation tries to solve:

```text
minimize_d ||d - d_contact||^2
subject to contact descent is preserved
           auxiliary directional derivatives stay below tolerance
```

ACT-017 keeps the same candidate-set approximation from ACT-015B and varies two knobs:

| Method | Aux tolerance | Contact descent ratio | Intent |
|---|---:|---:|---|
| `M8a` | 0.08 | 0.60 | Contact-aggressive ACT-015B reference. |
| `M8b` | 0.04 | 0.60 | Strict auxiliary ACT-015B reference. |
| `M8c` | 0.06 | 0.40 | Middle slack, gentler contact ratio. |
| `M8d` | 0.06 | 0.50 | Middle slack, middle contact ratio. |
| `M8e` | 0.08 | 0.40 | M8a slack, gentler contact ratio. |

This run also logs feasibility diagnostics for each QP update: selected primary ratio, auxiliary violation, contact violation, distance to the contact-only update, and whether the selected candidate is feasible.

## 4. Metrics

| Metric | Direction | Role |
|---|---|---|
| Harm rate | lower better | Fraction of oracle terms worsened by the local update. |
| Primary derivative | more negative better | Contact-oracle descent strength. |
| Auxiliary harm count | lower better | Number of auxiliary losses locally harmed. |
| Worst aux derivative | lower better | Magnitude of worst local auxiliary harm. |
| QP feasible rate | higher better | Whether selected updates satisfy the local QP constraints. |
| BT PAE | lower better | Main binder-target interface error metric. |
| BT ipTM | higher better | Binder-target interface confidence metric. |
| Contact loss | lower better | Protenix contact objective. |
| Trigram loss | lower better | Sequence naturalness sanity check. |

## 5. Results

### 5.1 Update-Level Geometry

| Method | Harm rate | Primary derivative | Aux harm count | Worst aux derivative | QP feasible | Selected primary ratio | Aux violation | Contact violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M7c | 0.1250 | -6.5726 | 0.5000 | 0.0117 | n/a | n/a | n/a | n/a |
| M3 | 0.2083 | -5.3055 | 0.8333 | 0.0102 | n/a | n/a | n/a | n/a |
| M8b | 0.2917 | -5.6076 | 1.1667 | 0.0074 | 0.8333 | 0.8650 | 0.0000 | 0.9555 |
| M8a | 0.3333 | -5.4445 | 1.3333 | 0.0192 | 1.0000 | 0.9349 | 0.0000 | 0.0000 |
| M8c | 0.3333 | -5.3822 | 1.3333 | 0.0187 | 1.0000 | 0.8523 | 0.0000 | 0.0000 |
| M8d | 0.3333 | -5.3822 | 1.3333 | 0.0187 | 1.0000 | 0.8523 | 0.0000 | 0.0000 |
| M8e | 0.3333 | -5.3822 | 1.3333 | 0.0187 | 1.0000 | 0.8523 | 0.0000 | 0.0000 |

Answer for this subquestion:

- Goal: reduce M8a's local auxiliary harm by loosening the contact ratio or changing auxiliary slack.
- Did we get the answer? Yes. The answer is negative: `M8c/M8d/M8e` do not reduce harm; they match M8a's 0.3333 harm rate.
- How did we find it? By logging local directional derivatives and QP feasibility for every seed and step.
- Are the results correct? Internally yes: all methods used the same seeds, target, steps, optimizer wrapper, and oracle evaluations. The caveat is scale: only 2 seeds and 3 steps.

### 5.2 Soft and Argmax Controls

| Method | Mode | pLDDT | BT PAE | BT ipTM | Contact loss | Trigram loss |
|---|---|---:|---:|---:|---:|---:|
| M8a | soft | 0.7338 | 8.4671 | 0.5120 | 2.2693 | 3.3025 |
| M8c | soft | 0.7235 | 8.8962 | 0.5058 | 2.3013 | 3.3052 |
| M8d | soft | 0.7235 | 8.8962 | 0.5058 | 2.3013 | 3.3052 |
| M8e | soft | 0.7235 | 8.8962 | 0.5058 | 2.3013 | 3.3052 |
| M3 | soft | 0.7070 | 9.1251 | 0.4977 | 2.2266 | 3.2113 |
| M8b | soft | 0.6726 | 10.2546 | 0.4244 | 2.3082 | 3.3071 |
| M7c | soft | 0.5238 | 15.0303 | 0.2147 | 2.8422 | 3.1777 |
| M7c | argmax | 0.5631 | 13.4850 | 0.2885 | 3.0756 | 3.1797 |
| M8b | argmax | 0.5276 | 14.0917 | 0.2580 | 2.9004 | 3.3772 |
| M3 | argmax | 0.5409 | 14.8048 | 0.2605 | 3.2083 | 3.2748 |
| M8a | argmax | 0.4992 | 15.7196 | 0.2583 | 3.1197 | 3.2589 |
| M8c | argmax | 0.4979 | 15.8491 | 0.2624 | 3.1350 | 3.2415 |
| M8d | argmax | 0.4979 | 15.8491 | 0.2624 | 3.1350 | 3.2415 |
| M8e | argmax | 0.4979 | 15.8491 | 0.2624 | 3.1350 | 3.2415 |

Answer for this subquestion:

- Goal: see whether tuned QP keeps M8a's relaxed-interface quality and whether that quality survives simple argmax.
- Did we get the answer? Yes. Tuned QP keeps most of the soft signal but does not improve hard argmax; argmax remains a destructive handoff.
- How did we find it? By scoring both relaxed soft terminals and argmax terminals with the same holdout candidate scoring.
- Are the results correct? The soft/argmax comparison is controlled, but it is not the final paper-facing metric because argmax is already known to be a poor handoff.

### 5.3 Best-of-Budget Top-K

BT PAE reranking:

| Budget | Best | M8b | M8c/d/e | M7c | M8a | M3 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | M8b | 13.4892 | 16.7240 | 16.3628 | 16.4225 | 16.1553 |
| 4 | M8b | 9.5652 | 11.0419 | 11.8927 | 13.1611 | 13.1135 |
| 8 | M8b | 9.5652 | 10.8922 | 11.8927 | 12.5003 | 12.8342 |

BT ipTM reranking:

| Budget | Best | M8b | M8c/d/e | M8a | M3 | M7c |
|---:|---|---:|---:|---:|---:|---:|
| 1 | M8b | 0.3203 | 0.2443 | 0.2513 | 0.2799 | 0.1551 |
| 4 | M8b | 0.4657 | 0.4368 | 0.3900 | 0.3620 | 0.3500 |
| 8 | M8b | 0.4657 | 0.4368 | 0.4544 | 0.4533 | 0.3500 |

Contact reranking:

| Budget | Best | M8b | M8c/d/e | M8a | M3 | M7c |
|---:|---|---:|---:|---:|---:|---:|
| 1 | M8b | 2.8616 | 3.3983 | 3.3906 | 3.3372 | 3.3948 |
| 4 | M8c/d/e | 2.5538 | 2.4611 | 2.7018 | 2.8287 | 2.8536 |
| 8 | M8c/d/e | 2.5538 | 2.4611 | 2.7018 | 2.8218 | 2.8536 |

Answer for this subquestion:

- Goal: determine whether any QP variant beats M3 under matched hard-candidate budget.
- Did we get the answer? Yes. In this run, QP variants beat M3 on hard-candidate metrics, but the winner is not the intended tuned middle variant. `M8b` wins BT PAE and BT ipTM; `M8c/d/e` win contact.
- How did we find it? By reusing the same cold candidate pool and recomputing best-per-seed top-k selection at budgets 1, 4, and 8 under BT PAE, BT ipTM, and contact reranking.
- Are the results correct? The matched-budget comparison is fair within this reduced run. The uncertainty is that `M8b` has one infeasible QP step and higher harm than M3, so this is not yet a clean method claim.

## 6. How to Read the Tables

The update-level table answers whether the optimizer obeys the intended local geometry. The soft/argmax table answers whether relaxed gains survive naive discretization. The top-k table answers the more important practical question: if we spend the same cold-sampling budget, which method gives better hard candidates?

The surprising result is that the best hard candidates come from `M8b`, even though `M8b` is not the best soft method. This reinforces the current project thesis that terminal relaxed quality is not enough; the optimizer has to be judged through hard-candidate handoff.

## 7. Interpretation

ACT-017 rejects the narrow hypothesis:

> A middle QP slack/contact-ratio setting will reduce M8a harm while preserving its quality.

The middle settings do not reduce harm. `M8c/M8d/M8e` are also nearly identical, which means the current candidate set and projection are coarse enough that these threshold changes do not expose meaningfully different updates.

ACT-017 supports a broader hypothesis:

> Constraint-aware geometry can improve hard-candidate quality over M3, but the current QP-grid implementation is not yet a well-controlled multi-objective optimizer.

`M8b` beating M3 under BT PAE and BT ipTM is a positive signal. The problem is that `M8b` does not satisfy the full acceptance check because its update harm is 0.2917 versus M3's 0.2083, and it has a QP contact violation fallback event.

## 8. Conclusion and Discussion

### What was the goal?

The goal was to decide whether QP-grid update control can be tuned into a cleaner M8 method: lower harm than M8a, better hard candidates than M3, and a more formal version of M7c's contact-preserving idea.

### Did we get the answer finally?

Yes. The specific tuning attempt failed, but the experiment produced a useful new lead. `M8c/M8d/M8e` are not the answer. `M8b` deserves a follow-up because it gives the strongest hard-candidate BT PAE/ipTM in this reduced run.

### How did we find the answer?

We implemented additional QP variants and diagnostics, ran them on the same reduced ProtenixMini IL7RA setup as previous ACT runs, then compared:

1. local update geometry;
2. soft and argmax terminal scoring;
3. matched-budget cold top-k selection under three reranking metrics.

### Are the results correct?

They are correct as a reduced controlled experiment. The raw files, markdown report, and top-k sensitivity artifacts are saved under the run ID above. The results should not be overclaimed as final binder-design performance because the run uses 2 seeds, 3 steps, reduced target length, and ProtenixMini.

## 9. Limitations and Caveats

- The QP is still a grid approximation over existing directions, not a continuous QP solve.
- `M8b` has one infeasible selected step, so its hard-candidate win is not a clean constrained-optimization win.
- `M8c/M8d/M8e` collapsing to identical outputs suggests the current direction set is too coarse for this threshold sweep.
- The benchmark remains small: 2 seeds, 3 steps, 24-residue binders, 48-residue target crop.
- The run measures reduced-oracle candidates only; it does not yet include full Boltz/Protenix cross-oracle robustness.

## 10. Next Steps

1. Keep `M8b` as the current QP hard-candidate reference, but do not scale it as a final method yet.
2. Add a stricter fallback policy for QP: fallback should penalize contact violation and auxiliary harm jointly, rather than allowing a large contact violation event.
3. Add a candidate-set expansion or continuous projected solver; the current threshold sweep is too coarse.
4. Start a hard-candidate optimizer line in parallel, preferably CEM-lite or gradual position-wise hardening, because ACT-017 again shows that hard handoff is the bottleneck.
5. Keep M3 as the primary matched-budget baseline for any next reduced gate.

## Reproducibility Notes

Run command on Quest:

```bash
bash jobs/quest_protenix_act017_qp_tuning_h100.slurm
```

Local validation before submission:

```bash
python3 -m py_compile scripts/run_update_geometry_diagnostic.py scripts/run_protenix_update_geometry_smoke.py scripts/summarize_topk_handoff_sensitivity.py
bash -n jobs/quest_protenix_act017_qp_tuning_h100.slurm
git diff --check
```

The run completed with exit status 0 and produced the artifacts listed in Section 2.
