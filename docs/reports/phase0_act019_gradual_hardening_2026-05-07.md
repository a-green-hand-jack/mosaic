# Phase 0 ACT-019 Gradual Position Hardening Report

## Summary

- Goal: test whether gradual position-wise hardening can reduce the relaxed-to-discrete gap that makes terminal argmax destroy soft gains.
- Setup: reduced IL7RA ProtenixMini smoke on Quest H100, 2 seeds, 3 update steps, 24-residue binder, 48-residue target crop, and 8 cold top-k samples per seed at temperature 0.25.
- Main answer: no for the tested variants. Hardening lowered entropy and did freeze positions, but it worsened update harm and degraded soft, argmax, and most hard top-k candidate metrics.
- Narrow positive signal: `M9b` is best by BT PAE at top-k budget 8, but it does not pass the combined gate because its update harm is higher than M3 and its soft terminal quality is much worse than M3/M8a.
- Decision: park naive gradual hardening as a standalone method. If revisited, it should be made contact-gated and delayed, or moved into a hard-candidate optimizer rather than freezing positions during early relaxed optimization.

## 1. Experiment Motivation

Earlier ACT-010 through ACT-018 runs showed a repeated pattern: relaxed trajectories can improve structure-facing objectives, but terminal argmax or naive sharpening often destroys the gain. ACT-014 specifically showed that post-update entropy annealing lowers entropy without improving candidate quality.

ACT-019 asks whether discretization can be made less destructive by committing positions gradually. The hypothesis was that high-confidence or low-conflict positions could be frozen before the final handoff, leaving uncertain interface-critical positions relaxed for further optimization.

This matters because the broader project needs a geometry-guided optimizer that produces usable hard sequences, not only better soft distributions.

## 2. Experiment Setup

| Item | Value |
|---|---|
| Run ID | `phase0_protenix_update_geometry_f9d49f8_20260507T114542Z` |
| Code commit | `f9d49f8` |
| Server | Quest `qgpu3019`, H100 80GB |
| Runtime | 16 min 56 sec, exit status 0 |
| Target | `IL7RA.cif` |
| Target length | 48 |
| Binder length | 24 |
| Seeds | 2 |
| Steps | 3 |
| Candidate top-k samples | 8 |
| Candidate sampling temperature | 0.25 |
| Compared methods | `M3,M7c,M8a,M8g,M9a,M9b,M9c` |

Primary artifacts:

- `docs/designs/phase0_act019_gradual_hardening.md`
- `docs/results/phase0_protenix_update_geometry_f9d49f8_20260507T114542Z.json`
- `docs/results/phase0_protenix_update_geometry_f9d49f8_20260507T114542Z_steps.csv`
- `docs/results/phase0_protenix_update_geometry_f9d49f8_20260507T114542Z_candidates.csv`
- `docs/results/phase0_protenix_update_geometry_f9d49f8_20260507T114542Z_topk_sensitivity.md`

## 3. Core Algorithm or Method

ACT-019 isolates position hardening by applying it on top of an `M3`-style naive weighted update, rather than stacking it with QP/contact-aggressive updates. This keeps the experimental question narrow: does gradual hardening itself help bridge soft-to-hard quality?

| Method | Rule | Threshold | Max frozen per step | Purpose |
|---|---|---:|---:|---|
| `M9a` | freeze positions whose top amino-acid probability is high | 0.42 | 20% | Test simple probability confidence. |
| `M9b` | freeze positions whose sampled cold candidates agree | 0.75 | 20% | Test sample-consensus confidence. |
| `M9c` | freeze high-margin, low-gradient-sensitivity positions | 0.12 | 20% | Test margin plus local sensitivity. |

Once a position is frozen, it is restored to a one-hot amino acid after subsequent relaxed updates.

## 4. Metrics

| Metric | Direction | Role |
|---|---|---|
| Harm rate | lower better | Whether local updates worsen any oracle term. |
| Primary derivative | more negative better | Whether the update descends the Protenix contact objective. |
| Entropy after update | lower means more discrete | Whether hardening actually commits positions. |
| Hardening fraction | diagnostic | Fraction of binder positions frozen during optimization. |
| BT PAE | lower better | Main hard-candidate interface error metric. |
| BT ipTM | higher better | Interface confidence metric. |
| Contact loss | lower better | Protenix contact objective on candidate sequences. |
| pLDDT | higher better | Structure confidence sanity check. |

The acceptance gate is stronger than winning one metric: a method should improve matched-budget hard candidate quality without exceeding M3 update harm.

## 5. Results

### 5.1 Update-Level Geometry

| Method | Harm rate | Primary derivative | Aux harm count | Worst aux derivative | Entropy after | New frozen / step | Final frozen frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| M7c | 0.1250 | -2.5559 | 0.5000 | 0.0062 | 2.3850 | 0.0000 | 0.0000 |
| M3 | 0.1667 | -5.0131 | 0.6667 | 0.0098 | 2.1931 | 0.0000 | 0.0000 |
| M9a | 0.2083 | -9.1911 | 0.8333 | 0.0104 | 1.9490 | 1.1667 | 0.1042 |
| M9b | 0.2500 | -9.8729 | 1.0000 | 0.0226 | 1.9704 | 1.0000 | 0.0833 |
| M8a | 0.2917 | -5.5514 | 1.1667 | 0.0181 | 2.1654 | 0.0000 | 0.0000 |
| M8g | 0.2917 | -5.5514 | 1.1667 | 0.0181 | 2.1654 | 0.0000 | 0.0000 |
| M9c | 0.3750 | -8.0953 | 1.5000 | 0.0138 | 1.7883 | 1.8333 | 0.1736 |

Answer for this subquestion:

- Goal: verify whether the hardening rules actually commit positions while keeping update safety near M3.
- Did we get the answer? Yes. The hardening rules are active and reduce entropy, but all three increase harm relative to M3.
- How did we find it? By logging per-step frozen-position counts, entropy, harm rate, primary contact derivatives, and auxiliary directional derivatives.
- Are the results correct? Internally yes: the run completed with exit status 0, and frozen-position diagnostics show the expected monotonic commitment behavior. The caveat is reduced scale.

### 5.2 Soft And Argmax Controls

| Method | Mode | pLDDT | BT PAE | BT ipTM | Contact loss | Trigram loss |
|---|---|---:|---:|---:|---:|---:|
| M3 | soft | 0.7264 | 8.8648 | 0.5203 | 2.2379 | 3.2177 |
| M8a | soft | 0.7232 | 8.9176 | 0.5123 | 2.1760 | 3.3094 |
| M8g | soft | 0.7232 | 8.9176 | 0.5123 | 2.1760 | 3.3094 |
| M9a | soft | 0.6550 | 11.8653 | 0.4342 | 2.5009 | 3.1514 |
| M9b | soft | 0.6415 | 11.9599 | 0.3836 | 2.5460 | 3.2242 |
| M7c | soft | 0.6097 | 12.0928 | 0.4546 | 2.4589 | 3.1408 |
| M9c | soft | 0.5577 | 13.5241 | 0.3101 | 2.8967 | 3.2674 |
| M8a | argmax | 0.5674 | 13.9343 | 0.4180 | 2.9447 | 3.3330 |
| M8g | argmax | 0.5674 | 13.9343 | 0.4180 | 2.9447 | 3.3330 |
| M7c | argmax | 0.5137 | 15.1793 | 0.3322 | 3.3261 | 3.1815 |
| M9b | argmax | 0.5077 | 15.7565 | 0.2706 | 3.1890 | 3.2454 |
| M9a | argmax | 0.4871 | 16.3863 | 0.2381 | 3.2720 | 3.1921 |
| M3 | argmax | 0.4826 | 16.4609 | 0.2255 | 3.2661 | 3.3044 |
| M9c | argmax | 0.4389 | 18.0034 | 0.1474 | 3.5877 | 3.3933 |

Answer for this subquestion:

- Goal: determine whether gradual hardening preserves relaxed quality and improves simple hard decoding.
- Did we get the answer? Yes. It does not preserve relaxed quality in this run. `M9a/M9b/M9c` all have worse soft BT PAE and BT ipTM than M3, and argmax remains poor.
- How did we find it? By scoring terminal soft and argmax candidates with the same Protenix candidate evaluation used for prior ACT runs.
- Are the results correct? The comparison is controlled by using the same target crop, seeds, steps, and scoring budget. It remains a reduced ProtenixMini smoke, not a final binder-design benchmark.

### 5.3 Matched-Budget Top-K Handoff

BT PAE reranking:

| Budget | Best method | M3 | M7c | M8a/M8g | M9a | M9b | M9c |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | M9b | 14.1572 | 16.1543 | 14.7522 | 13.8295 | 12.8929 | 17.3899 |
| 4 | M7c | 13.5439 | 11.5596 | 12.9376 | 13.8267 | 12.8902 | 15.1632 |
| 8 | M9b | 12.0435 | 11.4603 | 12.9376 | 12.2906 | 10.8833 | 14.4829 |

BT ipTM reranking:

| Budget | Best method | M3 | M7c | M8a/M8g | M9a | M9b | M9c |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | M8a/M8g | 0.3316 | 0.2002 | 0.3529 | 0.3118 | 0.3309 | 0.1306 |
| 4 | M7c | 0.4088 | 0.4869 | 0.4620 | 0.4002 | 0.4080 | 0.2816 |
| 8 | M7c | 0.4475 | 0.4869 | 0.4620 | 0.4496 | 0.4460 | 0.3265 |

Contact reranking:

| Budget | Best method | M3 | M7c | M8a/M8g | M9a | M9b | M9c |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | M9b | 2.9217 | 3.3885 | 3.0270 | 2.9259 | 2.8575 | 3.6134 |
| 4 | M7c | 2.8231 | 2.7551 | 2.8085 | 2.8972 | 2.8535 | 3.3652 |
| 8 | M3 | 2.6475 | 2.6948 | 2.7748 | 2.7216 | 2.6901 | 3.1365 |

Answer for this subquestion:

- Goal: test whether hardening helps under the cold top-k handoff regime that previously rescued some relaxed methods.
- Did we get the answer? Mostly yes. `M9b` gives the best BT PAE at budget 8, but it does not consistently win BT ipTM or contact, and it fails the update-harm gate.
- How did we find it? By reusing one candidate pool and recomputing best-per-seed top-k selection at budgets 1, 4, and 8 under BT PAE, BT ipTM, and contact reranking.
- Are the results correct? The matched-budget comparison is fair inside this run. The main uncertainty is that only 2 seeds and one target crop were used.

## 6. How to Read the Tables

The update table tests mechanism and safety: hardening should reduce entropy without increasing oracle harm. The soft/argmax table tests whether the terminal distribution itself is better. The top-k table tests whether a downstream cold sample-and-rerank handoff can rescue the terminal distribution under matched candidate scoring budget.

The important pattern is that lower entropy is not sufficient. `M9c` hardens the most and performs worst. `M9b` is the only useful hardening signal, but its positive result appears only at top-k budget 8 and only on BT PAE.

## 7. Interpretation

ACT-019 rejects the simple version of gradual hardening. The likely failure mode is premature commitment: the current confidence rules freeze residues before the structure oracle has a reliable discrete basin. Once frozen, those positions reduce the optimizer's ability to repair interface geometry.

This supports the earlier ACT-014 lesson. Making the distribution more discrete is not itself the objective. Discreteness only helps if the committed residues are already compatible with contact geometry and holdout structure metrics.

The useful part is diagnostic: `M9b` suggests consensus from cold samples may identify some better hard candidates, but it should probably be used for candidate selection or elite updates rather than irreversible early freezing.

## 8. Conclusion and Discussion

### What was the goal?

The goal was to test whether progressive position commitment can solve the soft-to-hard gap more safely than terminal argmax or naive entropy annealing.

### Did we get the answer finally?

Yes for the tested mechanism. Naive gradual hardening does not pass the Phase 0 gate.

### How did we find the answer?

We implemented three hardening rules, ran them against M3, M7c, and QP controls on the same reduced ProtenixMini IL7RA setup, then compared update safety, entropy/frozen-position diagnostics, soft/argmax scores, and matched-budget top-k handoff quality.

### Are the results correct?

The run completed with exit status 0, produced all expected CSV/JSON/Markdown artifacts, and the diagnostics show the method did what it was supposed to do mechanically. The conclusion is limited to this reduced setup and these thresholds.

## 9. Limitations and Caveats

- Reduced ProtenixMini setup only: 2 seeds, 3 steps, 24-residue binder, 48-residue target crop.
- No full-length IL7RA run, cross-oracle holdout, Boltz/Protenix agreement check, or final BindCraft-style filter pass-rate evaluation.
- The hardening thresholds are plausible but not exhaustively tuned.
- ACT-019 isolates hardening on top of M3-style updates; it does not test a delayed or contact-gated hardening rule stacked on a stronger trajectory.
- `M9b`'s budget-8 BT PAE win may be a sampling-budget artifact until repeated with more seeds.

## 10. Next Steps

1. Do not scale naive M9 hardening as-is.
2. Run a warm-started hard-candidate optimizer next: use the best relaxed/control distributions from M3/M7c/M8a as initialization, then optimize hard candidates by elite sampling or local mutation rather than freezing early.
3. If hardening is revisited, add gates: delay until contact loss is below a threshold, freeze only if hardening a position does not worsen contact on a local probe, and keep an unfreeze mechanism.
4. Keep `M9b` consensus as a candidate-selection feature or surrogate input, not as an irreversible update rule.

## Reproducibility Notes

Run command on Quest:

```bash
bash jobs/quest_protenix_act019_hardening_h100.slurm
```

Local validation before submission:

```bash
python3 -m py_compile scripts/run_update_geometry_diagnostic.py scripts/run_protenix_update_geometry_smoke.py scripts/summarize_topk_handoff_sensitivity.py
bash -n jobs/quest_protenix_act019_hardening_h100.slurm
git diff --check
```
