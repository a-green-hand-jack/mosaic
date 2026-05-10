# ACT-021 Boltz2 Cross-Oracle Agreement Report

## Summary

- Question: Do ACT-021 Protenix candidate scores agree with Boltz2 finite second-oracle scores enough to guide candidate selection?
- Setup: 24 deduplicated ACT-021 candidates, balanced as M3/M7c/M8a = 8/8/8, scored with Boltz2 finite distogram, pLDDT, and PAE outputs.
- Headline result: all 24 candidates produced finite Boltz2 distogram, pLDDT, and PAE outputs; all still had non-finite structure coordinates.
- Interpretation: Protenix bt_PAE and Boltz2 inter-PAE show moderate agreement, but the overlap is not strong enough to rely on Protenix alone.
- Next step: keep the Boltz2 second-oracle reranker for shortlists, and expand to WCEM/DWCEM only after summarizing whether those families add top-ranked Boltz2 candidates.

## 1. Experiment Motivation

The Phase 0 question is whether candidate sequences that look good under the Protenix oracle also remain promising under an independent structure-model oracle. This matters because the update-geometry claim should not be validated only against the oracle that generated the candidates. Boltz2 is currently usable as a reduced second oracle through finite PAE, pLDDT, and distogram/contact scores, while coordinate-level losses remain blocked by non-finite structure coordinates.

## 2. Experiment Setup

| Item | Value |
|---|---|
| Source candidates | `docs/results/phase0_protenix_cem_23ff9f1_20260507T232223Z_candidates.csv` |
| Selection | 24 deduplicated candidates, balanced round-robin by `method_id` |
| Methods | M3, M7c, M8a; 8 candidates per method |
| Target | `IL7RA.cif:A`, truncated to 48 residues |
| Binder length | 24 residues |
| Boltz2 settings | 1 recycling step, 1 sampling step, empty MSA |
| Run ID | `boltz2_candidate_holdout_act021_m3_m7c_m8a_24_20260510T134114Z` |
| Log | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/manual-boltz2-candidate-holdout-qgpu2013-20260510T134014Z.out` |
| Result CSV | `docs/results/boltz2_candidate_holdout_act021_m3_m7c_m8a_24_20260510T134114Z.csv` |
| Result report | `docs/reports/boltz2_candidate_holdout_act021_m3_m7c_m8a_24_20260510T134114Z.md` |
| Run commit | `6ac36a7` |
| Result artifact commit | `2e8bf32` |
| Hardware | Quest `qgpu2013`, A100-SXM4-80GB, GPU 0 |

## 3. Core Algorithm or Method

The analysis is a post-hoc reranking experiment. The scorer reads a candidate CSV, filters by method and score mode, deduplicates sequences, balances candidate selection across methods, and runs each sequence through the same Boltz2 feature template. Each candidate reuses the same binder-plus-target features with only the binder sequence replaced.

This is not yet a differentiable Boltz2 training loss. It is a finite-output second-oracle evaluator for candidate holdout selection.

## 4. Metrics

| Metric | Definition | Direction | Why it matters |
|---|---|---:|---|
| Boltz2 inter-PAE | Mean PAE from binder tokens to target tokens | Lower | Main second-oracle interface confidence proxy |
| Boltz2 contact@12A | Mean probability of binder-target residue pairs below 12A from distogram bins | Higher | Interface proximity proxy independent of Protenix |
| Boltz2 pLDDT | Mean confidence over all tokens | Higher | Global confidence sanity check |
| Protenix bt_PAE | Protenix binder-to-target PAE candidate metric | Lower | Main Protenix interface confidence comparator |
| Protenix contact loss | `candidate_loss_protenix_contact` | Lower | Protenix contact objective comparator |
| Protenix pLDDT | Protenix candidate pLDDT metric | Higher | Protenix confidence comparator |

## 5. Results

### 5.1 Method-Level Summary

| Method | N | Mean Boltz2 inter-PAE ↓ | Mean Boltz2 contact@12A ↑ | Mean Boltz2 pLDDT ↑ | Mean Protenix bt_PAE ↓ | Mean Protenix contact loss ↓ | Mean Protenix pLDDT ↑ | Mean Boltz2 rank ↓ | Boltz2 Top-8 count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M3 | 8 | 18.770 | 0.2359 | 0.4614 | 13.464 | 2.895 | 0.5397 | 11.38 | 2 |
| M7c | 8 | 19.745 | 0.2164 | 0.4716 | 15.143 | 3.207 | 0.5185 | 13.12 | 3 |
| M8a | 8 | 19.979 | 0.2187 | 0.4758 | 16.358 | 3.227 | 0.4827 | 13.00 | 3 |

M3 has the best average Boltz2 inter-PAE and contact@12A, and also the best average Protenix bt_PAE. However, Boltz2's Top-8 list is not dominated by M3; M7c and M8a each contribute three candidates.

### 5.2 Top Boltz2 Candidates

| Boltz2 rank | Method | Mode | Sequence | Boltz2 inter-PAE ↓ | Boltz2 contact@12A ↑ | Protenix bt_PAE ↓ | Protenix contact loss ↓ |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | M3 | topk_sample | `RFKIRATSKVNSPEGTFITISFWK` | 13.9913 | 0.2263 | 13.6047 | 2.8539 |
| 2 | M7c | topk_sample | `LRWPRPHYKPDSVNGNFITIPFWP` | 15.4656 | 0.2258 | 14.3528 | 3.1153 |
| 3 | M3 | topk_sample | `RISFRANYKVNSDDGQFITIWFPD` | 15.7529 | 0.2345 | 12.3593 | 2.7942 |
| 4 | M7c | topk_sample | `LRWPWPDIKPDGVNGTFITYSWWP` | 17.3908 | 0.2195 | 12.5055 | 2.8610 |
| 5 | M8a | topk_sample | `LRWPWPHYKPDSDNGNMITYDFPD` | 18.1562 | 0.2147 | 13.3733 | 2.8963 |
| 6 | M8a | soft | `LRWPRPHYKPDGDNGNFITYPFWD` | 18.6866 | 0.2188 | 13.2536 | 2.5710 |
| 7 | M7c | topk_sample | `LRFPRGNVKPDGDDGNFIEYDYWP` | 18.7359 | 0.2293 | 14.7898 | 3.3833 |
| 8 | M8a | topk_sample | `LRFPMPMYKPDADNGNFIEYPYWD` | 18.7418 | 0.2205 | 19.3975 | 3.7461 |

### 5.3 Cross-Oracle Agreement

| Comparison | Raw Pearson | Best-rank Spearman | Interpretation |
|---|---:|---:|---|
| Protenix bt_PAE ↓ vs Boltz2 inter-PAE ↓ | 0.3963 | 0.5096 | Moderate agreement; good enough for shortlist triage, not enough to skip Boltz2 reranking |
| Protenix contact loss ↓ vs Boltz2 contact@12A ↑ | -0.3333 | 0.3130 | Weak-to-moderate agreement after accounting for opposite directions |
| Protenix pLDDT ↑ vs Boltz2 pLDDT ↑ | -0.0592 | 0.0270 | No useful agreement in this small sample |

| K | Top-K overlap: Boltz2 inter-PAE vs Protenix bt_PAE | Top-K overlap: Boltz2 inter-PAE vs Protenix contact loss |
|---:|---:|---:|
| 3 | 1 | 0 |
| 5 | 3 | 2 |
| 8 | 5 | 5 |
| 10 | 6 | 6 |
| 12 | 9 | 8 |

## 6. How to Read the Tables

The method-level table describes aggregate behavior, not final candidate selection. Lower Boltz2 inter-PAE is the primary second-oracle ranking signal. Boltz2 contact@12A and pLDDT are sanity checks, not strict replacement objectives.

The Top-K overlap table asks whether Protenix-only selection would recover the same candidates that Boltz2 prefers. The Top-8 overlap of 5/8 suggests Protenix is directionally useful, but Boltz2 changes the shortlist enough to matter.

## 7. Interpretation

The second-oracle result supports keeping Boltz2 as a holdout reranker. M3 remains strongest on average, but the top Boltz2 shortlist includes candidates from M7c and M8a. This weakens a simple "M3 only" decision and suggests that update-geometry variants may produce useful candidates even when their average Protenix metrics are worse.

The agreement signal is strongest for interface PAE: Protenix bt_PAE and Boltz2 inter-PAE have moderate positive correlation and a 5/8 Top-8 overlap. Contact agreement is weaker. pLDDT agreement is effectively absent, so pLDDT should not drive cross-oracle selection at this stage.

## 8. Conclusion and Discussion

Boltz2 finite-output scoring is now useful enough for candidate holdout decisions. The current best candidate is:

```text
RFKIRATSKVNSPEGTFITISFWK
```

This is an M3 top-k sampled candidate with Boltz2 inter-PAE `13.9913`, Boltz2 contact@12A `0.2263`, and Protenix bt_PAE `13.6047`.

For method development, the more important conclusion is that Boltz2 does not simply reproduce the Protenix ordering. It preserves some Protenix signal but changes the top shortlist. This is the behavior we want from a second oracle.

## 9. Limitations and Caveats

- Sample size is 24 candidates, so method-level conclusions are provisional.
- The selection covers M3/M7c/M8a only; WCEM/DWCEM candidates are not included in this report.
- All candidates used empty MSA and a truncated 48-residue IL7RA target for reduced runtime.
- Boltz2 `structure_coordinates` remained non-finite for all candidates, so coordinate-level losses are still blocked.
- The result is a post-hoc scoring experiment, not a differentiable optimization run with Boltz2 in the loop.

## 10. Next Steps

1. Extend the same balanced Boltz2 scorer to WCEM/DWCEM candidates if the goal is to test whether CEM variants add better cross-oracle candidates.
2. Build a compact candidate shortlist that includes the best M3, M7c, and M8a sequences rather than selecting only by method average.
3. Diagnose non-finite Boltz2 structure coordinates before using coordinate-level losses or writing claims about generated structures.
4. If compute budget allows, rerun the top shortlist with more sampling steps or a less truncated target to check whether the ranking is stable.

## Reproducibility Notes

The analysis used `docs/results/boltz2_candidate_holdout_act021_m3_m7c_m8a_24_20260510T134114Z.csv` in the baseline worktree. The source scoring run was launched via `jobs/launch_boltz2_candidate_holdout_act021_24.sh` and logged to `/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/manual-boltz2-candidate-holdout-qgpu2013-20260510T134014Z.out`.
