# Phase 0 ACT-020 Warm-Started CEM Report

## Summary

- Goal: test whether hard-candidate optimization works better when warm-started from useful relaxed terminal distributions, instead of random CEM or irreversible early hardening.
- Setup: reduced IL7RA ProtenixMini smoke on Quest H100, 2 seeds, 3 relaxed update steps, 24-residue binder, 48-residue target crop, and 24 hard-candidate scores per method/seed.
- Main answer: partially yes. Warm CEM strongly improves final argmax quality, especially `WCEMp_M7c`, but it does not beat the best source top-k baselines under the matched 24-candidate budget.
- Best warm-started signal: `WCEMp_M7c` final argmax reaches BT PAE 9.8045, versus M7c argmax 13.1110 and M3 argmax 15.6512.
- Decision: keep warm-started hard-candidate optimization as useful, but treat it as a candidate-optimizer/handoff mechanism. Do not claim it beats matched-budget M7c/M8a top-k yet.

## 1. Experiment Motivation

ACT-015A showed that random CEM collapses entropy around poor hard candidates. ACT-019 showed that gradual early hardening also lowers entropy but commits residues too early and worsens most metrics.

ACT-020 tests a narrower hypothesis: the problem may not be CEM itself, but where it starts. If relaxed M3/M7c/M8a trajectories already find useful basins, then hard-candidate elite updates from those terminal distributions may bridge the soft-to-hard gap without freezing positions during early relaxed optimization.

This matters because the project needs a way to turn relaxed geometry gains into usable discrete sequences.

## 2. Experiment Setup

| Item | Value |
|---|---|
| Run ID | `phase0_protenix_cem_2dd3965_20260507T120914Z` |
| Code commit | `2dd3965` |
| Server | Quest `qgpu3019`, H100 80GB |
| Runtime | 8 min 17 sec, exit status 0 |
| Target | `IL7RA.cif` |
| Target length | 48 |
| Binder length | 24 |
| Seeds | 2 |
| Relaxed update steps | 3 |
| Baselines | `M3,M7c,M8a` |
| Warm CEM variants | `WCEMp,WCEMc` from each baseline terminal |
| CEM rounds | 3 |
| Samples per round | 8 |
| Elite count | 2 |
| Candidate sample temperature | 0.25 |

Primary artifacts:

- `docs/designs/phase0_act020_warm_cem.md`
- `docs/results/phase0_protenix_cem_2dd3965_20260507T120914Z.json`
- `docs/results/phase0_protenix_cem_2dd3965_20260507T120914Z_steps.csv`
- `docs/results/phase0_protenix_cem_2dd3965_20260507T120914Z_candidates.csv`
- `docs/results/phase0_protenix_cem_2dd3965_20260507T120914Z_cem_rounds.csv`
- `docs/results/phase0_protenix_cem_2dd3965_20260507T120914Z_topk_sensitivity.md`

## 3. Core Algorithm or Method

ACT-020 keeps baseline relaxed trajectories unchanged, then starts CEM from their terminal distributions:

1. Run a baseline optimizer (`M3`, `M7c`, or `M8a`) for 3 relaxed steps.
2. Mix the terminal distribution with 2% uniform mass.
3. Sample 8 hard top-k candidates per round at temperature 0.25.
4. Score candidates with ProtenixMini candidate metrics.
5. Select 2 elites by either BT PAE (`WCEMp`) or contact loss (`WCEMc`).
6. Move the distribution 60% toward the elite distribution with a 5% uniform floor.

The experiment compares warm-started CEM against source baselines with the same 24 hard-candidate scoring budget.

## 4. Metrics

| Metric | Direction | Role |
|---|---|---|
| BT PAE | lower better | Main interface error metric. |
| BT ipTM | higher better | Interface confidence metric. |
| Contact loss | lower better | Protenix contact objective. |
| pLDDT | higher better | Structure confidence sanity check. |
| Entropy | lower means more discrete | Whether CEM commits to a hard distribution. |
| Update harm | lower better | Safety of the source relaxed trajectory. Warm CEM itself is a post-hoc candidate optimizer. |

## 5. Results

### 5.1 Source Update Geometry

| Method | Harm rate | Primary derivative | Aux harm count | Worst aux derivative | Entropy after |
|---|---:|---:|---:|---:|---:|
| M7c | 0.1250 | -2.6729 | 0.5000 | 0.0094 | 2.3404 |
| M3 | 0.2500 | -5.3473 | 1.0000 | 0.0109 | 2.1875 |
| M8a | 0.2917 | -5.6065 | 1.1667 | 0.0203 | 2.1577 |

Answer for this subquestion:

- Goal: establish the safety context of the warm-start sources.
- Did we get the answer? Yes. `M7c` remains the safest source update; `M8a` has the strongest soft candidate but higher harm.
- How did we find it? By reusing the per-step update diagnostics logged by the relaxed baseline trajectories.
- Are the results correct? Internally yes; these are the same reduced ProtenixMini diagnostics used in earlier ACT runs.

### 5.2 Final Argmax Repair

| Method | pLDDT | BT PAE | BT ipTM | Contact loss | Trigram loss |
|---|---:|---:|---:|---:|---:|
| WCEMp_M7c | 0.6494 | 9.8045 | 0.4257 | 2.6458 | 3.1632 |
| WCEMp_M3 | 0.6302 | 10.8194 | 0.4065 | 2.7167 | 3.1996 |
| WCEMp_M8a | 0.6492 | 11.2032 | 0.4561 | 2.6186 | 3.2486 |
| WCEMc_M8a | 0.6191 | 11.6233 | 0.4095 | 2.6801 | 3.3384 |
| WCEMc_M3 | 0.6048 | 11.8462 | 0.3928 | 2.6973 | 3.1843 |
| M7c argmax | 0.5483 | 13.1110 | 0.3096 | 2.7968 | 3.0068 |
| M8a argmax | 0.5862 | 13.1562 | 0.3967 | 2.8396 | 3.2255 |
| WCEMc_M7c | 0.5401 | 13.7360 | 0.2975 | 2.9256 | 3.2172 |
| M3 argmax | 0.5085 | 15.6512 | 0.2595 | 3.1694 | 3.3941 |

Answer for this subquestion:

- Goal: determine whether warm CEM can repair terminal argmax collapse.
- Did we get the answer? Yes. The BT PAE-ranked warm CEM variants strongly improve argmax quality, especially from M7c.
- How did we find it? By comparing final CEM argmax candidates to the source baseline argmax candidates.
- Are the results correct? The improvement is directly measured in the candidate CSV. It is a real argmax-repair signal, but not yet a matched-budget top-k win.

### 5.3 Matched-Budget Top-K

BT PAE reranking:

| Budget | Best method | M3 | M7c | M8a | Best warm CEM |
|---:|---|---:|---:|---:|---:|
| 1 | M8a soft/control | 15.8845 | 14.1444 | 12.3055 | 16.5939 |
| 8 | M7c | 12.7517 | 8.8929 | 10.7932 | 11.2420 |
| 24 | M8a | 10.9494 | 8.8929 | 8.6586 | 9.1575 |

BT ipTM reranking:

| Budget | Best method | M3 | M7c | M8a | Best warm CEM |
|---:|---|---:|---:|---:|---:|
| 8 | M8a | 0.3584 | 0.4482 | 0.5023 | 0.4364 |
| 24 | WCEMc_M8a | 0.4673 | 0.4482 | 0.5173 | 0.5217 |

Contact reranking:

| Budget | Best method | M3 | M7c | M8a | Best warm CEM |
|---:|---|---:|---:|---:|---:|
| 8 | M7c | 2.8015 | 2.4213 | 2.5637 | 2.5516 |
| 24 | M8a | 2.6573 | 2.4213 | 2.3925 | 2.4055 |

Answer for this subquestion:

- Goal: decide whether warm CEM beats the source top-k baselines under the same hard-candidate scoring budget.
- Did we get the answer? Yes. It does not beat the strongest source top-k baseline at budget 24. `M8a` and `M7c` remain stronger on BT PAE/contact; `WCEMc_M8a` barely wins BT ipTM but with worse BT PAE/contact than M8a.
- How did we find it? By reusing one candidate pool and recomputing best-per-seed top-k selection at budgets 1, 8, and 24 under BT PAE, BT ipTM, and contact reranking.
- Are the results correct? The comparison is budget-matched inside this run. The caveat is that warm CEM's best final argmax should not be confused with winning the full top-k budget comparison.

## 6. How to Read the Tables

The argmax table tests whether CEM produces a single usable terminal sequence. The top-k table tests whether CEM is better than simply sampling and reranking from the original source distribution with the same candidate budget.

The key distinction is important: ACT-020 is positive for final argmax repair, but not positive for matched-budget top-k superiority.

## 7. Interpretation

Warm-starting fixes the main failure of random CEM. ACT-015A collapsed around poor candidates from a random distribution, while ACT-020 can recover discrete candidates close to the relaxed source quality. The strongest example is `WCEMp_M7c`, which nearly matches M7c soft BT PAE while greatly improving over M7c argmax.

However, the experiment also shows that CEM is not yet better than cold top-k handoff when both get the same candidate scoring budget. The source distribution already contains good samples; elite updates may over-commit and reduce diversity, so the best-of-24 source top-k remains stronger.

## 8. Conclusion and Discussion

### What was the goal?

The goal was to test whether hard-candidate optimization becomes useful when warm-started from a good relaxed terminal distribution.

### Did we get the answer finally?

Yes. Warm-starting makes CEM useful for argmax repair, but not yet for matched-budget top-k improvement.

### How did we find the answer?

We implemented `WCEMp/WCEMc`, initialized them from M3/M7c/M8a terminal distributions, ran the reduced ProtenixMini gate, then compared final argmax candidates and best-of-budget top-k candidates against the source baselines.

### Are the results correct?

The run completed with exit status 0 and produced all expected JSON, CSV, and Markdown artifacts. The conclusion is internally consistent: argmax rows show a strong repair signal, while top-k sensitivity rows show source top-k still wins.

## 9. Limitations and Caveats

- Reduced ProtenixMini setup only: 2 seeds, 3 relaxed steps, 24-residue binder, 48-residue target crop.
- Warm CEM uses hard-candidate oracle calls; it should be framed as a candidate optimizer or handoff stage unless integrated with update safety.
- The elite update may be too aggressive and reduce diversity. Only one update rate, elite count, and uniform floor were tested.
- No cross-oracle holdout, full-length target, calibrated filter pass rate, or multi-target replication was run.

## 10. Next Steps

1. Test a diversity-preserving warm CEM variant: lower update rate, keep a source-distribution mixture, or select elites from a Pareto score instead of one metric.
2. Pre-register whether the next gate is final argmax repair or matched-budget best-of-K quality. These are different claims.
3. If the goal is a single deployable sequence, scale `WCEMp_M7c` cautiously. If the goal is best-of-budget quality, improve diversity before scale-up.

## Reproducibility Notes

Run command on Quest:

```bash
bash jobs/quest_protenix_act020_warm_cem_h100.slurm
```

Local validation before submission:

```bash
python3 -m py_compile scripts/run_protenix_cem_optimizer.py scripts/run_protenix_update_geometry_smoke.py scripts/run_update_geometry_diagnostic.py
bash -n jobs/quest_protenix_act020_warm_cem_h100.slurm
git diff --check
```
