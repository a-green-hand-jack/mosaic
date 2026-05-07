# ACT-014 Entropy-Annealed Contact-Preserving Update

Date: 2026-05-07

## Question

ACT-013 showed that colder terminal top-k handoff can recover strong M7c discrete candidates. ACT-014 tested whether that signal can be moved inside the optimizer by sharpening the relaxed sequence after each contact-preserving update.

## Setup

- Commit: `22cacaf`
- Server: Quest H100 placeholder node `qgpu3019`
- Run ID: `phase0_protenix_update_geometry_22cacaf_20260507T061105Z`
- Runtime: 10:39 wall-clock, exit status 0
- Target: reduced IL7RA, target length 48, binder length 24
- Steps/seeds: `steps=3`, `num_seeds=2`
- Methods: `M3`, `M7c`, `M7d`, `M7e`
- Candidate handoff: `soft`, `argmax`, and `topk_sample`
- Top-k samples: 8 per method/seed
- Candidate sampling temperature: `0.25`

Raw artifacts:

- `docs/results/phase0_protenix_update_geometry_22cacaf_20260507T061105Z.json`
- `docs/results/phase0_protenix_update_geometry_22cacaf_20260507T061105Z_candidates.csv`
- `docs/results/phase0_protenix_update_geometry_22cacaf_20260507T061105Z_topk_sensitivity.csv`

## Method Variants

- `M7c`: contact-preserving soft cone, no entropy annealing.
- `M7d`: M7c direction plus post-update sharpening to final temperature `0.5`.
- `M7e`: M7c direction plus post-update sharpening to final temperature `0.25`.

The logged update-harm metrics use the actual post-sharpening update `x_new - x`.

## Update-Level Result

Entropy annealing made the terminal distributions colder, but it also worsened update safety and contact loss.

| Method | Harm rate | Worst derivative | Final entropy | Final contact loss | Step norm |
|---|---:|---:|---:|---:|---:|
| M7c | 0.0833 | 0.0056 | 2.2999 | 2.7127 | 0.5860 |
| M7d | 0.1667 | 0.0056 | 1.3308 | 2.9184 | 1.0758 |
| M3 | 0.2083 | 0.0109 | 2.1928 | 2.3890 | 0.8692 |
| M7e | 0.2500 | 0.0992 | 0.5259 | 3.0914 | 1.4224 |

`M7e` is too aggressive: it sharply lowers entropy but creates the worst update harm and worst contact loss among the tested methods.

## Candidate Result

BT PAE rerank, top-k candidates only:

| Budget | M3 BT PAE | M7c BT PAE | M7d BT PAE | M7e BT PAE | Winner |
|---:|---:|---:|---:|---:|---|
| 1 | 14.1454 | 17.1254 | 16.3435 | 14.0942 | M7e, marginal |
| 4 | 13.3245 | 12.1202 | 15.3398 | 13.8826 | M7c |
| 8 | 10.6019 | 12.1202 | 15.0673 | 13.5866 | M3 |

At the main budget-8 gate, `M3` is clearly strongest: BT PAE `10.6019`, BT ipTM `0.4004`, pLDDT `0.6770`, contact loss `2.5138`.

Budget 8 under alternative reranking metrics:

| Rerank metric | M3 BT PAE | M7c BT PAE | M7d BT PAE | M7e BT PAE | Winner |
|---|---:|---:|---:|---:|---|
| BT ipTM | 11.1887 | 15.3797 | 15.6885 | 13.8826 | M3 |
| contact | 10.6019 | 12.1202 | 15.3662 | 14.0942 | M3 |

## Interpretation

This is a negative result for the current optimizer-side entropy annealing implementation.

The result separates two mechanisms:

1. Cold terminal handoff can help after optimization, as ACT-013 showed.
2. Naively sharpening the relaxed sequence after every update is not the right way to get that benefit during optimization.

The most likely failure mode is that post-update sharpening changes the actual step geometry too abruptly. It lowers entropy but also increases step norm and harms auxiliary objectives, especially for `M7e`.

## Decision

Do not scale this entropy-annealed variant. Keep ACT-013's colder top-k handoff as an evaluation/handoff rule, but move the next optimizer revision toward a gentler mechanism:

- straight-through hard candidate scoring, or
- a weaker terminal discreteness regularizer with explicit contact-descent and harm constraints.

The acceptance gate remains unchanged: a revised geometry-aware method must beat M3 top-k under matched candidate budget while retaining update-safety advantage.
