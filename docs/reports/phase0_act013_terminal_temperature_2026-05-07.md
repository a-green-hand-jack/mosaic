# ACT-013 Terminal Temperature Sweep

Date: 2026-05-07

## Question

ACT-012 showed that top-k handoff helps, but M7c did not robustly beat M3 under temperature `1.0`. ACT-013 asks whether the failure is partly a terminal sampling-temperature problem: are we sampling too broadly from the relaxed sequence distribution?

## Setup

- Commit: `f89ee15`
- Server: Quest H100 placeholder node `qgpu3019`
- Target: reduced IL7RA, target length 48, binder length 24
- Steps/seeds: `steps=3`, `num_seeds=2`
- Methods: `M3`, `M7a`, `M7c`
- Candidate handoff: `argmax`, `soft`, and `topk_sample`
- Top-k samples: 8 per method/seed, top-k amino acids per residue = 4
- Temperatures: `0.5`, `0.25`
- Reference: ACT-012 used temperature `1.0`

Raw artifacts:

- `docs/results/phase0_protenix_update_geometry_f89ee15_20260507T054055Z*` for temperature `0.5`
- `docs/results/phase0_protenix_update_geometry_f89ee15_20260507T054922Z*` for temperature `0.25`

## Main Result

Lower terminal sampling temperature makes M7c competitive again under matched discrete top-k budget.

BT PAE rerank, top-k candidates only:

| Temperature | Budget | M3 BT PAE | M7c BT PAE | Winner | M3 BT ipTM | M7c BT ipTM |
|---:|---:|---:|---:|---|---:|---:|
| 0.5 | 1 | 14.6378 | 17.5181 | M3 | 0.2522 | 0.1387 |
| 0.5 | 4 | 12.7383 | 15.4502 | M3 | 0.3321 | 0.2413 |
| 0.5 | 8 | 12.7383 | 11.9860 | M7c | 0.3321 | 0.3822 |
| 0.25 | 1 | 15.2888 | 14.2063 | M7c | 0.3143 | 0.2945 |
| 0.25 | 4 | 15.1737 | 11.5974 | M7c | 0.2874 | 0.3627 |
| 0.25 | 8 | 13.9437 | 11.5974 | M7c | 0.3198 | 0.3627 |

At temperature `0.25`, M7c beats M3 top-k on BT PAE at every tested budget. The strongest top-k M7c point is BT PAE `11.5974`, BT ipTM `0.3627`, pLDDT `0.5908`.

## Rerank Robustness

Budget 8, top-k candidates only:

| Temperature | Rerank metric | M3 BT PAE | M7c BT PAE | M3 BT ipTM | M7c BT ipTM | M3 contact | M7c contact |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.5 | BT ipTM | 13.3704 | 11.9860 | 0.4395 | 0.3822 | 2.8933 | 2.8391 |
| 0.5 | contact | 12.7911 | 11.9860 | 0.3513 | 0.3822 | 2.8374 | 2.8391 |
| 0.25 | BT ipTM | 14.5211 | 11.8674 | 0.3460 | 0.4438 | 2.9939 | 2.9128 |
| 0.25 | contact | 13.9437 | 12.2823 | 0.3198 | 0.3969 | 2.9304 | 2.8855 |

Temperature `0.25` is the cleaner condition: M7c wins across BT PAE rerank, BT ipTM rerank, and contact rerank at budget 8.

## Important Caveat

This does not mean M7c has solved the whole problem. The soft terminal scores still favor M3 in these reduced runs:

- Temperature `0.5`: M3 soft BT PAE `9.2516`, M7c soft BT PAE `12.3838`
- Temperature `0.25`: M3 soft BT PAE `8.8039`, M7c soft BT PAE `12.1754`

So the current claim is narrower:

> Geometry-aware M7c can produce better discrete top-k candidates than M3 when the terminal handoff is sufficiently cold.

The result does not yet prove that the relaxed optimization objective itself is better than M3, and it does not yet prove robustness across more seeds or targets.

## Interpretation

ACT-013 resolves part of ACT-012's ambiguity. The negative ACT-012 result was not simply "M7c cannot produce good discrete candidates." Instead, the candidate quality depends strongly on the terminal handoff temperature.

This supports the next method revision:

1. Add colder terminal distributions or entropy annealing during optimization.
2. Keep the temperature/candidate budget matched against M3.
3. Use M3 top-k as the main reduced-run baseline, not M3 argmax.

## Next Decision

Do not scale to new targets yet. Implement a small optimization-side revision that makes the terminal distribution naturally colder, then rerun the same reduced gate:

- M3 vs revised M7c under matched top-k budget.
- Temperatures `0.25` and possibly `0.5`.
- Acceptance: revised geometry-aware method beats M3 top-k without relying on a larger candidate budget.
