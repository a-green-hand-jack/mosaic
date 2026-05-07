# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_22cacaf_20260507T061105Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M7c | contact_preserving_soft_cone | 0.083 | 0.0056 | 0.5860 | 2.300 | 2.7127 | 0.0439 | 0.0004 | 3.1962 |
| M7d | contact_preserving_entropy_annealed | 0.167 | 0.0056 | 1.0758 | 1.331 | 2.9184 | 0.0353 | 0.0070 | 3.1565 |
| M3 | naive_weighted | 0.208 | 0.0109 | 0.8692 | 2.193 | 2.3890 | 0.0700 | 0.0027 | 3.2817 |
| M7e | contact_preserving_entropy_annealed | 0.250 | 0.0992 | 1.4224 | 0.526 | 3.0914 | 0.0526 | 0.0086 | 3.1682 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.6266 | 11.9026 | 0.3785 | 0.2578 | 2.7358 | 3.2200 |
| M3 | naive_weighted | argmax | 2 | 0.5754 | 12.9226 | 0.3755 | 0.1941 | 2.8778 | 3.2050 |
| M7e | contact_preserving_entropy_annealed | argmax | 2 | 0.5400 | 14.2376 | 0.3273 | 0.1962 | 2.9447 | 3.1756 |
| M3 | naive_weighted | topk_sample | 16 | 0.5417 | 14.6199 | 0.2986 | 0.1722 | 2.9920 | 3.2695 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5254 | 14.6406 | 0.2723 | 0.1189 | 3.1461 | 3.1236 |
| M7e | contact_preserving_entropy_annealed | soft | 2 | 0.5027 | 14.9388 | 0.2328 | 0.1781 | 3.0993 | 3.1570 |
| M7d | contact_preserving_entropy_annealed | soft | 2 | 0.5068 | 15.3178 | 0.2288 | 0.1848 | 3.0000 | 3.1271 |
| M7e | contact_preserving_entropy_annealed | topk_sample | 16 | 0.4950 | 15.4663 | 0.2411 | 0.1867 | 3.1445 | 3.1566 |
| M7c | contact_preserving_soft_cone | topk_sample | 16 | 0.4735 | 16.0297 | 0.2008 | 0.1251 | 3.2835 | 3.0827 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4692 | 16.0334 | 0.2053 | 0.1313 | 3.3688 | 3.1727 |
| M7d | contact_preserving_entropy_annealed | topk_sample | 16 | 0.4902 | 16.4356 | 0.2340 | 0.1594 | 3.2170 | 3.1719 |
| M7d | contact_preserving_entropy_annealed | argmax | 2 | 0.4733 | 16.8705 | 0.2171 | 0.1524 | 3.2047 | 3.2409 |

## Best-Per-Seed Candidate Scores

Selection metric: `bt_pae`

| Method ID | Method | Score mode | Selected | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | topk_sample | 2 | 0.6770 | 10.6019 | 0.4004 | 0.2058 | 2.5138 | 3.3661 |
| M3 | naive_weighted | soft | 2 | 0.6266 | 11.9026 | 0.3785 | 0.2578 | 2.7358 | 3.2200 |
| M7c | contact_preserving_soft_cone | topk_sample | 2 | 0.5751 | 12.1202 | 0.2766 | 0.1443 | 2.6815 | 3.0726 |
| M3 | naive_weighted | argmax | 2 | 0.5754 | 12.9226 | 0.3755 | 0.1941 | 2.8778 | 3.2050 |
| M7e | contact_preserving_entropy_annealed | topk_sample | 2 | 0.5613 | 13.5866 | 0.3439 | 0.1755 | 2.9515 | 3.1645 |
| M7e | contact_preserving_entropy_annealed | argmax | 2 | 0.5400 | 14.2376 | 0.3273 | 0.1962 | 2.9447 | 3.1756 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5254 | 14.6406 | 0.2723 | 0.1189 | 3.1461 | 3.1236 |
| M7e | contact_preserving_entropy_annealed | soft | 2 | 0.5027 | 14.9388 | 0.2328 | 0.1781 | 3.0993 | 3.1570 |
| M7d | contact_preserving_entropy_annealed | topk_sample | 2 | 0.5283 | 15.0673 | 0.2847 | 0.1418 | 3.0901 | 3.1350 |
| M7d | contact_preserving_entropy_annealed | soft | 2 | 0.5068 | 15.3178 | 0.2288 | 0.1848 | 3.0000 | 3.1271 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4692 | 16.0334 | 0.2053 | 0.1313 | 3.3688 | 3.1727 |
| M7d | contact_preserving_entropy_annealed | argmax | 2 | 0.4733 | 16.8705 | 0.2171 | 0.1524 | 3.2047 | 3.2409 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
