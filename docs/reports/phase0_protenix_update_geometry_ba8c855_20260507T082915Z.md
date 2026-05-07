# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_ba8c855_20260507T082915Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | 0.167 | 0.0112 | 0.8843 | 2.199 | 2.4304 | 0.0668 | 0.0027 | 3.2787 |
| M7c | contact_preserving_soft_cone | 0.167 | 0.0093 | 0.7482 | 2.184 | 2.6208 | 0.0505 | 0.0004 | 3.2147 |
| M8b | contact_qp_grid | 0.167 | 0.0031 | 0.8758 | 2.170 | 2.6692 | 0.0934 | 0.0033 | 3.2984 |
| M8a | contact_qp_grid | 0.292 | 0.0199 | 0.9283 | 2.170 | 2.3646 | 0.0863 | 0.0025 | 3.3355 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M8a | contact_qp_grid | soft | 2 | 0.7824 | 7.0264 | 0.5880 | 0.2969 | 2.1255 | 3.3207 |
| M3 | naive_weighted | soft | 2 | 0.7378 | 8.7660 | 0.5183 | 0.2678 | 2.2628 | 3.2151 |
| M8b | contact_qp_grid | soft | 2 | 0.6846 | 10.0012 | 0.4598 | 0.2252 | 2.4894 | 3.2887 |
| M8a | contact_qp_grid | argmax | 2 | 0.5965 | 13.1877 | 0.4136 | 0.1947 | 2.7416 | 3.2922 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5565 | 13.5479 | 0.2855 | 0.1682 | 2.9122 | 3.1773 |
| M3 | naive_weighted | argmax | 2 | 0.5610 | 13.7054 | 0.3732 | 0.1697 | 2.8769 | 3.1307 |
| M3 | naive_weighted | topk_sample | 16 | 0.5692 | 13.7269 | 0.3517 | 0.1750 | 2.9210 | 3.2066 |
| M8b | contact_qp_grid | argmax | 2 | 0.5380 | 13.9792 | 0.3380 | 0.1838 | 2.9759 | 3.1842 |
| M8b | contact_qp_grid | topk_sample | 16 | 0.4991 | 15.1089 | 0.2155 | 0.1262 | 3.1631 | 3.2943 |
| M8a | contact_qp_grid | topk_sample | 16 | 0.5269 | 15.3293 | 0.3091 | 0.1515 | 3.1343 | 3.3343 |
| M7c | contact_preserving_soft_cone | topk_sample | 16 | 0.5036 | 15.5321 | 0.2476 | 0.1291 | 3.2510 | 3.0642 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4860 | 16.0428 | 0.2656 | 0.1591 | 3.2637 | 3.0185 |

## Best-Per-Seed Candidate Scores

Selection metric: `bt_pae`

| Method ID | Method | Score mode | Selected | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M8a | contact_qp_grid | soft | 2 | 0.7824 | 7.0264 | 0.5880 | 0.2969 | 2.1255 | 3.3207 |
| M3 | naive_weighted | soft | 2 | 0.7378 | 8.7660 | 0.5183 | 0.2678 | 2.2628 | 3.2151 |
| M3 | naive_weighted | topk_sample | 2 | 0.6985 | 9.3614 | 0.4409 | 0.2153 | 2.3917 | 3.1540 |
| M8b | contact_qp_grid | soft | 2 | 0.6846 | 10.0012 | 0.4598 | 0.2252 | 2.4894 | 3.2887 |
| M8a | contact_qp_grid | topk_sample | 2 | 0.6391 | 11.3780 | 0.4519 | 0.2195 | 2.6477 | 3.3463 |
| M7c | contact_preserving_soft_cone | topk_sample | 2 | 0.5872 | 12.1912 | 0.4153 | 0.1970 | 2.7613 | 2.9777 |
| M8b | contact_qp_grid | topk_sample | 2 | 0.5762 | 12.9887 | 0.3481 | 0.2001 | 2.7209 | 3.3703 |
| M8a | contact_qp_grid | argmax | 2 | 0.5965 | 13.1877 | 0.4136 | 0.1947 | 2.7416 | 3.2922 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5565 | 13.5479 | 0.2855 | 0.1682 | 2.9122 | 3.1773 |
| M3 | naive_weighted | argmax | 2 | 0.5610 | 13.7054 | 0.3732 | 0.1697 | 2.8769 | 3.1307 |
| M8b | contact_qp_grid | argmax | 2 | 0.5380 | 13.9792 | 0.3380 | 0.1838 | 2.9759 | 3.1842 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4860 | 16.0428 | 0.2656 | 0.1591 | 3.2637 | 3.0185 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
