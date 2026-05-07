# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_f89ee15_20260507T054055Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M7a | contact_preserving_soft_cone | 0.000 | -0.0043 | 0.5372 | 2.412 | 2.6667 | 0.0325 | 0.0054 | 3.1554 |
| M3 | naive_weighted | 0.167 | 0.0121 | 0.9601 | 2.114 | 2.5217 | 0.0709 | 0.0032 | 3.2772 |
| M7c | contact_preserving_soft_cone | 0.167 | 0.0097 | 0.8642 | 2.279 | 2.6138 | 0.0603 | 0.0012 | 3.2179 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.7121 | 9.2516 | 0.4768 | 0.2354 | 2.3568 | 3.2300 |
| M7a | contact_preserving_soft_cone | soft | 2 | 0.6655 | 9.8743 | 0.4511 | 0.2147 | 2.4897 | 3.1069 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5903 | 12.3838 | 0.3132 | 0.1852 | 2.7865 | 3.1889 |
| M3 | naive_weighted | argmax | 2 | 0.5383 | 14.3020 | 0.3482 | 0.1673 | 2.9194 | 3.2245 |
| M3 | naive_weighted | topk_sample | 16 | 0.5220 | 14.6275 | 0.2948 | 0.1665 | 3.0774 | 3.1988 |
| M7a | contact_preserving_soft_cone | argmax | 2 | 0.4640 | 15.7244 | 0.2532 | 0.1575 | 3.1017 | 2.9411 |
| M7a | contact_preserving_soft_cone | topk_sample | 16 | 0.4782 | 15.8409 | 0.2250 | 0.1410 | 3.2185 | 3.0352 |
| M7c | contact_preserving_soft_cone | topk_sample | 16 | 0.4630 | 16.6909 | 0.1740 | 0.0963 | 3.4250 | 3.1189 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4491 | 17.1274 | 0.1236 | 0.0674 | 3.4257 | 2.9787 |

## Best-Per-Seed Candidate Scores

Selection metric: `bt_pae`

| Method ID | Method | Score mode | Selected | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.7121 | 9.2516 | 0.4768 | 0.2354 | 2.3568 | 3.2300 |
| M7a | contact_preserving_soft_cone | soft | 2 | 0.6655 | 9.8743 | 0.4511 | 0.2147 | 2.4897 | 3.1069 |
| M7c | contact_preserving_soft_cone | topk_sample | 2 | 0.6043 | 11.9860 | 0.3822 | 0.1779 | 2.8391 | 3.0675 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5903 | 12.3838 | 0.3132 | 0.1852 | 2.7865 | 3.1889 |
| M3 | naive_weighted | topk_sample | 2 | 0.5692 | 12.7383 | 0.3321 | 0.1937 | 2.9221 | 3.3114 |
| M7a | contact_preserving_soft_cone | topk_sample | 2 | 0.5294 | 13.3883 | 0.2772 | 0.1863 | 2.9555 | 2.9412 |
| M3 | naive_weighted | argmax | 2 | 0.5383 | 14.3020 | 0.3482 | 0.1673 | 2.9194 | 3.2245 |
| M7a | contact_preserving_soft_cone | argmax | 2 | 0.4640 | 15.7244 | 0.2532 | 0.1575 | 3.1017 | 2.9411 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4491 | 17.1274 | 0.1236 | 0.0674 | 3.4257 | 2.9787 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
