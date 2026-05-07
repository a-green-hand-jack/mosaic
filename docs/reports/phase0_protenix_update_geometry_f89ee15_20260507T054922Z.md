# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_f89ee15_20260507T054922Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M7a | contact_preserving_soft_cone | 0.000 | -0.0045 | 0.5730 | 2.411 | 2.7609 | 0.0371 | 0.0058 | 3.1589 |
| M7c | contact_preserving_soft_cone | 0.083 | 0.0059 | 0.5851 | 2.360 | 2.6775 | 0.0448 | 0.0004 | 3.1960 |
| M3 | naive_weighted | 0.167 | 0.0107 | 0.8967 | 2.158 | 2.4185 | 0.0710 | 0.0028 | 3.2789 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.7300 | 8.8039 | 0.5043 | 0.2510 | 2.4163 | 3.2174 |
| M7a | contact_preserving_soft_cone | soft | 2 | 0.6843 | 9.9190 | 0.4735 | 0.2358 | 2.5749 | 3.1136 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5888 | 12.1754 | 0.3553 | 0.2095 | 2.6064 | 3.1223 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4924 | 15.2979 | 0.2354 | 0.1137 | 3.3585 | 3.2104 |
| M7c | contact_preserving_soft_cone | topk_sample | 16 | 0.5012 | 15.3392 | 0.2485 | 0.1185 | 3.2615 | 3.1426 |
| M3 | naive_weighted | topk_sample | 16 | 0.5027 | 15.9857 | 0.2643 | 0.1638 | 3.2103 | 3.3387 |
| M3 | naive_weighted | argmax | 2 | 0.4976 | 16.0961 | 0.2372 | 0.1234 | 3.2173 | 3.3406 |
| M7a | contact_preserving_soft_cone | argmax | 2 | 0.4638 | 16.5552 | 0.1747 | 0.1232 | 3.3827 | 3.1593 |
| M7a | contact_preserving_soft_cone | topk_sample | 16 | 0.4490 | 17.5332 | 0.1611 | 0.1300 | 3.4677 | 3.1524 |

## Best-Per-Seed Candidate Scores

Selection metric: `bt_pae`

| Method ID | Method | Score mode | Selected | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.7300 | 8.8039 | 0.5043 | 0.2510 | 2.4163 | 3.2174 |
| M7a | contact_preserving_soft_cone | soft | 2 | 0.6843 | 9.9190 | 0.4735 | 0.2358 | 2.5749 | 3.1136 |
| M7c | contact_preserving_soft_cone | topk_sample | 2 | 0.5908 | 11.5974 | 0.3627 | 0.1464 | 3.1651 | 3.1870 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5888 | 12.1754 | 0.3553 | 0.2095 | 2.6064 | 3.1223 |
| M3 | naive_weighted | topk_sample | 2 | 0.5466 | 13.9437 | 0.3198 | 0.2181 | 2.9304 | 3.2688 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4924 | 15.2979 | 0.2354 | 0.1137 | 3.3585 | 3.2104 |
| M7a | contact_preserving_soft_cone | topk_sample | 2 | 0.4937 | 15.6819 | 0.2797 | 0.2102 | 3.1842 | 3.1622 |
| M3 | naive_weighted | argmax | 2 | 0.4976 | 16.0961 | 0.2372 | 0.1234 | 3.2173 | 3.3406 |
| M7a | contact_preserving_soft_cone | argmax | 2 | 0.4638 | 16.5552 | 0.1747 | 0.1232 | 3.3827 | 3.1593 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
