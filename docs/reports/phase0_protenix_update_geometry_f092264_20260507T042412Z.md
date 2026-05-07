# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_f092264_20260507T042412Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M4 | normalized_weighted | 0.000 | -0.0087 | 0.4298 | 2.518 | 2.7916 | 0.0056 | 0.0067 | 3.0744 |
| M6 | soft_cone_correction | 0.000 | -0.0087 | 0.4298 | 2.518 | 2.7916 | 0.0056 | 0.0067 | 3.0744 |
| M7a | contact_preserving_soft_cone | 0.000 | -0.0046 | 0.5378 | 2.407 | 2.7701 | 0.0334 | 0.0053 | 3.1577 |
| M7c | contact_preserving_soft_cone | 0.125 | 0.0099 | 0.7046 | 2.267 | 2.7187 | 0.0569 | 0.0011 | 3.2117 |
| M7b | contact_preserving_soft_cone | 0.167 | 0.0009 | 0.7053 | 2.439 | 2.7165 | 0.0573 | 0.0001 | 3.2345 |
| M3 | naive_weighted | 0.208 | 0.0111 | 1.0041 | 2.135 | 2.5494 | 0.0687 | 0.0028 | 3.2791 |
| M1 | single_protenix_contact | 0.333 | 0.0295 | 1.2076 | 2.049 | 2.8228 | 0.0958 | 0.0032 | 3.3481 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M7c | contact_preserving_soft_cone | soft | 2 | 0.7266 | 8.7199 | 0.5104 | 0.2393 | 2.4296 | 3.1720 |
| M3 | naive_weighted | soft | 2 | 0.6989 | 9.5244 | 0.4834 | 0.2515 | 2.3242 | 3.1991 |
| M1 | single_protenix_contact | soft | 2 | 0.7054 | 10.1270 | 0.4274 | 0.2140 | 2.4100 | 3.3572 |
| M7a | contact_preserving_soft_cone | soft | 2 | 0.6435 | 10.4727 | 0.4373 | 0.2069 | 2.5198 | 3.0851 |
| M4 | normalized_weighted | soft | 2 | 0.6342 | 11.0390 | 0.3745 | 0.2294 | 2.5172 | 3.0196 |
| M6 | soft_cone_correction | soft | 2 | 0.6342 | 11.0390 | 0.3745 | 0.2294 | 2.5172 | 3.0196 |
| M7b | contact_preserving_soft_cone | soft | 2 | 0.5617 | 14.0963 | 0.3814 | 0.1760 | 2.9195 | 3.1439 |
| M1 | single_protenix_contact | argmax | 2 | 0.5186 | 14.8712 | 0.2268 | 0.1858 | 2.9805 | 3.3192 |
| M3 | naive_weighted | argmax | 2 | 0.5091 | 15.4820 | 0.2665 | 0.1775 | 3.1018 | 3.0568 |
| M7a | contact_preserving_soft_cone | argmax | 2 | 0.4437 | 16.4103 | 0.1959 | 0.1881 | 3.1935 | 2.9829 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4483 | 16.4140 | 0.1350 | 0.0679 | 3.3825 | 2.9832 |
| M4 | normalized_weighted | argmax | 2 | 0.4256 | 17.5593 | 0.1499 | 0.0816 | 3.5190 | 2.7464 |
| M6 | soft_cone_correction | argmax | 2 | 0.4256 | 17.5593 | 0.1499 | 0.0816 | 3.5190 | 2.7464 |
| M7b | contact_preserving_soft_cone | argmax | 2 | 0.4044 | 19.3324 | 0.0876 | 0.0000 | 3.7192 | 2.9972 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
