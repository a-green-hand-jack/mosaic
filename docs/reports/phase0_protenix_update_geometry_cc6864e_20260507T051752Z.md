# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_cc6864e_20260507T051752Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M4 | normalized_weighted | 0.000 | -0.0086 | 0.4296 | 2.518 | 2.7962 | 0.0055 | 0.0067 | 3.0752 |
| M7a | contact_preserving_soft_cone | 0.000 | -0.0047 | 0.5310 | 2.411 | 2.6736 | 0.0295 | 0.0053 | 3.1582 |
| M7c | contact_preserving_soft_cone | 0.167 | 0.0136 | 0.7136 | 2.285 | 2.6248 | 0.0588 | 0.0010 | 3.2224 |
| M3 | naive_weighted | 0.208 | 0.0102 | 0.8710 | 2.152 | 2.4398 | 0.0626 | 0.0029 | 3.2780 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.6553 | 10.3150 | 0.4110 | 0.2792 | 2.5400 | 3.2187 |
| M4 | normalized_weighted | soft | 2 | 0.6476 | 10.4867 | 0.3880 | 0.2264 | 2.4862 | 3.0198 |
| M7a | contact_preserving_soft_cone | soft | 2 | 0.6005 | 11.6437 | 0.3313 | 0.1931 | 2.6909 | 3.0915 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5878 | 12.3732 | 0.3078 | 0.1205 | 2.8503 | 3.1874 |
| M3 | naive_weighted | argmax | 2 | 0.5169 | 15.2533 | 0.2636 | 0.1462 | 3.1603 | 3.1661 |
| M7a | contact_preserving_soft_cone | argmax | 2 | 0.4677 | 15.5214 | 0.2082 | 0.1625 | 3.0077 | 3.0349 |
| M3 | naive_weighted | topk_sample | 16 | 0.5065 | 16.1417 | 0.2553 | 0.1389 | 3.2231 | 3.2671 |
| M7c | contact_preserving_soft_cone | topk_sample | 16 | 0.4480 | 16.9372 | 0.1489 | 0.0876 | 3.4865 | 3.1692 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4519 | 17.0432 | 0.1226 | 0.0714 | 3.4578 | 3.1161 |
| M7a | contact_preserving_soft_cone | topk_sample | 16 | 0.4525 | 17.1327 | 0.1675 | 0.0993 | 3.4344 | 3.0463 |
| M4 | normalized_weighted | argmax | 2 | 0.4251 | 17.5722 | 0.1484 | 0.0802 | 3.5223 | 2.7464 |
| M4 | normalized_weighted | topk_sample | 16 | 0.4473 | 18.2753 | 0.1507 | 0.0664 | 3.6355 | 2.8816 |

## Best-Per-Seed Candidate Scores

Selection metric: `bt_pae`

| Method ID | Method | Score mode | Selected | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.6553 | 10.3150 | 0.4110 | 0.2792 | 2.5400 | 3.2187 |
| M4 | normalized_weighted | soft | 2 | 0.6476 | 10.4867 | 0.3880 | 0.2264 | 2.4862 | 3.0198 |
| M7a | contact_preserving_soft_cone | soft | 2 | 0.6005 | 11.6437 | 0.3313 | 0.1931 | 2.6909 | 3.0915 |
| M3 | naive_weighted | topk_sample | 2 | 0.6329 | 11.8081 | 0.4425 | 0.2004 | 2.7006 | 3.1570 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5878 | 12.3732 | 0.3078 | 0.1205 | 2.8503 | 3.1874 |
| M7a | contact_preserving_soft_cone | topk_sample | 2 | 0.5404 | 14.1877 | 0.2909 | 0.1610 | 2.9404 | 3.1811 |
| M7c | contact_preserving_soft_cone | topk_sample | 2 | 0.4921 | 14.7162 | 0.2427 | 0.1503 | 3.1527 | 3.1196 |
| M3 | naive_weighted | argmax | 2 | 0.5169 | 15.2533 | 0.2636 | 0.1462 | 3.1603 | 3.1661 |
| M7a | contact_preserving_soft_cone | argmax | 2 | 0.4677 | 15.5214 | 0.2082 | 0.1625 | 3.0077 | 3.0349 |
| M4 | normalized_weighted | topk_sample | 2 | 0.4970 | 15.9600 | 0.2697 | 0.1543 | 3.2979 | 2.9302 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4519 | 17.0432 | 0.1226 | 0.0714 | 3.4578 | 3.1161 |
| M4 | normalized_weighted | argmax | 2 | 0.4251 | 17.5722 | 0.1484 | 0.0802 | 3.5223 | 2.7464 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
