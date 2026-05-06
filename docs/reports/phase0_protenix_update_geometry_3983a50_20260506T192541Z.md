# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_3983a50_20260506T192541Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normalized_weighted | 0.000 | -0.0090 | 0.4315 | 2.517 | 2.8836 | 0.0056 | 0.0070 | 3.0765 |
| soft_cone_correction | 0.000 | -0.0090 | 0.4315 | 2.517 | 2.8836 | 0.0056 | 0.0070 | 3.0765 |
| naive_weighted | 0.292 | 0.0112 | 0.8950 | 2.159 | 2.4012 | 0.0740 | 0.0028 | 3.2829 |
| single_protenix_contact | 0.417 | 0.0317 | 1.0152 | 2.066 | 2.6051 | 0.1010 | 0.0030 | 3.3508 |

## Candidate Holdout Scores

| Method | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| single_protenix_contact | 2 | 0.5147 | 16.1696 | 0.2892 | 0.2030 | 3.0773 | 3.3319 |
| naive_weighted | 2 | 0.5030 | 16.1962 | 0.2563 | 0.1622 | 3.1925 | 3.3537 |
| normalized_weighted | 2 | 0.4381 | 18.1139 | 0.1347 | 0.1327 | 3.6494 | 2.7761 |
| soft_cone_correction | 2 | 0.4381 | 18.1139 | 0.1347 | 0.1327 | 3.6494 | 2.7761 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
