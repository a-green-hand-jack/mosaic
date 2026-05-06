# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_3983a50_20260506T154811Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_weighted | 0.000 | -0.0007 | 1.0044 | 2.785 | 3.6358 | 0.1129 | 0.0010 | 3.3392 |
| normalized_weighted | 0.000 | -0.0062 | 0.4790 | 2.785 | 3.6358 | 0.1129 | 0.0010 | 3.3392 |
| soft_cone_correction | 0.000 | -0.0062 | 0.4790 | 2.785 | 3.6358 | 0.1129 | 0.0010 | 3.3392 |
| single_protenix_contact | 0.250 | 0.0497 | 1.0045 | 2.785 | 3.6358 | 0.1129 | 0.0010 | 3.3392 |

## Candidate Holdout Scores

| Method | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_weighted | 1 | 0.4315 | 16.6958 | 0.1814 | 0.1768 | 3.0497 | 3.4308 |
| single_protenix_contact | 1 | 0.4315 | 16.6958 | 0.1814 | 0.1768 | 3.0497 | 3.4308 |
| normalized_weighted | 1 | 0.4407 | 19.3010 | 0.0911 | 0.0000 | 3.6748 | 3.2864 |
| soft_cone_correction | 1 | 0.4407 | 19.3010 | 0.0911 | 0.0000 | 3.6748 | 3.2864 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
