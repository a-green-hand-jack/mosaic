# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_c9af27f_20260506T141236Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normalized_weighted | 0.000 | -0.0089 | 0.4312 | 2.517 | 2.9122 | 0.0055 | 0.0071 | 3.0765 |
| soft_cone_correction | 0.000 | -0.0089 | 0.4312 | 2.517 | 2.9122 | 0.0055 | 0.0071 | 3.0765 |
| naive_weighted | 0.208 | 0.0127 | 0.9539 | 2.114 | 2.4697 | 0.0746 | 0.0029 | 3.2852 |
| single_protenix_contact | 0.292 | 0.0311 | 1.3101 | 2.010 | 2.8578 | 0.0992 | 0.0032 | 3.3484 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
