# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_c9af27f_20260506T114159Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normalized_weighted | 0.000 | -0.0110 | 0.4825 | 2.739 | 3.6859 | 0.1158 | 0.0039 | 3.3278 |
| soft_cone_correction | 0.000 | -0.0110 | 0.4825 | 2.739 | 3.6859 | 0.1158 | 0.0039 | 3.3278 |
| naive_weighted | 0.250 | 0.0248 | 1.2501 | 2.739 | 3.6859 | 0.1158 | 0.0039 | 3.3278 |
| single_protenix_contact | 0.375 | 0.0677 | 1.2532 | 2.739 | 3.6859 | 0.1158 | 0.0039 | 3.3278 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
