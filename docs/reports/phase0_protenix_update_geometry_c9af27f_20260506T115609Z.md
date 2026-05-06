# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_c9af27f_20260506T115609Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normalized_weighted | 0.000 | -0.0090 | 0.4345 | 2.517 | 2.9111 | 0.0057 | 0.0070 | 3.0762 |
| soft_cone_correction | 0.000 | -0.0090 | 0.4345 | 2.517 | 2.9111 | 0.0057 | 0.0070 | 3.0762 |
| naive_weighted | 0.208 | 0.0108 | 0.8964 | 2.148 | 2.4104 | 0.0700 | 0.0028 | 3.2799 |
| single_protenix_contact | 0.333 | 0.0317 | 1.0546 | 2.060 | 2.6333 | 0.0958 | 0.0032 | 3.3487 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
