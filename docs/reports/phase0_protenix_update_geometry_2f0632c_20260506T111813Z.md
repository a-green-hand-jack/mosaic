# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_2f0632c_20260506T111813Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive_weighted | 0.000 | -0.0007 | 0.9342 | 2.785 | 3.6313 | 0.1129 | 0.0010 | 3.3392 |
| normalized_weighted | 0.000 | -0.0062 | 0.4799 | 2.785 | 3.6313 | 0.1129 | 0.0010 | 3.3392 |
| soft_cone_correction | 0.000 | -0.0062 | 0.4799 | 2.785 | 3.6313 | 0.1129 | 0.0010 | 3.3392 |
| single_protenix_contact | 0.250 | 0.0449 | 0.9315 | 2.785 | 3.6313 | 0.1129 | 0.0010 | 3.3392 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
