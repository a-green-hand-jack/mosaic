# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_36ec419_20260506T201747Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normalized_weighted | 0.000 | -0.0090 | 0.4319 | 2.518 | 2.8563 | 0.0057 | 0.0071 | 3.0770 |
| soft_cone_correction | 0.000 | -0.0090 | 0.4319 | 2.518 | 2.8563 | 0.0057 | 0.0071 | 3.0770 |
| contact_preserving_soft_cone | 0.083 | 0.0005 | 0.5403 | 2.404 | 2.6861 | 0.0589 | 0.0005 | 3.2190 |
| naive_weighted | 0.167 | 0.0110 | 0.8721 | 2.145 | 2.4405 | 0.0666 | 0.0028 | 3.2797 |
| single_protenix_contact | 0.333 | 0.0301 | 1.0848 | 2.078 | 2.5358 | 0.1040 | 0.0031 | 3.3665 |

## Candidate Holdout Scores

| Method | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| single_protenix_contact | 2 | 0.5731 | 14.0659 | 0.4018 | 0.1783 | 2.9227 | 3.2146 |
| naive_weighted | 2 | 0.5420 | 14.2808 | 0.3136 | 0.1865 | 2.9382 | 3.1307 |
| contact_preserving_soft_cone | 2 | 0.4755 | 15.8992 | 0.2538 | 0.1273 | 3.2703 | 3.2232 |
| normalized_weighted | 2 | 0.4696 | 16.4343 | 0.1858 | 0.0870 | 3.5123 | 2.7696 |
| soft_cone_correction | 2 | 0.4696 | 16.4343 | 0.1858 | 0.0870 | 3.5123 | 2.7696 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
