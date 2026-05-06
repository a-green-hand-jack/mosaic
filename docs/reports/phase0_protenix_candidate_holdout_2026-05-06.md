# Phase 0 Protenix Candidate Holdout

Date: 2026-05-06

## Scope

This note adds candidate-level holdout scoring to the Protenix update-geometry diagnostics. The terminal relaxed sequence from each method is converted to an argmax discrete binder sequence, then scored with ProtenixMini-derived structure metrics and sequence sanity metrics.

Runs:

- Minimum candidate smoke: `phase0_protenix_update_geometry_3983a50_20260506T154811Z`, `steps=1`, `num_seeds=1`.
- Small candidate run: `phase0_protenix_update_geometry_3983a50_20260506T192541Z`, `steps=3`, `num_seeds=2`.

## Update Geometry

| Run | Method | Harm rate | Worst derivative |
|---|---|---:|---:|
| `steps=1,seeds=1` | single_protenix_contact | 0.250 | 0.0497 |
| `steps=1,seeds=1` | naive_weighted | 0.000 | -0.0007 |
| `steps=1,seeds=1` | normalized_weighted | 0.000 | -0.0062 |
| `steps=1,seeds=1` | soft_cone_correction | 0.000 | -0.0062 |
| `steps=3,seeds=2` | single_protenix_contact | 0.417 | 0.0317 |
| `steps=3,seeds=2` | naive_weighted | 0.292 | 0.0112 |
| `steps=3,seeds=2` | normalized_weighted | 0.000 | -0.0090 |
| `steps=3,seeds=2` | soft_cone_correction | 0.000 | -0.0090 |

## Candidate Holdout

| Run | Method | pLDDT | BT PAE | BT ipTM | IPSAE min | Trigram loss | Hydrophobic mean |
|---|---|---:|---:|---:|---:|---:|---:|
| `steps=1,seeds=1` | single_protenix_contact | 0.431 | 16.696 | 0.181 | 0.177 | 3.431 | 0.375 |
| `steps=1,seeds=1` | naive_weighted | 0.431 | 16.696 | 0.181 | 0.177 | 3.431 | 0.375 |
| `steps=1,seeds=1` | normalized_weighted | 0.441 | 19.301 | 0.091 | 0.000 | 3.286 | 0.292 |
| `steps=1,seeds=1` | soft_cone_correction | 0.441 | 19.301 | 0.091 | 0.000 | 3.286 | 0.292 |
| `steps=3,seeds=2` | single_protenix_contact | 0.515 | 16.170 | 0.289 | 0.203 | 3.332 | 0.396 |
| `steps=3,seeds=2` | naive_weighted | 0.503 | 16.196 | 0.256 | 0.162 | 3.354 | 0.333 |
| `steps=3,seeds=2` | normalized_weighted | 0.438 | 18.114 | 0.135 | 0.133 | 2.776 | 0.208 |
| `steps=3,seeds=2` | soft_cone_correction | 0.438 | 18.114 | 0.135 | 0.133 | 2.776 | 0.208 |

## Interpretation

The update-geometry result remains positive: normalized weighted and soft-cone correction eliminate measured first-order oracle harm in these reduced Protenix runs.

The candidate-level result is more mixed. After argmax discretization, normalized weighted and soft-cone produce better sequence sanity in this small run, especially lower trigram loss and lower hydrophobicity, but worse interface structure metrics than single-Protenix and naive weighted: higher binder-target PAE, lower binder-target ipTM, and lower IPSAE.

This is an important design signal. Current safe update geometry may be over-regularizing away from the contact objective, or argmax discretization may be losing the structure-oracle advantage. The next method step should test geometry-aware updates that preserve contact descent more explicitly, and should compare soft versus argmax scoring to separate optimization behavior from discretization artifacts.
