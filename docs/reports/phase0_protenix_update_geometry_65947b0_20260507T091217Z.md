# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_65947b0_20260507T091217Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M7c | contact_preserving_soft_cone | 0.125 | 0.0117 | 0.8784 | 2.003 | 2.7999 | 0.0452 | 0.0004 | 3.1976 |
| M3 | naive_weighted | 0.208 | 0.0102 | 0.8882 | 2.193 | 2.5167 | 0.0696 | 0.0028 | 3.2800 |
| M8b | contact_qp_grid | 0.292 | 0.0074 | 0.7989 | 2.217 | 2.6660 | 0.1022 | 0.0032 | 3.3024 |
| M8a | contact_qp_grid | 0.333 | 0.0192 | 0.8757 | 2.160 | 2.4764 | 0.0748 | 0.0026 | 3.3108 |
| M8c | contact_qp_grid | 0.333 | 0.0187 | 0.8399 | 2.161 | 2.4764 | 0.0748 | 0.0026 | 3.3108 |
| M8d | contact_qp_grid | 0.333 | 0.0187 | 0.8399 | 2.161 | 2.4764 | 0.0748 | 0.0026 | 3.3108 |
| M8e | contact_qp_grid | 0.333 | 0.0187 | 0.8399 | 2.161 | 2.4764 | 0.0748 | 0.0026 | 3.3108 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M8a | contact_qp_grid | soft | 2 | 0.7338 | 8.4671 | 0.5120 | 0.2537 | 2.2693 | 3.3025 |
| M8c | contact_qp_grid | soft | 2 | 0.7235 | 8.8962 | 0.5058 | 0.2517 | 2.3013 | 3.3052 |
| M8d | contact_qp_grid | soft | 2 | 0.7235 | 8.8962 | 0.5058 | 0.2517 | 2.3013 | 3.3052 |
| M8e | contact_qp_grid | soft | 2 | 0.7235 | 8.8962 | 0.5058 | 0.2517 | 2.3013 | 3.3052 |
| M3 | naive_weighted | soft | 2 | 0.7070 | 9.1251 | 0.4977 | 0.2621 | 2.2266 | 3.2113 |
| M8b | contact_qp_grid | soft | 2 | 0.6726 | 10.2546 | 0.4244 | 0.2442 | 2.3082 | 3.3071 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.5631 | 13.4850 | 0.2885 | 0.1491 | 3.0756 | 3.1797 |
| M8b | contact_qp_grid | argmax | 2 | 0.5276 | 14.0917 | 0.2580 | 0.1823 | 2.9004 | 3.3772 |
| M3 | naive_weighted | topk_sample | 16 | 0.5349 | 14.6252 | 0.3089 | 0.1640 | 3.0565 | 3.1999 |
| M8b | contact_qp_grid | topk_sample | 16 | 0.5249 | 14.6786 | 0.2504 | 0.1439 | 3.1026 | 3.3603 |
| M3 | naive_weighted | argmax | 2 | 0.5409 | 14.8048 | 0.2605 | 0.1205 | 3.2083 | 3.2748 |
| M8c | contact_qp_grid | topk_sample | 16 | 0.5383 | 14.9105 | 0.3015 | 0.1489 | 3.0606 | 3.3235 |
| M8d | contact_qp_grid | topk_sample | 16 | 0.5383 | 14.9105 | 0.3015 | 0.1489 | 3.0606 | 3.3235 |
| M8e | contact_qp_grid | topk_sample | 16 | 0.5383 | 14.9105 | 0.3015 | 0.1489 | 3.0606 | 3.3235 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5238 | 15.0303 | 0.2147 | 0.1998 | 2.8422 | 3.1777 |
| M8a | contact_qp_grid | topk_sample | 16 | 0.5245 | 15.1613 | 0.2873 | 0.1519 | 3.0850 | 3.2785 |
| M7c | contact_preserving_soft_cone | topk_sample | 16 | 0.4995 | 15.2499 | 0.2136 | 0.1205 | 3.2693 | 3.0983 |
| M8a | contact_qp_grid | argmax | 2 | 0.4992 | 15.7196 | 0.2583 | 0.1696 | 3.1197 | 3.2589 |
| M8c | contact_qp_grid | argmax | 2 | 0.4979 | 15.8491 | 0.2624 | 0.1623 | 3.1350 | 3.2415 |
| M8d | contact_qp_grid | argmax | 2 | 0.4979 | 15.8491 | 0.2624 | 0.1623 | 3.1350 | 3.2415 |
| M8e | contact_qp_grid | argmax | 2 | 0.4979 | 15.8491 | 0.2624 | 0.1623 | 3.1350 | 3.2415 |

## Best-Per-Seed Candidate Scores

Selection metric: `bt_pae`

| Method ID | Method | Score mode | Selected | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M8a | contact_qp_grid | soft | 2 | 0.7338 | 8.4671 | 0.5120 | 0.2537 | 2.2693 | 3.3025 |
| M8c | contact_qp_grid | soft | 2 | 0.7235 | 8.8962 | 0.5058 | 0.2517 | 2.3013 | 3.3052 |
| M8d | contact_qp_grid | soft | 2 | 0.7235 | 8.8962 | 0.5058 | 0.2517 | 2.3013 | 3.3052 |
| M8e | contact_qp_grid | soft | 2 | 0.7235 | 8.8962 | 0.5058 | 0.2517 | 2.3013 | 3.3052 |
| M3 | naive_weighted | soft | 2 | 0.7070 | 9.1251 | 0.4977 | 0.2621 | 2.2266 | 3.2113 |
| M8b | contact_qp_grid | topk_sample | 2 | 0.6678 | 9.5652 | 0.4657 | 0.2331 | 2.5538 | 3.5562 |
| M8b | contact_qp_grid | soft | 2 | 0.6726 | 10.2546 | 0.4244 | 0.2442 | 2.3082 | 3.3071 |
| M8c | contact_qp_grid | topk_sample | 2 | 0.6651 | 10.8922 | 0.4151 | 0.2060 | 2.4777 | 3.1538 |
| M8d | contact_qp_grid | topk_sample | 2 | 0.6651 | 10.8922 | 0.4151 | 0.2060 | 2.4777 | 3.1538 |
| M8e | contact_qp_grid | topk_sample | 2 | 0.6651 | 10.8922 | 0.4151 | 0.2060 | 2.4777 | 3.1538 |
| M7c | contact_preserving_soft_cone | topk_sample | 2 | 0.6052 | 11.8927 | 0.3434 | 0.1642 | 2.8536 | 3.0667 |
| M8a | contact_qp_grid | topk_sample | 2 | 0.5979 | 12.5003 | 0.4327 | 0.1975 | 2.8127 | 3.1949 |
| M3 | naive_weighted | topk_sample | 2 | 0.5803 | 12.8342 | 0.4270 | 0.1946 | 2.8678 | 3.1827 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.5631 | 13.4850 | 0.2885 | 0.1491 | 3.0756 | 3.1797 |
| M8b | contact_qp_grid | argmax | 2 | 0.5276 | 14.0917 | 0.2580 | 0.1823 | 2.9004 | 3.3772 |
| M3 | naive_weighted | argmax | 2 | 0.5409 | 14.8048 | 0.2605 | 0.1205 | 3.2083 | 3.2748 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5238 | 15.0303 | 0.2147 | 0.1998 | 2.8422 | 3.1777 |
| M8a | contact_qp_grid | argmax | 2 | 0.4992 | 15.7196 | 0.2583 | 0.1696 | 3.1197 | 3.2589 |
| M8c | contact_qp_grid | argmax | 2 | 0.4979 | 15.8491 | 0.2624 | 0.1623 | 3.1350 | 3.2415 |
| M8d | contact_qp_grid | argmax | 2 | 0.4979 | 15.8491 | 0.2624 | 0.1623 | 3.1350 | 3.2415 |
| M8e | contact_qp_grid | argmax | 2 | 0.4979 | 15.8491 | 0.2624 | 0.1623 | 3.1350 | 3.2415 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
