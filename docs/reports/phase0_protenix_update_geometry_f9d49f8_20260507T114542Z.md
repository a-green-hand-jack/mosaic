# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_f9d49f8_20260507T114542Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M7c | contact_preserving_soft_cone | 0.125 | 0.0062 | 0.6354 | 2.297 | 2.5949 | 0.0495 | 0.0004 | 3.2119 |
| M3 | naive_weighted | 0.167 | 0.0098 | 0.8403 | 2.214 | 2.4320 | 0.0632 | 0.0027 | 3.2744 |
| M9a | position_hardening_probability | 0.208 | 0.0104 | 1.2741 | 1.864 | 2.7108 | 0.0594 | 0.0034 | 3.2226 |
| M9b | position_hardening_consensus | 0.250 | 0.0226 | 1.3712 | 1.844 | 2.7170 | 0.0712 | 0.0025 | 3.2273 |
| M8a | contact_qp_grid | 0.292 | 0.0181 | 0.9140 | 2.153 | 2.3771 | 0.0799 | 0.0026 | 3.3254 |
| M8g | contact_qp_grid_contact_first | 0.292 | 0.0181 | 0.9140 | 2.153 | 2.3771 | 0.0799 | 0.0026 | 3.3254 |
| M9c | position_hardening_margin_lowsensitivity | 0.375 | 0.0138 | 1.4306 | 1.665 | 2.7766 | 0.0626 | 0.0070 | 3.3474 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.7264 | 8.8648 | 0.5203 | 0.2641 | 2.2379 | 3.2177 |
| M8a | contact_qp_grid | soft | 2 | 0.7232 | 8.9176 | 0.5123 | 0.2796 | 2.1760 | 3.3094 |
| M8g | contact_qp_grid_contact_first | soft | 2 | 0.7232 | 8.9176 | 0.5123 | 0.2796 | 2.1760 | 3.3094 |
| M9a | position_hardening_probability | soft | 2 | 0.6550 | 11.8653 | 0.4342 | 0.2249 | 2.5009 | 3.1514 |
| M9b | position_hardening_consensus | soft | 2 | 0.6415 | 11.9599 | 0.3836 | 0.2047 | 2.5460 | 3.2242 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.6097 | 12.0928 | 0.4546 | 0.2673 | 2.4589 | 3.1408 |
| M9c | position_hardening_margin_lowsensitivity | soft | 2 | 0.5577 | 13.5241 | 0.3101 | 0.1834 | 2.8967 | 3.2674 |
| M9b | position_hardening_consensus | topk_sample | 16 | 0.5581 | 13.6960 | 0.3364 | 0.1687 | 2.9631 | 3.2101 |
| M8a | contact_qp_grid | argmax | 2 | 0.5674 | 13.9343 | 0.4180 | 0.1890 | 2.9447 | 3.3330 |
| M8g | contact_qp_grid_contact_first | argmax | 2 | 0.5674 | 13.9343 | 0.4180 | 0.1890 | 2.9447 | 3.3330 |
| M9a | position_hardening_probability | topk_sample | 16 | 0.5532 | 14.1966 | 0.3383 | 0.1685 | 2.9823 | 3.1781 |
| M3 | naive_weighted | topk_sample | 16 | 0.5470 | 14.2652 | 0.3314 | 0.1664 | 2.9759 | 3.2352 |
| M8a | contact_qp_grid | topk_sample | 16 | 0.5420 | 14.7611 | 0.3590 | 0.1845 | 2.9907 | 3.2700 |
| M8g | contact_qp_grid_contact_first | topk_sample | 16 | 0.5420 | 14.7611 | 0.3590 | 0.1845 | 2.9907 | 3.2700 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.5137 | 15.1793 | 0.3322 | 0.1491 | 3.3261 | 3.1815 |
| M9b | position_hardening_consensus | argmax | 2 | 0.5077 | 15.7565 | 0.2706 | 0.1389 | 3.1890 | 3.2454 |
| M7c | contact_preserving_soft_cone | topk_sample | 16 | 0.4956 | 16.1536 | 0.2689 | 0.1284 | 3.2982 | 3.2259 |
| M9a | position_hardening_probability | argmax | 2 | 0.4871 | 16.3863 | 0.2381 | 0.1199 | 3.2720 | 3.1921 |
| M3 | naive_weighted | argmax | 2 | 0.4826 | 16.4609 | 0.2255 | 0.1168 | 3.2661 | 3.3044 |
| M9c | position_hardening_margin_lowsensitivity | topk_sample | 16 | 0.4528 | 17.3013 | 0.1644 | 0.0889 | 3.5482 | 3.2986 |
| M9c | position_hardening_margin_lowsensitivity | argmax | 2 | 0.4389 | 18.0034 | 0.1474 | 0.1050 | 3.5877 | 3.3933 |

## Best-Per-Seed Candidate Scores

Selection metric: `bt_pae`

| Method ID | Method | Score mode | Selected | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.7264 | 8.8648 | 0.5203 | 0.2641 | 2.2379 | 3.2177 |
| M8a | contact_qp_grid | soft | 2 | 0.7232 | 8.9176 | 0.5123 | 0.2796 | 2.1760 | 3.3094 |
| M8g | contact_qp_grid_contact_first | soft | 2 | 0.7232 | 8.9176 | 0.5123 | 0.2796 | 2.1760 | 3.3094 |
| M9b | position_hardening_consensus | topk_sample | 2 | 0.6474 | 10.8833 | 0.4278 | 0.1950 | 2.7431 | 3.2544 |
| M7c | contact_preserving_soft_cone | topk_sample | 2 | 0.6220 | 11.4603 | 0.4739 | 0.2213 | 2.6948 | 3.2957 |
| M9a | position_hardening_probability | soft | 2 | 0.6550 | 11.8653 | 0.4342 | 0.2249 | 2.5009 | 3.1514 |
| M9b | position_hardening_consensus | soft | 2 | 0.6415 | 11.9599 | 0.3836 | 0.2047 | 2.5460 | 3.2242 |
| M3 | naive_weighted | topk_sample | 2 | 0.6001 | 12.0435 | 0.4251 | 0.1954 | 2.7943 | 3.2524 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.6097 | 12.0928 | 0.4546 | 0.2673 | 2.4589 | 3.1408 |
| M9a | position_hardening_probability | topk_sample | 2 | 0.5891 | 12.2906 | 0.4200 | 0.1901 | 2.8949 | 3.0248 |
| M8a | contact_qp_grid | topk_sample | 2 | 0.5866 | 12.9376 | 0.4620 | 0.2111 | 2.8984 | 3.3058 |
| M8g | contact_qp_grid_contact_first | topk_sample | 2 | 0.5866 | 12.9376 | 0.4620 | 0.2111 | 2.8984 | 3.3058 |
| M9c | position_hardening_margin_lowsensitivity | soft | 2 | 0.5577 | 13.5241 | 0.3101 | 0.1834 | 2.8967 | 3.2674 |
| M8a | contact_qp_grid | argmax | 2 | 0.5674 | 13.9343 | 0.4180 | 0.1890 | 2.9447 | 3.3330 |
| M8g | contact_qp_grid_contact_first | argmax | 2 | 0.5674 | 13.9343 | 0.4180 | 0.1890 | 2.9447 | 3.3330 |
| M9c | position_hardening_margin_lowsensitivity | topk_sample | 2 | 0.5368 | 14.4829 | 0.3265 | 0.1912 | 3.1365 | 3.3582 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.5137 | 15.1793 | 0.3322 | 0.1491 | 3.3261 | 3.1815 |
| M9b | position_hardening_consensus | argmax | 2 | 0.5077 | 15.7565 | 0.2706 | 0.1389 | 3.1890 | 3.2454 |
| M9a | position_hardening_probability | argmax | 2 | 0.4871 | 16.3863 | 0.2381 | 0.1199 | 3.2720 | 3.1921 |
| M3 | naive_weighted | argmax | 2 | 0.4826 | 16.4609 | 0.2255 | 0.1168 | 3.2661 | 3.3044 |
| M9c | position_hardening_margin_lowsensitivity | argmax | 2 | 0.4389 | 18.0034 | 0.1474 | 0.1050 | 3.5877 | 3.3933 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
