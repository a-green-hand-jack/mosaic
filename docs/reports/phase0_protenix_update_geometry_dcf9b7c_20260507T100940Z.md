# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_dcf9b7c_20260507T100940Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M7c | contact_preserving_soft_cone | 0.125 | 0.0095 | 0.7319 | 2.209 | 2.8205 | 0.0459 | 0.0004 | 3.1999 |
| M3 | naive_weighted | 0.208 | 0.0103 | 0.8930 | 2.183 | 2.4115 | 0.0690 | 0.0028 | 3.2777 |
| M8b | contact_qp_grid | 0.292 | 0.0061 | 0.8628 | 2.178 | 2.5926 | 0.1013 | 0.0032 | 3.3026 |
| M8h | contact_qp_grid | 0.292 | 0.0061 | 0.8628 | 2.178 | 2.5926 | 0.1013 | 0.0032 | 3.3026 |
| M8f | contact_qp_grid_contact_first | 0.333 | 0.0154 | 0.8752 | 2.167 | 2.4162 | 0.0823 | 0.0028 | 3.3076 |
| M8a | contact_qp_grid | 0.375 | 0.0203 | 0.8757 | 2.170 | 2.4490 | 0.0752 | 0.0026 | 3.3119 |
| M8g | contact_qp_grid_contact_first | 0.375 | 0.0203 | 0.8757 | 2.170 | 2.4490 | 0.0752 | 0.0026 | 3.3119 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M8a | contact_qp_grid | soft | 2 | 0.7317 | 8.7733 | 0.5189 | 0.2626 | 2.2505 | 3.3112 |
| M8g | contact_qp_grid_contact_first | soft | 2 | 0.7317 | 8.7733 | 0.5189 | 0.2626 | 2.2505 | 3.3112 |
| M3 | naive_weighted | soft | 2 | 0.7061 | 9.5202 | 0.4521 | 0.2623 | 2.3085 | 3.2164 |
| M8f | contact_qp_grid_contact_first | soft | 2 | 0.6275 | 12.0293 | 0.3605 | 0.2344 | 2.7618 | 3.3061 |
| M8b | contact_qp_grid | soft | 2 | 0.6045 | 12.7037 | 0.3013 | 0.2043 | 2.8432 | 3.3066 |
| M8h | contact_qp_grid | soft | 2 | 0.6045 | 12.7037 | 0.3013 | 0.2043 | 2.8432 | 3.3066 |
| M8a | contact_qp_grid | argmax | 2 | 0.5601 | 13.5433 | 0.3320 | 0.1652 | 2.8391 | 3.1758 |
| M8g | contact_qp_grid_contact_first | argmax | 2 | 0.5601 | 13.5433 | 0.3320 | 0.1652 | 2.8391 | 3.1758 |
| M8a | contact_qp_grid | topk_sample | 16 | 0.5557 | 13.9175 | 0.3357 | 0.1750 | 2.9499 | 3.3578 |
| M8g | contact_qp_grid_contact_first | topk_sample | 16 | 0.5557 | 13.9175 | 0.3357 | 0.1750 | 2.9499 | 3.3578 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.5286 | 14.6620 | 0.2724 | 0.1195 | 3.2752 | 2.8716 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5114 | 15.1401 | 0.2151 | 0.1731 | 2.9642 | 3.1569 |
| M3 | naive_weighted | topk_sample | 16 | 0.5107 | 15.5571 | 0.2752 | 0.1629 | 3.0808 | 3.3169 |
| M8f | contact_qp_grid_contact_first | topk_sample | 16 | 0.5104 | 15.6328 | 0.2678 | 0.1585 | 3.1268 | 3.3575 |
| M7c | contact_preserving_soft_cone | topk_sample | 16 | 0.4849 | 16.0244 | 0.1906 | 0.1190 | 3.3080 | 3.1168 |
| M3 | naive_weighted | argmax | 2 | 0.4935 | 16.2362 | 0.2374 | 0.1331 | 3.1624 | 3.3002 |
| M8b | contact_qp_grid | topk_sample | 16 | 0.4780 | 16.4981 | 0.1679 | 0.1210 | 3.3642 | 3.3199 |
| M8h | contact_qp_grid | topk_sample | 16 | 0.4780 | 16.4981 | 0.1679 | 0.1210 | 3.3642 | 3.3199 |
| M8f | contact_qp_grid_contact_first | argmax | 2 | 0.4862 | 16.7880 | 0.2241 | 0.1411 | 3.1296 | 3.3409 |
| M8b | contact_qp_grid | argmax | 2 | 0.4090 | 19.0436 | 0.0837 | 0.0599 | 3.5486 | 3.3275 |
| M8h | contact_qp_grid | argmax | 2 | 0.4090 | 19.0436 | 0.0837 | 0.0599 | 3.5486 | 3.3275 |

## Best-Per-Seed Candidate Scores

Selection metric: `bt_pae`

| Method ID | Method | Score mode | Selected | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M8a | contact_qp_grid | soft | 2 | 0.7317 | 8.7733 | 0.5189 | 0.2626 | 2.2505 | 3.3112 |
| M8g | contact_qp_grid_contact_first | soft | 2 | 0.7317 | 8.7733 | 0.5189 | 0.2626 | 2.2505 | 3.3112 |
| M3 | naive_weighted | soft | 2 | 0.7061 | 9.5202 | 0.4521 | 0.2623 | 2.3085 | 3.2164 |
| M8a | contact_qp_grid | topk_sample | 2 | 0.6151 | 11.3190 | 0.4128 | 0.1855 | 2.7149 | 3.1891 |
| M8g | contact_qp_grid_contact_first | topk_sample | 2 | 0.6151 | 11.3190 | 0.4128 | 0.1855 | 2.7149 | 3.1891 |
| M8f | contact_qp_grid_contact_first | soft | 2 | 0.6275 | 12.0293 | 0.3605 | 0.2344 | 2.7618 | 3.3061 |
| M8b | contact_qp_grid | soft | 2 | 0.6045 | 12.7037 | 0.3013 | 0.2043 | 2.8432 | 3.3066 |
| M8h | contact_qp_grid | soft | 2 | 0.6045 | 12.7037 | 0.3013 | 0.2043 | 2.8432 | 3.3066 |
| M8f | contact_qp_grid_contact_first | topk_sample | 2 | 0.5736 | 12.8555 | 0.3366 | 0.2302 | 2.7247 | 3.3704 |
| M8b | contact_qp_grid | topk_sample | 2 | 0.5763 | 12.9064 | 0.3019 | 0.2365 | 2.6947 | 3.4897 |
| M8h | contact_qp_grid | topk_sample | 2 | 0.5763 | 12.9064 | 0.3019 | 0.2365 | 2.6947 | 3.4897 |
| M8a | contact_qp_grid | argmax | 2 | 0.5601 | 13.5433 | 0.3320 | 0.1652 | 2.8391 | 3.1758 |
| M8g | contact_qp_grid_contact_first | argmax | 2 | 0.5601 | 13.5433 | 0.3320 | 0.1652 | 2.8391 | 3.1758 |
| M3 | naive_weighted | topk_sample | 2 | 0.5494 | 13.5906 | 0.3783 | 0.1974 | 2.9790 | 3.3315 |
| M7c | contact_preserving_soft_cone | topk_sample | 2 | 0.5577 | 13.6600 | 0.3315 | 0.1895 | 2.9760 | 3.0839 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.5286 | 14.6620 | 0.2724 | 0.1195 | 3.2752 | 2.8716 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.5114 | 15.1401 | 0.2151 | 0.1731 | 2.9642 | 3.1569 |
| M3 | naive_weighted | argmax | 2 | 0.4935 | 16.2362 | 0.2374 | 0.1331 | 3.1624 | 3.3002 |
| M8f | contact_qp_grid_contact_first | argmax | 2 | 0.4862 | 16.7880 | 0.2241 | 0.1411 | 3.1296 | 3.3409 |
| M8b | contact_qp_grid | argmax | 2 | 0.4090 | 19.0436 | 0.0837 | 0.0599 | 3.5486 | 3.3275 |
| M8h | contact_qp_grid | argmax | 2 | 0.4090 | 19.0436 | 0.0837 | 0.0599 | 3.5486 | 3.3275 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
