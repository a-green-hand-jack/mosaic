# Phase 0 ACT-012 Top-K Handoff Sensitivity

Source candidates: `docs/results/phase0_protenix_update_geometry_f89ee15_20260507T054055Z_candidates.csv`

## Scope

This analysis reuses one candidate pool and recomputes best-per-seed top-k selection under different sample budgets and reranking metrics. It tests whether the ACT-011 M7c advantage is robust to the decoding budget and selection rule.

## Controls

| Metric | Score mode | Method | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bt_pae | soft | M3 | 0.7121 | 9.2516 | 0.4768 | 0.2354 | 2.3568 | 3.2300 |
| bt_pae | soft | M7a | 0.6655 | 9.8743 | 0.4511 | 0.2147 | 2.4897 | 3.1069 |
| bt_pae | soft | M7c | 0.5903 | 12.3838 | 0.3132 | 0.1852 | 2.7865 | 3.1889 |
| bt_pae | argmax | M3 | 0.5383 | 14.3020 | 0.3482 | 0.1673 | 2.9194 | 3.2245 |
| bt_pae | argmax | M7a | 0.4640 | 15.7244 | 0.2532 | 0.1575 | 3.1017 | 2.9411 |
| bt_pae | argmax | M7c | 0.4491 | 17.1274 | 0.1236 | 0.0674 | 3.4257 | 2.9787 |
| bt_iptm | soft | M3 | 0.7121 | 9.2516 | 0.4768 | 0.2354 | 2.3568 | 3.2300 |
| bt_iptm | soft | M7a | 0.6655 | 9.8743 | 0.4511 | 0.2147 | 2.4897 | 3.1069 |
| bt_iptm | soft | M7c | 0.5903 | 12.3838 | 0.3132 | 0.1852 | 2.7865 | 3.1889 |
| bt_iptm | argmax | M3 | 0.5383 | 14.3020 | 0.3482 | 0.1673 | 2.9194 | 3.2245 |
| bt_iptm | argmax | M7a | 0.4640 | 15.7244 | 0.2532 | 0.1575 | 3.1017 | 2.9411 |
| bt_iptm | argmax | M7c | 0.4491 | 17.1274 | 0.1236 | 0.0674 | 3.4257 | 2.9787 |
| contact | soft | M3 | 0.7121 | 9.2516 | 0.4768 | 0.2354 | 2.3568 | 3.2300 |
| contact | soft | M7a | 0.6655 | 9.8743 | 0.4511 | 0.2147 | 2.4897 | 3.1069 |
| contact | soft | M7c | 0.5903 | 12.3838 | 0.3132 | 0.1852 | 2.7865 | 3.1889 |
| contact | argmax | M3 | 0.5383 | 14.3020 | 0.3482 | 0.1673 | 2.9194 | 3.2245 |
| contact | argmax | M7a | 0.4640 | 15.7244 | 0.2532 | 0.1575 | 3.1017 | 2.9411 |
| contact | argmax | M7c | 0.4491 | 17.1274 | 0.1236 | 0.0674 | 3.4257 | 2.9787 |

## Top-K Selection: `bt_pae`

| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | M3 | 2 | 0.4910 | 14.6378 | 0.2522 | 0.1882 | 3.1526 | 3.1967 |
| 1 | M7a | 2 | 0.5112 | 14.7479 | 0.2843 | 0.2046 | 2.9754 | 3.0344 |
| 1 | M7c | 2 | 0.4317 | 17.5181 | 0.1387 | 0.0980 | 3.5281 | 3.1254 |
| 4 | M3 | 2 | 0.5692 | 12.7383 | 0.3321 | 0.1937 | 2.9221 | 3.3114 |
| 4 | M7a | 2 | 0.5294 | 13.3883 | 0.2772 | 0.1863 | 2.9555 | 2.9412 |
| 4 | M7c | 2 | 0.5092 | 15.4502 | 0.2413 | 0.1101 | 3.3838 | 3.1055 |
| 8 | M7c | 2 | 0.6043 | 11.9860 | 0.3822 | 0.1779 | 2.8391 | 3.0675 |
| 8 | M3 | 2 | 0.5692 | 12.7383 | 0.3321 | 0.1937 | 2.9221 | 3.3114 |
| 8 | M7a | 2 | 0.5294 | 13.3883 | 0.2772 | 0.1863 | 2.9555 | 2.9412 |

## Top-K Selection: `bt_iptm`

| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | M3 | 2 | 0.4910 | 14.6378 | 0.2522 | 0.1882 | 3.1526 | 3.1967 |
| 1 | M7a | 2 | 0.5112 | 14.7479 | 0.2843 | 0.2046 | 2.9754 | 3.0344 |
| 1 | M7c | 2 | 0.4317 | 17.5181 | 0.1387 | 0.0980 | 3.5281 | 3.1254 |
| 4 | M3 | 2 | 0.5735 | 13.2157 | 0.4067 | 0.1946 | 2.8411 | 3.1367 |
| 4 | M7a | 2 | 0.5407 | 13.6787 | 0.3230 | 0.1780 | 2.9604 | 2.9512 |
| 4 | M7c | 2 | 0.4826 | 15.9400 | 0.2550 | 0.1753 | 3.2087 | 3.1716 |
| 8 | M7c | 2 | 0.6043 | 11.9860 | 0.3822 | 0.1779 | 2.8391 | 3.0675 |
| 8 | M3 | 2 | 0.5809 | 13.3704 | 0.4395 | 0.1993 | 2.8933 | 3.1563 |
| 8 | M7a | 2 | 0.5481 | 13.8335 | 0.3559 | 0.1828 | 3.0126 | 2.9709 |

## Top-K Selection: `contact`

| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | M3 | 2 | 0.4910 | 14.6378 | 0.2522 | 0.1882 | 3.1526 | 3.1967 |
| 1 | M7a | 2 | 0.5112 | 14.7479 | 0.2843 | 0.2046 | 2.9754 | 3.0344 |
| 1 | M7c | 2 | 0.4317 | 17.5181 | 0.1387 | 0.0980 | 3.5281 | 3.1254 |
| 4 | M3 | 2 | 0.5735 | 13.2157 | 0.4067 | 0.1946 | 2.8411 | 3.1367 |
| 4 | M7a | 2 | 0.5294 | 13.3883 | 0.2772 | 0.1863 | 2.9555 | 2.9412 |
| 4 | M7c | 2 | 0.4826 | 15.9400 | 0.2550 | 0.1753 | 3.2087 | 3.1716 |
| 8 | M7c | 2 | 0.6043 | 11.9860 | 0.3822 | 0.1779 | 2.8391 | 3.0675 |
| 8 | M3 | 2 | 0.5872 | 12.7911 | 0.3513 | 0.1763 | 2.8374 | 3.0960 |
| 8 | M7a | 2 | 0.5294 | 13.3883 | 0.2772 | 0.1863 | 2.9555 | 2.9412 |

## Interpretation Guide

- A robust positive signal means M7c stays near the top across budgets and rerank metrics under matched candidate scoring budget.
- If gains appear only at high budget, report the quality/cost tradeoff rather than claiming a free decoding improvement.
- If metric choice changes the winner, the next experiment should pre-register the paper-facing selection metric before scale-up.
