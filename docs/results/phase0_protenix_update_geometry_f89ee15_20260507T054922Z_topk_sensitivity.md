# Phase 0 ACT-012 Top-K Handoff Sensitivity

Source candidates: `docs/results/phase0_protenix_update_geometry_f89ee15_20260507T054922Z_candidates.csv`

## Scope

This analysis reuses one candidate pool and recomputes best-per-seed top-k selection under different sample budgets and reranking metrics. It tests whether the ACT-011 M7c advantage is robust to the decoding budget and selection rule.

## Controls

| Metric | Score mode | Method | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bt_pae | soft | M3 | 0.7300 | 8.8039 | 0.5043 | 0.2510 | 2.4163 | 3.2174 |
| bt_pae | soft | M7a | 0.6843 | 9.9190 | 0.4735 | 0.2358 | 2.5749 | 3.1136 |
| bt_pae | soft | M7c | 0.5888 | 12.1754 | 0.3553 | 0.2095 | 2.6064 | 3.1223 |
| bt_pae | argmax | M7c | 0.4924 | 15.2979 | 0.2354 | 0.1137 | 3.3585 | 3.2104 |
| bt_pae | argmax | M3 | 0.4976 | 16.0961 | 0.2372 | 0.1234 | 3.2173 | 3.3406 |
| bt_pae | argmax | M7a | 0.4638 | 16.5552 | 0.1747 | 0.1232 | 3.3827 | 3.1593 |
| bt_iptm | soft | M3 | 0.7300 | 8.8039 | 0.5043 | 0.2510 | 2.4163 | 3.2174 |
| bt_iptm | soft | M7a | 0.6843 | 9.9190 | 0.4735 | 0.2358 | 2.5749 | 3.1136 |
| bt_iptm | soft | M7c | 0.5888 | 12.1754 | 0.3553 | 0.2095 | 2.6064 | 3.1223 |
| bt_iptm | argmax | M7c | 0.4924 | 15.2979 | 0.2354 | 0.1137 | 3.3585 | 3.2104 |
| bt_iptm | argmax | M3 | 0.4976 | 16.0961 | 0.2372 | 0.1234 | 3.2173 | 3.3406 |
| bt_iptm | argmax | M7a | 0.4638 | 16.5552 | 0.1747 | 0.1232 | 3.3827 | 3.1593 |
| contact | soft | M3 | 0.7300 | 8.8039 | 0.5043 | 0.2510 | 2.4163 | 3.2174 |
| contact | soft | M7a | 0.6843 | 9.9190 | 0.4735 | 0.2358 | 2.5749 | 3.1136 |
| contact | soft | M7c | 0.5888 | 12.1754 | 0.3553 | 0.2095 | 2.6064 | 3.1223 |
| contact | argmax | M7c | 0.4924 | 15.2979 | 0.2354 | 0.1137 | 3.3585 | 3.2104 |
| contact | argmax | M3 | 0.4976 | 16.0961 | 0.2372 | 0.1234 | 3.2173 | 3.3406 |
| contact | argmax | M7a | 0.4638 | 16.5552 | 0.1747 | 0.1232 | 3.3827 | 3.1593 |

## Top-K Selection: `bt_pae`

| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | M7c | 2 | 0.5305 | 14.2063 | 0.2945 | 0.1594 | 2.9968 | 3.1843 |
| 1 | M3 | 2 | 0.5304 | 15.2888 | 0.3143 | 0.2021 | 3.1166 | 3.2105 |
| 1 | M7a | 2 | 0.4677 | 16.6733 | 0.1991 | 0.1909 | 3.2451 | 3.0965 |
| 4 | M7c | 2 | 0.5908 | 11.5974 | 0.3627 | 0.1464 | 3.1651 | 3.1870 |
| 4 | M3 | 2 | 0.5299 | 15.1737 | 0.2874 | 0.1914 | 3.0334 | 3.1396 |
| 4 | M7a | 2 | 0.4672 | 16.5582 | 0.1722 | 0.1802 | 3.1620 | 3.0256 |
| 8 | M7c | 2 | 0.5908 | 11.5974 | 0.3627 | 0.1464 | 3.1651 | 3.1870 |
| 8 | M3 | 2 | 0.5466 | 13.9437 | 0.3198 | 0.2181 | 2.9304 | 3.2688 |
| 8 | M7a | 2 | 0.4937 | 15.6819 | 0.2797 | 0.2102 | 3.1842 | 3.1622 |

## Top-K Selection: `bt_iptm`

| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | M7c | 2 | 0.5305 | 14.2063 | 0.2945 | 0.1594 | 2.9968 | 3.1843 |
| 1 | M3 | 2 | 0.5304 | 15.2888 | 0.3143 | 0.2021 | 3.1166 | 3.2105 |
| 1 | M7a | 2 | 0.4677 | 16.6733 | 0.1991 | 0.1909 | 3.2451 | 3.0965 |
| 4 | M7c | 2 | 0.5966 | 11.8674 | 0.4438 | 0.1887 | 2.9128 | 3.2530 |
| 4 | M3 | 2 | 0.5304 | 15.2888 | 0.3143 | 0.2021 | 3.1166 | 3.2105 |
| 4 | M7a | 2 | 0.4677 | 16.6733 | 0.1991 | 0.1909 | 3.2451 | 3.0965 |
| 8 | M7c | 2 | 0.5966 | 11.8674 | 0.4438 | 0.1887 | 2.9128 | 3.2530 |
| 8 | M3 | 2 | 0.5450 | 14.5211 | 0.3460 | 0.2284 | 2.9939 | 3.1818 |
| 8 | M7a | 2 | 0.4937 | 15.6819 | 0.2797 | 0.2102 | 3.1842 | 3.1622 |

## Top-K Selection: `contact`

| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | M7c | 2 | 0.5305 | 14.2063 | 0.2945 | 0.1594 | 2.9968 | 3.1843 |
| 1 | M3 | 2 | 0.5304 | 15.2888 | 0.3143 | 0.2021 | 3.1166 | 3.2105 |
| 1 | M7a | 2 | 0.4677 | 16.6733 | 0.1991 | 0.1909 | 3.2451 | 3.0965 |
| 4 | M7c | 2 | 0.5783 | 12.2823 | 0.3969 | 0.1729 | 2.8855 | 3.2275 |
| 4 | M3 | 2 | 0.5249 | 15.2886 | 0.2839 | 0.1962 | 2.9930 | 3.4003 |
| 4 | M7a | 2 | 0.4672 | 16.5582 | 0.1722 | 0.1802 | 3.1620 | 3.0256 |
| 8 | M7c | 2 | 0.5783 | 12.2823 | 0.3969 | 0.1729 | 2.8855 | 3.2275 |
| 8 | M3 | 2 | 0.5466 | 13.9437 | 0.3198 | 0.2181 | 2.9304 | 3.2688 |
| 8 | M7a | 2 | 0.4823 | 15.9056 | 0.2308 | 0.2172 | 3.1224 | 3.0678 |

## Interpretation Guide

- A robust positive signal means M7c stays near the top across budgets and rerank metrics under matched candidate scoring budget.
- If gains appear only at high budget, report the quality/cost tradeoff rather than claiming a free decoding improvement.
- If metric choice changes the winner, the next experiment should pre-register the paper-facing selection metric before scale-up.
