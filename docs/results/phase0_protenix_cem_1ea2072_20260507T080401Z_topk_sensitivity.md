# Phase 0 ACT-012 Top-K Handoff Sensitivity

Source candidates: `docs/results/phase0_protenix_cem_1ea2072_20260507T080401Z_candidates.csv`

## Scope

This analysis reuses one candidate pool and recomputes best-per-seed top-k selection under different sample budgets and reranking metrics. It tests whether the ACT-011 M7c advantage is robust to the decoding budget and selection rule.

## Controls

| Metric | Score mode | Method | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bt_pae | soft | M7c | 0.7099 | 8.6385 | 0.4787 | 0.2266 | 2.3606 | 3.1651 |
| bt_pae | soft | M3 | 0.7506 | 8.6535 | 0.5177 | 0.2725 | 2.1734 | 3.2226 |
| bt_pae | argmax | M3 | 0.5477 | 13.8367 | 0.3213 | 0.1787 | 2.8021 | 3.2898 |
| bt_pae | argmax | M7c | 0.4971 | 14.8590 | 0.2006 | 0.1055 | 3.0861 | 3.0562 |
| bt_pae | argmax | CEMp | 0.5319 | 15.1513 | 0.1884 | 0.1351 | 3.2460 | 3.3614 |
| bt_pae | argmax | CEMc | 0.5028 | 17.5544 | 0.1852 | 0.2555 | 3.1708 | 3.4563 |
| bt_iptm | soft | M7c | 0.7099 | 8.6385 | 0.4787 | 0.2266 | 2.3606 | 3.1651 |
| bt_iptm | soft | M3 | 0.7506 | 8.6535 | 0.5177 | 0.2725 | 2.1734 | 3.2226 |
| bt_iptm | argmax | M3 | 0.5477 | 13.8367 | 0.3213 | 0.1787 | 2.8021 | 3.2898 |
| bt_iptm | argmax | M7c | 0.4971 | 14.8590 | 0.2006 | 0.1055 | 3.0861 | 3.0562 |
| bt_iptm | argmax | CEMp | 0.5319 | 15.1513 | 0.1884 | 0.1351 | 3.2460 | 3.3614 |
| bt_iptm | argmax | CEMc | 0.5028 | 17.5544 | 0.1852 | 0.2555 | 3.1708 | 3.4563 |
| contact | soft | M7c | 0.7099 | 8.6385 | 0.4787 | 0.2266 | 2.3606 | 3.1651 |
| contact | soft | M3 | 0.7506 | 8.6535 | 0.5177 | 0.2725 | 2.1734 | 3.2226 |
| contact | argmax | M3 | 0.5477 | 13.8367 | 0.3213 | 0.1787 | 2.8021 | 3.2898 |
| contact | argmax | M7c | 0.4971 | 14.8590 | 0.2006 | 0.1055 | 3.0861 | 3.0562 |
| contact | argmax | CEMp | 0.5319 | 15.1513 | 0.1884 | 0.1351 | 3.2460 | 3.3614 |
| contact | argmax | CEMc | 0.5028 | 17.5544 | 0.1852 | 0.2555 | 3.1708 | 3.4563 |

## Top-K Selection: `bt_pae`

| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | M3 | 2 | 0.5634 | 13.7316 | 0.3803 | 0.1732 | 2.8684 | 3.1258 |
| 1 | M7c | 2 | 0.5069 | 15.2092 | 0.2628 | 0.1349 | 3.1941 | 3.1779 |
| 1 | CEMc | 2 | 0.5013 | 16.4580 | 0.1844 | 0.2330 | 3.1427 | 3.3009 |
| 1 | CEMp | 2 | 0.5013 | 16.4580 | 0.1844 | 0.2330 | 3.1427 | 3.3009 |
| 8 | M3 | 2 | 0.6417 | 10.7001 | 0.4304 | 0.2035 | 2.5283 | 3.3103 |
| 8 | M7c | 2 | 0.5968 | 11.3088 | 0.3819 | 0.1764 | 2.7732 | 3.2971 |
| 8 | CEMc | 2 | 0.5010 | 16.2828 | 0.1611 | 0.1665 | 3.2061 | 3.3267 |
| 8 | CEMp | 2 | 0.5010 | 16.2828 | 0.1611 | 0.1665 | 3.2061 | 3.3267 |
| 24 | M3 | 2 | 0.6417 | 10.7001 | 0.4304 | 0.2035 | 2.5283 | 3.3103 |
| 24 | M7c | 2 | 0.6095 | 10.8029 | 0.3637 | 0.1759 | 2.7334 | 3.1720 |
| 24 | CEMp | 2 | 0.5358 | 14.9997 | 0.1832 | 0.1763 | 3.2081 | 3.3197 |
| 24 | CEMc | 2 | 0.5117 | 15.8457 | 0.1618 | 0.1759 | 3.1704 | 3.3489 |

## Top-K Selection: `bt_iptm`

| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | M3 | 2 | 0.5634 | 13.7316 | 0.3803 | 0.1732 | 2.8684 | 3.1258 |
| 1 | M7c | 2 | 0.5069 | 15.2092 | 0.2628 | 0.1349 | 3.1941 | 3.1779 |
| 1 | CEMc | 2 | 0.5013 | 16.4580 | 0.1844 | 0.2330 | 3.1427 | 3.3009 |
| 1 | CEMp | 2 | 0.5013 | 16.4580 | 0.1844 | 0.2330 | 3.1427 | 3.3009 |
| 8 | M3 | 2 | 0.6309 | 11.3655 | 0.4668 | 0.2166 | 2.6039 | 3.1732 |
| 8 | M7c | 2 | 0.6097 | 11.5211 | 0.3997 | 0.1864 | 2.7459 | 3.1721 |
| 8 | CEMc | 2 | 0.5013 | 16.4580 | 0.1844 | 0.2330 | 3.1427 | 3.3009 |
| 8 | CEMp | 2 | 0.5013 | 16.4580 | 0.1844 | 0.2330 | 3.1427 | 3.3009 |
| 24 | M7c | 2 | 0.5913 | 11.1558 | 0.4158 | 0.1832 | 2.8117 | 3.2145 |
| 24 | M3 | 2 | 0.6163 | 11.4539 | 0.4671 | 0.2043 | 2.9233 | 3.2326 |
| 24 | CEMp | 2 | 0.5304 | 15.5263 | 0.2244 | 0.2260 | 3.0181 | 3.4053 |
| 24 | CEMc | 2 | 0.5261 | 15.8520 | 0.2111 | 0.2323 | 3.0677 | 3.3875 |

## Top-K Selection: `contact`

| Budget | Method | Selected | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | M3 | 2 | 0.5634 | 13.7316 | 0.3803 | 0.1732 | 2.8684 | 3.1258 |
| 1 | M7c | 2 | 0.5069 | 15.2092 | 0.2628 | 0.1349 | 3.1941 | 3.1779 |
| 1 | CEMc | 2 | 0.5013 | 16.4580 | 0.1844 | 0.2330 | 3.1427 | 3.3009 |
| 1 | CEMp | 2 | 0.5013 | 16.4580 | 0.1844 | 0.2330 | 3.1427 | 3.3009 |
| 8 | M3 | 2 | 0.6417 | 10.7001 | 0.4304 | 0.2035 | 2.5283 | 3.3103 |
| 8 | M7c | 2 | 0.5810 | 12.2175 | 0.3357 | 0.1972 | 2.6590 | 3.0205 |
| 8 | CEMc | 2 | 0.4979 | 16.6591 | 0.1540 | 0.1946 | 3.1353 | 3.4392 |
| 8 | CEMp | 2 | 0.4979 | 16.6591 | 0.1540 | 0.1946 | 3.1353 | 3.4392 |
| 24 | M3 | 2 | 0.6349 | 10.8668 | 0.4268 | 0.2134 | 2.5231 | 3.3580 |
| 24 | M7c | 2 | 0.6043 | 10.9744 | 0.3579 | 0.1696 | 2.5964 | 3.0716 |
| 24 | CEMp | 2 | 0.5285 | 15.4387 | 0.2097 | 0.2641 | 2.9890 | 3.3692 |
| 24 | CEMc | 2 | 0.5261 | 15.8520 | 0.2111 | 0.2323 | 3.0677 | 3.3875 |

## Interpretation Guide

- A robust positive signal means M7c stays near the top across budgets and rerank metrics under matched candidate scoring budget.
- If gains appear only at high budget, report the quality/cost tradeoff rather than claiming a free decoding improvement.
- If metric choice changes the winner, the next experiment should pre-register the paper-facing selection metric before scale-up.
