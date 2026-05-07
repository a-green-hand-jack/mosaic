# Phase 0 Protenix Update-Geometry Smoke

Run ID: `phase0_protenix_update_geometry_09ed71d_20260507T045421Z`

## Scope

This smoke run adds one structure oracle to the update-geometry diagnostic: ProtenixMini binder-target contact. It remains small-scale and should be treated as a runtime and mechanism check, not as a final binder-design result.

## Summary

| Method ID | Method | Harm rate | Worst derivative | Step norm | Final entropy | Final Protenix contact loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M4 | normalized_weighted | 0.000 | -0.0086 | 0.4295 | 2.519 | 2.7998 | 0.0056 | 0.0068 | 3.0752 |
| M7a | contact_preserving_soft_cone | 0.000 | -0.0046 | 0.5359 | 2.411 | 2.8597 | 0.0282 | 0.0054 | 3.1591 |
| M7c | contact_preserving_soft_cone | 0.167 | 0.0088 | 0.9304 | 2.312 | 2.8918 | 0.0529 | 0.0008 | 3.2183 |
| M3 | naive_weighted | 0.208 | 0.0148 | 0.9074 | 2.152 | 2.6145 | 0.0612 | 0.0027 | 3.2794 |

## Candidate Holdout Scores

| Method ID | Method | Score mode | Candidates | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.6759 | 9.9052 | 0.4424 | 0.2993 | 2.3982 | 3.2272 |
| M4 | normalized_weighted | soft | 2 | 0.6368 | 10.7309 | 0.3795 | 0.2217 | 2.5153 | 3.0205 |
| M7a | contact_preserving_soft_cone | soft | 2 | 0.6070 | 11.4650 | 0.3637 | 0.2282 | 2.5162 | 3.0918 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.6010 | 11.6110 | 0.3747 | 0.1835 | 2.8562 | 3.1321 |
| M3 | naive_weighted | argmax | 2 | 0.5142 | 15.1363 | 0.2715 | 0.1927 | 3.1093 | 3.2178 |
| M7a | contact_preserving_soft_cone | argmax | 2 | 0.4660 | 15.5488 | 0.2049 | 0.1662 | 3.1162 | 3.0815 |
| M7c | contact_preserving_soft_cone | topk_sample | 8 | 0.5096 | 15.8924 | 0.2609 | 0.1404 | 3.2741 | 3.1026 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4710 | 16.5451 | 0.2141 | 0.1045 | 3.3326 | 3.2278 |
| M3 | naive_weighted | topk_sample | 8 | 0.4915 | 16.5645 | 0.2408 | 0.1196 | 3.2889 | 3.2856 |
| M7a | contact_preserving_soft_cone | topk_sample | 8 | 0.4689 | 16.8694 | 0.1968 | 0.1234 | 3.3834 | 2.9955 |
| M4 | normalized_weighted | argmax | 2 | 0.4279 | 17.6580 | 0.1483 | 0.0813 | 3.5159 | 2.7658 |
| M4 | normalized_weighted | topk_sample | 8 | 0.4625 | 18.0069 | 0.1549 | 0.0608 | 3.6168 | 2.8754 |

## Best-Per-Seed Candidate Scores

Selection metric: `bt_pae`

| Method ID | Method | Score mode | Selected | pLDDT | Binder-target PAE | Binder-target ipTM | IPSAE min | Contact loss | Trigram loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M3 | naive_weighted | soft | 2 | 0.6759 | 9.9052 | 0.4424 | 0.2993 | 2.3982 | 3.2272 |
| M4 | normalized_weighted | soft | 2 | 0.6368 | 10.7309 | 0.3795 | 0.2217 | 2.5153 | 3.0205 |
| M7a | contact_preserving_soft_cone | soft | 2 | 0.6070 | 11.4650 | 0.3637 | 0.2282 | 2.5162 | 3.0918 |
| M7c | contact_preserving_soft_cone | soft | 2 | 0.6010 | 11.6110 | 0.3747 | 0.1835 | 2.8562 | 3.1321 |
| M7c | contact_preserving_soft_cone | topk_sample | 2 | 0.5953 | 11.9610 | 0.4313 | 0.1931 | 2.8101 | 3.2221 |
| M7a | contact_preserving_soft_cone | topk_sample | 2 | 0.5417 | 14.0909 | 0.3045 | 0.2241 | 2.7869 | 3.0146 |
| M3 | naive_weighted | topk_sample | 2 | 0.5475 | 14.8959 | 0.3663 | 0.2048 | 2.9420 | 3.1765 |
| M3 | naive_weighted | argmax | 2 | 0.5142 | 15.1363 | 0.2715 | 0.1927 | 3.1093 | 3.2178 |
| M7a | contact_preserving_soft_cone | argmax | 2 | 0.4660 | 15.5488 | 0.2049 | 0.1662 | 3.1162 | 3.0815 |
| M4 | normalized_weighted | topk_sample | 2 | 0.4981 | 15.9520 | 0.2228 | 0.0991 | 3.2629 | 2.8506 |
| M7c | contact_preserving_soft_cone | argmax | 2 | 0.4710 | 16.5451 | 0.2141 | 0.1045 | 3.3326 | 3.2278 |
| M4 | normalized_weighted | argmax | 2 | 0.4279 | 17.6580 | 0.1483 | 0.0813 | 3.5159 | 2.7658 |

## Caveat

This uses ProtenixMini with reduced target length, recycling, sampling, seeds, and steps. It tests whether the instrumentation and update rule behave sensibly with a real structure oracle before larger IL7RA runs.
