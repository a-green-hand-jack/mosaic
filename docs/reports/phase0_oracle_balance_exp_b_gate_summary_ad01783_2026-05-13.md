# Phase 0 Oracle Balance Gate Summary

Created: `2026-05-13T07:33:29.922805+00:00`

## Gate

- Baseline: `M4`
- Harm tolerance: `0.0`
- Worst-derivative tolerance: `0.0`
- Minimum step-norm ratio: `0.5`

## Overall

| Method | Gate | Targets | Zero-harm targets | Harm | Worst dir | Protenix dir | Boltz2 dir | Step norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M11a / pcgrad_normalized | pass | 5 | 5 | 0.000 | -0.0145 | -2.0539 | -0.0858 | 0.5752 |
| M4 / normalized_weighted | baseline | 5 | 5 | 0.000 | -0.0135 | -1.8767 | -0.0722 | 0.5022 |
| M10a / balanced_zero_harm_cone | fail | 5 | 5 | 0.000 | -0.0028 | -0.4388 | -0.0163 | 0.1098 |
| M7c / contact_preserving_soft_cone | fail | 5 | 1 | 0.080 | 0.0045 | -5.4264 | -0.0666 | 0.8099 |
| M3 / naive_weighted | fail | 5 | 0 | 0.250 | 0.0372 | -7.9825 | -0.0423 | 1.0573 |
| M8a / contact_qp_grid | fail | 5 | 0 | 0.370 | 0.0364 | -7.4735 | -0.0292 | 0.9609 |

## Per-Target

| Target | Method | Zero harm | Harm | Worst dir | Protenix dir | Boltz2 dir | Step norm |
|---|---|---:|---:|---:|---:|---:|---:|
| 1ALU:A | M10a / balanced_zero_harm_cone | True | 0.000 | -0.0029 | -0.1526 | -0.0099 | 0.1035 |
| 1ALU:A | M11a / pcgrad_normalized | True | 0.000 | -0.0175 | -0.8814 | -0.0533 | 0.6147 |
| 1ALU:A | M3 / naive_weighted | False | 0.300 | 0.0628 | -2.2546 | -0.0372 | 0.6967 |
| 1ALU:A | M4 / normalized_weighted | True | 0.000 | -0.0143 | -0.6922 | -0.0445 | 0.4868 |
| 1ALU:A | M7c / contact_preserving_soft_cone | True | 0.000 | -0.0143 | -0.6922 | -0.0445 | 0.4868 |
| 1ALU:A | M8a / contact_qp_grid | False | 0.350 | 0.0305 | -1.3767 | -0.0194 | 0.4949 |
| 2JJS:C | M10a / balanced_zero_harm_cone | True | 0.000 | -0.0028 | -0.3099 | -0.0191 | 0.1108 |
| 2JJS:C | M11a / pcgrad_normalized | True | 0.000 | -0.0137 | -1.4059 | -0.0998 | 0.5701 |
| 2JJS:C | M3 / naive_weighted | False | 0.150 | 0.0341 | -4.2381 | -0.0670 | 0.8951 |
| 2JJS:C | M4 / normalized_weighted | True | 0.000 | -0.0133 | -1.2943 | -0.0849 | 0.5043 |
| 2JJS:C | M7c / contact_preserving_soft_cone | False | 0.050 | 0.0135 | -3.4337 | -0.0905 | 0.7905 |
| 2JJS:C | M8a / contact_qp_grid | False | 0.250 | 0.0363 | -4.2314 | -0.0478 | 0.8826 |
| 4ZQK:A | M10a / balanced_zero_harm_cone | True | 0.000 | -0.0029 | -1.0226 | -0.0111 | 0.1119 |
| 4ZQK:A | M11a / pcgrad_normalized | True | 0.000 | -0.0145 | -4.6448 | -0.0612 | 0.5688 |
| 4ZQK:A | M3 / naive_weighted | False | 0.400 | 0.0132 | -19.2787 | -0.0226 | 1.5523 |
| 4ZQK:A | M4 / normalized_weighted | True | 0.000 | -0.0138 | -4.3839 | -0.0471 | 0.5058 |
| 4ZQK:A | M7c / contact_preserving_soft_cone | False | 0.100 | -0.0055 | -13.7752 | -0.0456 | 1.0728 |
| 4ZQK:A | M8a / contact_qp_grid | False | 0.400 | 0.0143 | -19.2383 | -0.0197 | 1.5418 |
| 4ZQK:B | M10a / balanced_zero_harm_cone | True | 0.000 | -0.0027 | -0.3235 | -0.0225 | 0.1090 |
| 4ZQK:B | M11a / pcgrad_normalized | True | 0.000 | -0.0128 | -1.6323 | -0.1128 | 0.5654 |
| 4ZQK:B | M3 / naive_weighted | False | 0.200 | 0.0096 | -6.7348 | -0.0869 | 1.0913 |
| 4ZQK:B | M4 / normalized_weighted | True | 0.000 | -0.0125 | -1.4540 | -0.0994 | 0.5005 |
| 4ZQK:B | M7c / contact_preserving_soft_cone | False | 0.200 | 0.0253 | -6.7450 | -0.0843 | 1.0907 |
| 4ZQK:B | M8a / contact_qp_grid | False | 0.450 | 0.0523 | -6.7804 | -0.0469 | 1.0889 |
| IL7RA:A | M10a / balanced_zero_harm_cone | True | 0.000 | -0.0028 | -0.3854 | -0.0190 | 0.1137 |
| IL7RA:A | M11a / pcgrad_normalized | True | 0.000 | -0.0141 | -1.7052 | -0.1020 | 0.5568 |
| IL7RA:A | M3 / naive_weighted | False | 0.200 | 0.0665 | -7.4062 | 0.0022 | 1.0511 |
| IL7RA:A | M4 / normalized_weighted | True | 0.000 | -0.0136 | -1.5589 | -0.0849 | 0.5134 |
| IL7RA:A | M7c / contact_preserving_soft_cone | False | 0.050 | 0.0036 | -2.4860 | -0.0682 | 0.6086 |
| IL7RA:A | M8a / contact_qp_grid | False | 0.400 | 0.0488 | -5.7406 | -0.0123 | 0.7965 |

## Interpretation

- `pass` means the method matches the configured update-level gate against the baseline.
- This gate is intentionally update-level only; candidate-level Boltz2 holdout remains a separate decision.
