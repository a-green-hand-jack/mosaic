# Phase 0 Oracle Balance Gate Summary

Created: `2026-05-12T22:14:27.854007+00:00`

## Gate

- Baseline: `M4`
- Harm tolerance: `0.0`
- Worst-derivative tolerance: `0.0`
- Minimum step-norm ratio: `0.5`

## Overall

| Method | Gate | Targets | Zero-harm targets | Harm | Worst dir | Protenix dir | Boltz2 dir | Step norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M4 / normalized_weighted | baseline | 5 | 5 | 0.000 | -0.0135 | -1.8417 | -0.0719 | 0.5023 |
| M7c / contact_preserving_soft_cone | fail | 5 | 1 | 0.080 | 0.0047 | -5.3903 | -0.0669 | 0.8103 |
| M3 / naive_weighted | fail | 5 | 0 | 0.250 | 0.0353 | -7.7743 | -0.0439 | 1.0425 |
| M8a / contact_qp_grid | fail | 5 | 0 | 0.390 | 0.0354 | -7.2694 | -0.0301 | 0.9464 |

## Per-Target

| Target | Method | Zero harm | Harm | Worst dir | Protenix dir | Boltz2 dir | Step norm |
|---|---|---:|---:|---:|---:|---:|---:|
| 1ALU:A | M3 / naive_weighted | False | 0.300 | 0.0644 | -2.2591 | -0.0356 | 0.6968 |
| 1ALU:A | M4 / normalized_weighted | True | 0.000 | -0.0143 | -0.6873 | -0.0449 | 0.4867 |
| 1ALU:A | M7c / contact_preserving_soft_cone | True | 0.000 | -0.0143 | -0.6873 | -0.0449 | 0.4867 |
| 1ALU:A | M8a / contact_qp_grid | False | 0.350 | 0.0317 | -1.3651 | -0.0178 | 0.4921 |
| 2JJS:C | M3 / naive_weighted | False | 0.150 | 0.0295 | -3.8017 | -0.0745 | 0.8490 |
| 2JJS:C | M4 / normalized_weighted | True | 0.000 | -0.0133 | -1.2383 | -0.0864 | 0.5037 |
| 2JJS:C | M7c / contact_preserving_soft_cone | False | 0.050 | 0.0135 | -3.3484 | -0.0938 | 0.7872 |
| 2JJS:C | M8a / contact_qp_grid | False | 0.300 | 0.0316 | -3.8022 | -0.0540 | 0.8382 |
| 4ZQK:A | M3 / naive_weighted | False | 0.400 | 0.0127 | -19.0981 | -0.0194 | 1.5433 |
| 4ZQK:A | M4 / normalized_weighted | True | 0.000 | -0.0137 | -4.3037 | -0.0491 | 0.5071 |
| 4ZQK:A | M7c / contact_preserving_soft_cone | False | 0.100 | -0.0054 | -13.7994 | -0.0450 | 1.0799 |
| 4ZQK:A | M8a / contact_qp_grid | False | 0.450 | 0.0146 | -19.0592 | -0.0166 | 1.5330 |
| 4ZQK:B | M3 / naive_weighted | False | 0.200 | 0.0101 | -6.6506 | -0.0866 | 1.0868 |
| 4ZQK:B | M4 / normalized_weighted | True | 0.000 | -0.0124 | -1.4669 | -0.0966 | 0.5002 |
| 4ZQK:B | M7c / contact_preserving_soft_cone | False | 0.200 | 0.0257 | -6.6598 | -0.0849 | 1.0861 |
| 4ZQK:B | M8a / contact_qp_grid | False | 0.450 | 0.0525 | -6.6960 | -0.0486 | 1.0846 |
| IL7RA:A | M3 / naive_weighted | False | 0.200 | 0.0595 | -7.0620 | -0.0032 | 1.0367 |
| IL7RA:A | M4 / normalized_weighted | True | 0.000 | -0.0136 | -1.5122 | -0.0827 | 0.5138 |
| IL7RA:A | M7c / contact_preserving_soft_cone | False | 0.050 | 0.0042 | -2.4563 | -0.0661 | 0.6116 |
| IL7RA:A | M8a / contact_qp_grid | False | 0.400 | 0.0464 | -5.4242 | -0.0136 | 0.7841 |

## Interpretation

- `pass` means the method matches the configured update-level gate against the baseline.
- This gate is intentionally update-level only; candidate-level Boltz2 holdout remains a separate decision.
