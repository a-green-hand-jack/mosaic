# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T060330Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M10a / balanced_zero_harm_cone | 0.000 | -0.0029 | -1.0226 | -0.0111 | 0.0451 | 0.1119 |
| M4 / normalized_weighted | 0.000 | -0.0138 | -4.3839 | -0.0471 | 0.0451 | 0.5058 |
| M11a / pcgrad_normalized | 0.000 | -0.0145 | -4.6448 | -0.0612 | 0.0451 | 0.5688 |
| M7c / contact_preserving_soft_cone | 0.100 | -0.0055 | -13.7752 | -0.0456 | 0.0451 | 1.0728 |
| M8a / contact_qp_grid | 0.400 | 0.0143 | -19.2383 | -0.0197 | 0.0451 | 1.5418 |
| M3 / naive_weighted | 0.400 | 0.0132 | -19.2787 | -0.0226 | 0.0451 | 1.5523 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T060330Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T060330Z_steps.csv`
