# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T062942Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M10a / balanced_zero_harm_cone | 0.000 | -0.0027 | -0.3235 | -0.0225 | 0.0384 | 0.1090 |
| M4 / normalized_weighted | 0.000 | -0.0125 | -1.4540 | -0.0994 | 0.0384 | 0.5005 |
| M11a / pcgrad_normalized | 0.000 | -0.0128 | -1.6323 | -0.1128 | 0.0384 | 0.5654 |
| M7c / contact_preserving_soft_cone | 0.200 | 0.0253 | -6.7450 | -0.0843 | 0.0384 | 1.0907 |
| M3 / naive_weighted | 0.200 | 0.0096 | -6.7348 | -0.0869 | 0.0384 | 1.0913 |
| M8a / contact_qp_grid | 0.450 | 0.0523 | -6.7804 | -0.0469 | 0.0384 | 1.0889 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T062942Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T062942Z_steps.csv`
