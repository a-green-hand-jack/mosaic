# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T053719Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M10a / balanced_zero_harm_cone | 0.000 | -0.0028 | -0.3854 | -0.0190 | -0.0042 | 0.1137 |
| M4 / normalized_weighted | 0.000 | -0.0136 | -1.5589 | -0.0849 | -0.0042 | 0.5134 |
| M11a / pcgrad_normalized | 0.000 | -0.0141 | -1.7052 | -0.1020 | -0.0042 | 0.5568 |
| M7c / contact_preserving_soft_cone | 0.050 | 0.0036 | -2.4860 | -0.0682 | -0.0042 | 0.6086 |
| M3 / naive_weighted | 0.200 | 0.0665 | -7.4062 | 0.0022 | -0.0042 | 1.0511 |
| M8a / contact_qp_grid | 0.400 | 0.0488 | -5.7406 | -0.0123 | -0.0042 | 0.7965 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T053719Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T053719Z_steps.csv`
