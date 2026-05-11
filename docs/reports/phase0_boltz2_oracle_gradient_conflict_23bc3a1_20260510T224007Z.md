# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T224007Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M4 / normalized_weighted | 0.000 | -0.0137 | -4.3037 | -0.0491 | 0.0337 | 0.5071 |
| M7c / contact_preserving_soft_cone | 0.100 | -0.0054 | -13.7994 | -0.0450 | 0.0337 | 1.0799 |
| M3 / naive_weighted | 0.400 | 0.0127 | -19.0981 | -0.0194 | 0.0337 | 1.5433 |
| M8a / contact_qp_grid | 0.450 | 0.0146 | -19.0592 | -0.0166 | 0.0337 | 1.5330 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T224007Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T224007Z_steps.csv`
