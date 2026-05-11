# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T220950Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M4 / normalized_weighted | 0.000 | -0.0136 | -1.5122 | -0.0827 | 0.0005 | 0.5138 |
| M7c / contact_preserving_soft_cone | 0.050 | 0.0042 | -2.4563 | -0.0661 | 0.0005 | 0.6116 |
| M3 / naive_weighted | 0.200 | 0.0595 | -7.0620 | -0.0032 | 0.0005 | 1.0367 |
| M8a / contact_qp_grid | 0.400 | 0.0464 | -5.4242 | -0.0136 | 0.0005 | 0.7841 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T220950Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T220950Z_steps.csv`
