# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T231100Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M4 / normalized_weighted | 0.000 | -0.0124 | -1.4669 | -0.0966 | 0.0469 | 0.5002 |
| M7c / contact_preserving_soft_cone | 0.200 | 0.0257 | -6.6598 | -0.0849 | 0.0469 | 1.0861 |
| M3 / naive_weighted | 0.200 | 0.0101 | -6.6506 | -0.0866 | 0.0469 | 1.0868 |
| M8a / contact_qp_grid | 0.450 | 0.0525 | -6.6960 | -0.0486 | 0.0469 | 1.0846 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T231100Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T231100Z_steps.csv`
