# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T234048Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M4 / normalized_weighted | 0.000 | -0.0133 | -1.2383 | -0.0864 | 0.1024 | 0.5037 |
| M7c / contact_preserving_soft_cone | 0.050 | 0.0135 | -3.3484 | -0.0938 | 0.1024 | 0.7872 |
| M3 / naive_weighted | 0.150 | 0.0295 | -3.8017 | -0.0745 | 0.1024 | 0.8490 |
| M8a / contact_qp_grid | 0.300 | 0.0316 | -3.8022 | -0.0540 | 0.1024 | 0.8382 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T234048Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T234048Z_steps.csv`
