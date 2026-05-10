# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_66634b1_20260510T204158Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M3 / naive_weighted | 0.000 | -0.0007 | -4.0062 | -0.0257 | 0.2113 | 0.9049 |
| M4 / normalized_weighted | 0.000 | -0.0065 | -1.2774 | -0.0301 | 0.2113 | 0.5278 |
| M7c / contact_preserving_soft_cone | 0.200 | 0.0430 | -4.0282 | -0.0258 | 0.2113 | 0.8976 |
| M8a / contact_qp_grid | 0.200 | 0.0430 | -4.0282 | -0.0258 | 0.2113 | 0.8976 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_66634b1_20260510T204158Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_66634b1_20260510T204158Z_steps.csv`
