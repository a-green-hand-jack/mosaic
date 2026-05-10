# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_cebb18a_20260510T211432Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M4 / normalized_weighted | 0.000 | -0.0136 | -1.5177 | -0.0815 | -0.0010 | 0.5138 |
| M7c / contact_preserving_soft_cone | 0.050 | 0.0035 | -2.3545 | -0.0643 | -0.0010 | 0.5969 |
| M3 / naive_weighted | 0.200 | 0.0561 | -7.1463 | -0.0064 | -0.0010 | 1.0331 |
| M8a / contact_qp_grid | 0.400 | 0.0456 | -5.4684 | -0.0131 | -0.0010 | 0.7772 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_cebb18a_20260510T211432Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_cebb18a_20260510T211432Z_steps.csv`
