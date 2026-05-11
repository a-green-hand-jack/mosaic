# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260511T001153Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M7c / contact_preserving_soft_cone | 0.000 | -0.0143 | -0.6873 | -0.0449 | 0.2113 | 0.4867 |
| M4 / normalized_weighted | 0.000 | -0.0143 | -0.6873 | -0.0449 | 0.2113 | 0.4867 |
| M3 / naive_weighted | 0.300 | 0.0644 | -2.2591 | -0.0356 | 0.2113 | 0.6968 |
| M8a / contact_qp_grid | 0.350 | 0.0317 | -1.3651 | -0.0178 | 0.2113 | 0.4921 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260511T001153Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260511T001153Z_steps.csv`
