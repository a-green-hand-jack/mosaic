# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T072100Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M10a / balanced_zero_harm_cone | 0.000 | -0.0029 | -0.1526 | -0.0099 | 0.2218 | 0.1035 |
| M7c / contact_preserving_soft_cone | 0.000 | -0.0143 | -0.6922 | -0.0445 | 0.2218 | 0.4868 |
| M4 / normalized_weighted | 0.000 | -0.0143 | -0.6922 | -0.0445 | 0.2218 | 0.4868 |
| M11a / pcgrad_normalized | 0.000 | -0.0175 | -0.8814 | -0.0533 | 0.2218 | 0.6147 |
| M3 / naive_weighted | 0.300 | 0.0628 | -2.2546 | -0.0372 | 0.2218 | 0.6967 |
| M8a / contact_qp_grid | 0.350 | 0.0305 | -1.3767 | -0.0194 | 0.2218 | 0.4949 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T072100Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T072100Z_steps.csv`
