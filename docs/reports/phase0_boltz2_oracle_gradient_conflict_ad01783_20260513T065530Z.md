# Protenix-vs-Boltz2 Oracle Gradient Conflict Smoke

Run ID: `phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T065530Z`

## Scope

This reduced Phase 0 experiment directly measures gradient conflict between a Protenix contact oracle and a Boltz2 distogram-contact surrogate. It avoids Boltz2 diffusion/coordinate outputs because `structure_coordinates` are currently non-finite.

## Summary

| Method | Harm rate | Worst derivative | Protenix dir | Boltz2 dir | Protenix/Boltz2 cosine | Step norm |
|---|---:|---:|---:|---:|---:|---:|
| M10a / balanced_zero_harm_cone | 0.000 | -0.0028 | -0.3099 | -0.0191 | 0.0868 | 0.1108 |
| M4 / normalized_weighted | 0.000 | -0.0133 | -1.2943 | -0.0849 | 0.0868 | 0.5043 |
| M11a / pcgrad_normalized | 0.000 | -0.0137 | -1.4059 | -0.0998 | 0.0868 | 0.5701 |
| M7c / contact_preserving_soft_cone | 0.050 | 0.0135 | -3.4337 | -0.0905 | 0.0868 | 0.7905 |
| M3 / naive_weighted | 0.150 | 0.0341 | -4.2381 | -0.0670 | 0.0868 | 0.8951 |
| M8a / contact_qp_grid | 0.250 | 0.0363 | -4.2314 | -0.0478 | 0.0868 | 0.8826 |

## Interpretation

- A negative Protenix/Boltz2 cosine indicates direct gradient disagreement.
- A positive directional derivative means the projected update locally harms that oracle.
- This is a smoke-scale mechanism test; it is not yet a full SCH optimizer run.

## Outputs

- JSON: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T065530Z.json`
- CSV: `docs/results/phase0_boltz2_oracle_gradient_conflict_ad01783_20260513T065530Z_steps.csv`
