# Phase 0 Update-Geometry Diagnostic

Run ID: `phase0_update_geometry_581f872_20260506T105002Z`

## Scope

This is a lightweight Mosaic-internal smoke experiment using sequence-level proxy oracles. It validates the update-geometry logging and gives an initial read on whether geometry-aware updates reduce oracle harm versus scalarized updates.

## Summary

| Method | Harm rate | Worst derivative | Step norm | Final entropy | Final hydrophobic loss | Final solubility loss | Final charge loss | Final trigram loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| soft_cone_correction | 0.245 | 0.0011 | 0.0888 | 2.608 | -0.3676 | 0.0754 | 0.0015 | 3.1257 |
| normalized_weighted | 0.279 | 0.0029 | 0.4735 | 2.031 | -0.3621 | 0.0709 | 0.0038 | 2.6778 |
| naive_weighted | 0.331 | 0.0019 | 0.0962 | 2.305 | -0.4036 | 0.1070 | 0.0032 | 2.7875 |
| single_hydrophobic_proxy | 0.622 | 0.0203 | 0.0945 | 2.169 | -0.8766 | 0.5766 | 0.0042 | 3.4915 |

## Interpretation

- `oracle_harm_rate` is the fraction of oracle gradients for which the actual projected update has positive directional derivative.
- Lower harm rate and lower worst-oracle directional derivative indicate safer multi-oracle updates.
- This run does not yet include structure oracles; it is a fast mechanism check before spending structure-model compute.
