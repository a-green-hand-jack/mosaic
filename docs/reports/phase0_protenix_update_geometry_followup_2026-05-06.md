# Phase 0 Protenix Update-Geometry Follow-up

Date: 2026-05-06

## Scope

This follow-up extends the minimum Protenix-backed smoke with three small runs:

- H100 portability smoke: `steps=1`, `num_seeds=2`, run ID `phase0_protenix_update_geometry_c9af27f_20260506T114159Z`.
- A100 small multi-step run: `steps=3`, `num_seeds=2`, run ID `phase0_protenix_update_geometry_c9af27f_20260506T115609Z`.
- H100 small multi-step run: `steps=3`, `num_seeds=2`, run ID `phase0_protenix_update_geometry_c9af27f_20260506T141236Z`.

Both runs use ProtenixMini binder-target contact plus solubility, charge, and trigram sequence sanity losses on a reduced IL7RA setup.

## Summary

| Run | Method | Harm rate | Worst derivative | Step norm | Final entropy |
|---|---|---:|---:|---:|---:|
| H100 `steps=1,seeds=2` | single_protenix_contact | 0.375 | 0.0677 | 1.2532 | 2.739 |
| H100 `steps=1,seeds=2` | naive_weighted | 0.250 | 0.0248 | 1.2501 | 2.739 |
| H100 `steps=1,seeds=2` | normalized_weighted | 0.000 | -0.0110 | 0.4825 | 2.739 |
| H100 `steps=1,seeds=2` | soft_cone_correction | 0.000 | -0.0110 | 0.4825 | 2.739 |
| A100 `steps=3,seeds=2` | single_protenix_contact | 0.333 | 0.0317 | 1.0546 | 2.060 |
| A100 `steps=3,seeds=2` | naive_weighted | 0.208 | 0.0108 | 0.8964 | 2.148 |
| A100 `steps=3,seeds=2` | normalized_weighted | 0.000 | -0.0090 | 0.4345 | 2.517 |
| A100 `steps=3,seeds=2` | soft_cone_correction | 0.000 | -0.0090 | 0.4345 | 2.517 |
| H100 `steps=3,seeds=2` | single_protenix_contact | 0.292 | 0.0311 | 1.3101 | 2.010 |
| H100 `steps=3,seeds=2` | naive_weighted | 0.208 | 0.0127 | 0.9539 | 2.114 |
| H100 `steps=3,seeds=2` | normalized_weighted | 0.000 | -0.0089 | 0.4312 | 2.517 |
| H100 `steps=3,seeds=2` | soft_cone_correction | 0.000 | -0.0089 | 0.4312 | 2.517 |

## Interpretation

The positive mechanism signal from the minimum smoke reproduced on both A100 and H100:

- Single Protenix-contact updates harm other oracles in these small settings.
- Naive weighted scalarization still leaves positive worst-oracle directional derivatives.
- Normalized weighted and soft-cone correction remove measured first-order oracle harm in these runs.

The A100 and H100 `steps=3,seeds=2` runs agree closely, which is useful operationally: the uv/JAX/Protenix environment is portable across both GPU classes for this reduced setup. The current soft-cone solution matches normalized weighted in these small Protenix runs, which suggests that gradient normalization alone is a strong baseline and must remain in the main comparison.

The H100 `steps=3,seeds=2` detached run took 10:01 wall-clock time with `/usr/bin/time -v`; the H100 `steps=1,seeds=2` run took 4:13. This makes larger seed sweeps feasible but expensive enough to justify keeping the target/binder sizes reduced until candidate-level scoring is added.

## Caveats

These are still reduced ProtenixMini runs, not final binder-design benchmarks. The next gate should test whether the update-geometry gains persist under more seeds and whether they translate to final candidate-level structure and sequence scores.
