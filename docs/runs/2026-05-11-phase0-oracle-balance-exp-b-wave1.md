# Phase 0 Oracle Balance EXP-B Wave 1

## Purpose

Test a balanced multi-oracle update rule, not only an opposite-gradient or
conflict detector. The new method should prefer directions that improve the
average normalized first-order signal across oracles while avoiding first-order
harm to any oracle when such a direction is available.

This run compares the new balanced method against the same baselines used in
the Wave 1 conflict map on the same five-target panel.

## Launch

- Started: 2026-05-11 11:08:15 UTC
- Quest node: `qgpu3020`
- GPU class: NVIDIA H100 80GB HBM3
- GPU binding: `CUDA_VISIBLE_DEVICES=0`
- Launcher commit: `82ca381`
- Process PID: `2319628`
- Resource monitor PID: `2319629`
- Main log: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/manual-boltz2-oracle-balance-expb-wave1-qgpu3020-20260511T110814Z.out`
- Resource log: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/resource-boltz2-oracle-balance-expb-wave1-qgpu3020-20260511T110814Z.out`

## Configuration

- Script: `jobs/launch_boltz2_oracle_balance_exp_b_wave1.sh`
- Driver: `scripts/run_boltz2_oracle_gradient_conflict.py`
- Methods: `M3,M4,M7c,M8a,M10a`
- New method: `M10a=balanced_zero_harm_cone`
- Seeds: 4
- Steps per seed: 1
- Binder length: 24
- Protenix recycling steps: 1
- Boltz2 sampling steps: 1
- Boltz2 recycling steps: 1
- Step size: 0.25

## Method Under Test

`M10a` enumerates cone candidates from the available oracle gradients. It first
filters for candidates with no positive first-order oracle harm
(`max_aux_harms=0`, `aux_slack=0.0`). Among feasible candidates it chooses the
direction with the best average normalized directional derivative across
oracles. If no feasible direction exists, it falls back to the candidate with
the fewest harms, then the best worst/mean normalized derivative.

Interpretation target: this tests whether explicit oracle balancing is better
than single-oracle, mean-gradient, or QP-style baseline updates. It should not
be interpreted as requiring strong opposite gradients between Protenix and
Boltz2.

## Targets

| Target | Structure | Chain | Target Length |
| --- | --- | --- | --- |
| `il7ra` | `IL7RA.cif` | `A` | 48 |
| `pdl1` | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb/4ZQK.cif` | `A` | 48 |
| `pd1` | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb/4ZQK.cif` | `B` | 48 |
| `cd47` | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb/2JJS.cif` | `C` | 48 |
| `il6` | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb/1ALU.cif` | `A` | 48 |

## Expected Outputs

Each target invocation writes a report under `docs/reports/` and raw metrics
under `docs/results/`. The log marks target boundaries with `===TARGET ...===`
lines.

## Initial Status

The run started normally on the already allocated idle H100 node `qgpu3020`.
At the first post-launch check, target `il7ra` was in progress and GPU0 held
about 8.8 GiB memory while GPUs 1 and 2 remained idle.
