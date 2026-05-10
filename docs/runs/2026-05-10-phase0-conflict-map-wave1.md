# Phase 0 Conflict Map Wave 1

## Purpose

Measure whether Protenix/Boltz2/sequence-oracle update directions disagree across
a broader target slice than the initial IL7RA-only smoke tests.

## Launch

- Started: 2026-05-10 21:39:09 UTC
- Quest node: `qgpu2013`
- GPU binding: `CUDA_VISIBLE_DEVICES=0`
- Launcher commit: `23bc3a1`
- Process PID: `1807153`
- Resource monitor PID: `1807674`
- Main log: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/manual-boltz2-oracle-conflict-map-wave1-qgpu2013-20260510T213909Z.out`
- Resource log: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/resource-boltz2-oracle-conflict-map-wave1-qgpu2013-20260510T213909Z.out`

## Configuration

- Script: `jobs/launch_boltz2_oracle_conflict_map_wave1.sh`
- Driver: `scripts/run_boltz2_oracle_gradient_conflict.py`
- Methods: `M3,M4,M7c,M8a`
- Seeds: 4
- Steps per seed: 1
- Binder length: 24
- Protenix recycling steps: 1
- Boltz2 sampling steps: 1
- Boltz2 recycling steps: 1
- Step size: 0.25

## Targets

| Target | Structure | Chain | Target Length |
| --- | --- | --- | --- |
| `il7ra` | `IL7RA.cif` | `A` | 48 |
| `pdl1` | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb/4ZQK.cif` | `A` | 48 |
| `pd1` | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb/4ZQK.cif` | `B` | 48 |
| `cd47` | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb/2JJS.cif` | `C` | 48 |
| `il6` | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb/1ALU.cif` | `A` | 48 |

## Expected Outputs

Each target invocation writes a report under `docs/reports/` and raw metrics under
`docs/results/`. The log marks target boundaries with `===TARGET ...===` lines.

## Notes

The job is intentionally launched on an already allocated idle Quest GPU node
instead of submitted through SLURM, because the queued jobs were blocked by
`QOSMaxGRESPerUser` while `qgpu2013` had idle A100 GPUs available.
