# Phase 0 Baseline Plan: Mosaic Binder Design

Date: 2026-05-06

Branch: `baseline/phase0-mosaic-baselines`

## Goal

Establish a reproducible baseline surface for SCH-BinderDesign before adding the SCH optimizer. The immediate objective is not to claim improvement over BindCraft or BoltzDesign1; it is to make Mosaic-derived baseline runs executable, logged, and comparable on Quest.

## Research Question

Can we produce a small, controlled set of binder-design candidates under Mosaic-style objectives and score them with the same holdout-oracle and filter metrics that will later define SCH success?

## Hypothesis

The initial Mosaic baselines should produce valid run artifacts and some candidates with reasonable training-oracle confidence. Their main expected weakness is a holdout-oracle gap: candidates optimized against one oracle may score worse under alternate oracles or post-hoc filters.

## Blocking Gates

- ACT-005: verify JAX CUDA visibility on at least one Quest A100 and one H100 allocation when available.
- ACT-007: populate or verify `/projects/p32572/Jieke/.cache/boltz` before Boltz/BoltzGen jobs.
- ACT-006: calibrate exact threshold sources before treating any filter threshold as paper-grade.

If these gates are not complete, run only dry-run provenance, Protenix/MPNN-only diagnostics, or smoke-scale jobs.

## Baseline Matrix

| ID | Baseline | Role | Initial form | Status |
|---|---|---|---|---|
| B0 | Random sequence / unoptimized PSSM scoring | control baseline | Score random 80-aa binders against IL7RA with the same metrics as designed candidates. | planned |
| B1 | Mosaic Protenix single-oracle hallucination | nearest runnable Mosaic baseline | Adapt `examples/batched_protenix.py`; optimize contact, PAE, ipTM, pLDDT, and ProteinMPNN terms. | planned |
| B2 | BoltzGen plus Boltz2 ranking | direct generative baseline | Adapt `examples/boltzgen_pipeline.py`; generate candidates, inverse-fold with ProteinMPNN, rank with Boltz2 confidence terms. | blocked on Boltz cache |
| B3 | Mosaic weighted composite objective | nearest previous method to SCH | Combine train-oracle confidence, MPNN, stability, sequence sanity, and monomer checks as a fixed weighted loss. | planned after B1/B2 |
| B4 | Official BindCraft | domain-required external baseline | Run official pipeline on matched targets and budget. | later paper baseline |
| B5 | Official BoltzDesign1 | direct competitor external baseline | Run official or author-faithful pipeline on matched targets and budget. | later paper baseline |

## Initial Targets

| Target | Source | Use |
|---|---|---|
| IL7RA | repo file `IL7RA.cif`; also used in Mosaic examples | first smoke/pilot target |
| PDL1 | repo file `PDL1.pdb` | second pilot target after IL7RA path is stable |

For Phase 0, start with IL7RA only. Expand only after the provenance and candidate scoring schema are stable.

## Metrics

Primary structure-confidence metrics:

- binder mean pLDDT: higher is better
- interface PAE / iPAE: lower is better
- ipTM: higher is better
- pTM: higher is better as a global sanity check

Robustness metrics:

- holdout-oracle ipTM and iPAE
- adversarial gap: train-oracle score minus holdout-oracle score under a normalized score convention
- cross-oracle agreement: fraction of candidates passing thresholds under two or more independent oracles

Sequence and binder sanity metrics:

- cysteine count
- maximum hydrophobic stretch
- net charge and approximate pI if available
- N-glycosylation motif count
- ProteinMPNN inverse-folding score when structure is available
- monomer confidence or monomer RMSD when available

Resource metrics:

- GPU class and node
- wall-clock runtime
- peak GPU memory if logged
- oracle calls or forward passes
- git commit and config path

## Controls

- Same target structure and chain selection within a comparison.
- Same binder length unless a baseline requires a documented alternative.
- Same candidate budget and seed list for B0/B1/B3.
- Same post-hoc filters for all methods.
- Same holdout-oracle scoring script for all methods.
- Record GPU family because A100/H100 runtime and memory behavior may differ.

## Seed Plan

- Dry run: no oracle calls; one JSON provenance output.
- Smoke: 2 to 4 seeds, one target, reduced steps.
- Pilot: 8 to 16 seeds per method on IL7RA.
- Paper scale: 50 or more candidates per method per target, after the protocol stabilizes.

## Stop Rules

- If JAX CUDA fails on a compute node, stop and repair the environment before running design.
- If Boltz/BoltzGen loaders download during SLURM runtime, stop and move checkpoint hydration into the shared cache setup.
- If B1 cannot produce complete candidate, metric, and provenance artifacts, do not start B2/B3.
- If B0 and B1 are indistinguishable under holdout metrics, prioritize instrumentation and metric sanity over more seeds.

## Expected Artifacts

- `docs/runs/<run_id>.json`: run-level provenance and config snapshot.
- `docs/results/<run_id>_candidates.jsonl`: candidate-level sequence, source method, seed, and metrics.
- `docs/reports/<run_id>_summary.md`: short interpretation after a completed pilot.

Raw logs, checkpoints, and large structures should stay under `/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs`, `/projects/p32572/Jieke/.cache`, or a documented durable output directory, not in git.

## Immediate Execution Order

1. Run `scripts/run_baseline_pilot.py --dry-run` locally to validate provenance output.
2. Push this branch and create the matching Quest worktree.
3. Run the JAX CUDA smoke job on A100 and H100 when available.
4. Run the baseline pilot job in dry-run mode from Quest.
5. Hydrate Boltz cache and then enable B1 smoke.
