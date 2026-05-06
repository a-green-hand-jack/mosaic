# Phase 0 Baseline Plan: Mosaic Binder Design

Date: 2026-05-06

Branch: `baseline/phase0-mosaic-baselines`

## Goal

Establish a reproducible Mosaic-internal baseline surface before adding the SCH optimizer. The immediate objective is not to compare against generation models, BindCraft, or BoltzDesign1. It is to compare update rules for relaxed-sequence optimization under the same Mosaic oracle set.

## Research Question

Can we produce a small, controlled set of binder-design candidates under Mosaic-style objectives and measure whether geometry-aware update rules choose better multi-oracle sequence updates than naive weighted scalarization?

## Hypothesis

The initial Mosaic baselines should produce valid run artifacts and expose gradient conflicts between oracles. Naive weighted scalarization may improve an average loss while locally harming one or more oracle objectives. SCH should reduce this oracle harm without collapsing update size or candidate diversity.

## Blocking Gates

- ACT-005: verify JAX CUDA visibility on at least one Quest A100 and one H100 allocation when available.
- ACT-007: populate or verify `/projects/p32572/Jieke/.cache/boltz` before Boltz-based oracle jobs.
- ACT-006: calibrate exact threshold sources before treating any filter threshold as paper-grade.

If these gates are not complete, run only dry-run provenance, Protenix/MPNN-only diagnostics, or smoke-scale jobs.

## Baseline Matrix

| ID | Method | Role | Initial form | Status |
|---|---|---|---|---|
| M0 | Random sequence / unoptimized PSSM scoring | control baseline | Score random 80-aa binders against IL7RA with the same metrics as designed candidates. | planned |
| M1 | Mosaic Protenix single-oracle update | single-oracle baseline | Adapt `examples/batched_protenix.py`; optimize contact, PAE, ipTM, pLDDT, and ProteinMPNN terms. | planned |
| M2 | Mosaic Boltz or AF2 single-oracle update | symmetric single-oracle baseline | Use a second structure oracle when cache and loader setup are ready. | blocked on ACT-007 if Boltz is used |
| M3 | Naive weighted multi-oracle loss | current Mosaic-style scalarization baseline | Combine structure confidence, MPNN, stability, and sequence sanity terms as a fixed weighted loss. | planned after M1 |
| M4 | Normalized or clipped weighted loss | stronger scalarization baseline | Control for simple scale fixes before claiming geometry is necessary. | planned |
| M5 | Post-hoc reranking | filtering baseline | Test whether gains can be obtained by rescoring M1/M3 candidates without changing updates. | planned after M1/M3 |
| M6 | SCH soft-cone / geometry-guided update | proposed method, first version | Choose update directions using multi-oracle geometric constraints rather than direct scalarization. | planned after M1/M3 instrumentation |
| M7 | SCH hard-cone update | geometry ablation | Test strict common-descent feasibility and conservativeness. | planned after M6 |
| M8 | SCH Fisher/KL variant | geometry ablation | Test whether simplex-aware geometry improves over Euclidean constraints. | planned after M6 |

Excluded from the current stage:

- BoltzGen plus Boltz2 ranking, because it is a generation/ranking workflow rather than a comparable relaxed-sequence update rule.
- Official BindCraft and BoltzDesign1, because the first claim is Mosaic-internal update geometry rather than external binder-design SOTA.

## Initial Targets

| Target | Source | Use |
|---|---|---|
| IL7RA | repo file `IL7RA.cif`; also used in Mosaic examples | first smoke/pilot target |
| PDL1 | repo file `PDL1.pdb` | second pilot target after IL7RA path is stable |

For Phase 0, start with IL7RA only. Expand only after the provenance and candidate scoring schema are stable.

## Metrics

Primary update-level metrics:

- pairwise gradient cosine between oracle gradients
- oracle harm indicator: whether `<g_i, d_t> > 0` for oracle `i`
- oracle harm rate across steps and oracles
- worst-oracle directional derivative: `max_i <g_i, d_t>`
- mean oracle descent: `mean_i <g_i, d_t>`
- cone violation rate for SCH variants
- step norm and sequence entropy change

Final structure-confidence metrics:

- binder mean pLDDT: higher is better
- interface PAE / iPAE: lower is better
- ipTM: higher is better
- pTM: higher is better as a global sanity check

Robustness metrics:

- cross-oracle pass count and worst-oracle score
- optional holdout-oracle ipTM and iPAE for leave-one-oracle diagnostics
- adversarial gap: optimized-oracle score minus excluded-oracle score under a normalized score convention

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
- Same candidate budget and seed list for M0-M8.
- Same post-hoc filters for all methods.
- Same holdout-oracle scoring script for all methods.
- Same oracle call budget or recorded oracle calls for all update-rule comparisons.
- Record GPU family because A100/H100 runtime and memory behavior may differ.

## Seed Plan

- Dry run: no oracle calls; one JSON provenance output.
- Smoke: 2 to 4 seeds, one target, reduced steps.
- Pilot: 8 to 16 seeds per method on IL7RA.
- Paper scale: 50 or more candidates per method per target, after the protocol stabilizes.

## Stop Rules

- If JAX CUDA fails on a compute node, stop and repair the environment before running design.
- If Boltz loaders download during SLURM runtime, stop and move checkpoint hydration into the shared cache setup.
- If M1 cannot produce complete candidate, metric, trajectory, and provenance artifacts, do not start M3/M6.
- If M3 and M6 have indistinguishable update-level metrics, prioritize debugging geometry instrumentation before running more final-candidate seeds.

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
5. Implement M0/M1 scoring and trajectory logging.
6. Add M3 naive weighted loss and M6 SCH update once update-level metrics are reliable.
