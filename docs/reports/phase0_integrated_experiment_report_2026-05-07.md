# Phase 0 Integrated Experiment Report

Date: 2026-05-07

## Summary

- Question: Can Mosaic's multi-oracle binder hallucination be improved by geometry-aware sequence updates, and what blocks the relaxed optimization gains from becoming good discrete candidates?
- Setup: Reduced IL7RA ProtenixMini experiments on Quest A100/H100, mostly `target_length=48`, `binder_length=24`, `steps=3`, `num_seeds=2`, with Protenix contact plus sequence sanity losses.
- Headline result: The update-conflict motivation is supported, and contact-preserving geometry can improve relaxed/soft terminal sequences. The candidate-quality claim is not yet solved because discretization is the main bottleneck.
- Key answer: Cold top-k handoff can recover strong M7c discrete candidates, but naive optimizer-side post-update entropy annealing does not reproduce that benefit.
- Next step: ACT-015 should test straight-through hard candidate scoring or a weaker constrained discreteness regularizer, with M3 top-k as the main baseline.

## Direct Answer To The Four Questions

### What was the goal?

The overall goal was to test the project's core mechanism hypothesis in a cheap, controllable setting before spending on larger benchmarks:

> Mosaic already supports multiple differentiable oracles, but currently combines them too crudely. A more geometry-aware update rule should reduce oracle harm and, if discretization is handled correctly, produce better binder candidates.

In practice, Phase 0 asked three nested questions:

1. Do multi-oracle gradient conflicts actually exist in Mosaic/Protenix?
2. Do safer/contact-preserving updates improve relaxed or final candidate quality?
3. If relaxed gains exist but disappear after argmax, can we recover them through better discrete handoff or optimizer-side discreteness?

### Did we get the answers?

Partially, and the boundary is now clear.

- Yes: multi-oracle update conflict exists in these reduced runs.
- Yes: contact-preserving geometry can improve relaxed soft terminal interface metrics.
- Yes: argmax discretization can destroy the relaxed-sequence gain.
- Yes: cold top-k handoff can recover strong M7c discrete candidates.
- No: we have not yet proven robust benchmark-scale candidate superiority over M3 naive weighted.
- No: naive post-update entropy annealing is not a good optimizer-side repair.

### How did we find the answers?

We ran a staged experiment chain:

1. Sequence proxy diagnostics to validate update-geometry instrumentation.
2. ProtenixMini smoke and A100/H100 follow-up runs to confirm structure-oracle gradient paths.
3. Candidate scoring diagnostics to compare update-level safety against final interface metrics.
4. Contact-preserving update sweeps to preserve the Protenix contact objective while controlling sequence sanity losses.
5. Soft-vs-argmax, top-k handoff, sampling-temperature, and entropy-annealing ablations to isolate the discretization bottleneck.

Every major run wrote raw JSON/CSV artifacts and a report under `docs/results/` and `docs/reports/`, with code commits recorded in project memory.

### Are the results correct?

They are correct as reduced-run mechanism evidence, not as final binder-design benchmark evidence.

The results are internally auditable because:

- runs exited successfully on Quest GPU nodes;
- raw metrics are preserved in CSV/JSON;
- code commits, run IDs, and reports are recorded;
- several conclusions were tested with follow-up sensitivity runs rather than accepted from one positive result.

The main limits are:

- most Protenix runs use only two seeds;
- target and binder lengths are reduced;
- ProtenixMini is the only structure oracle in this chain;
- no Boltz/AF2 cross-oracle holdout or full pass-rate threshold evaluation has been run yet.

## 1. Experiment Motivation

The original project motivation is not "beat BindCraft immediately." The immediate question is more specific:

> In Mosaic's multi-oracle relaxed sequence optimization, can we use oracle geometry more precisely than naive scalarization?

This matters because binder hallucination pipelines often score/filter candidates with scalar metrics such as pLDDT, interface PAE, ipTM, contact loss, sequence naturalness, charge, and hydrophobicity. If a single weighted objective improves one metric by locally harming another, then a geometry-aware update rule could be a meaningful contribution. But if update-level improvements do not translate to final discrete candidates, the method claim is not enough for a paper.

## 2. Experiment Setup

| Item | Value |
|---|---|
| Project phase | Phase 0, Mosaic-internal update-geometry validation |
| Target | Reduced IL7RA |
| Main structure oracle | ProtenixMini binder-target contact |
| Auxiliary losses | Solubility/hydrophobicity, charge target, trigram naturalness |
| Main baseline | `M3` naive weighted scalarization |
| Geometry variants | `M4` normalized weighted, `M6` soft cone, `M7a/M7b/M7c` contact-preserving cone |
| Discretization variants | argmax, soft scoring, top-k sampling/reranking |
| Optimizer-side discreteness attempt | `M7d/M7e` entropy-annealed contact-preserving variants |
| Typical reduced settings | `target_length=48`, `binder_length=24`, `steps=3`, `num_seeds=2`, recycling/sampling steps 1 |
| Server | Quest A100/H100; latest runs on `qgpu3019` H100 |
| Latest code commit | `2b6fb58` on `baseline/phase0-mosaic-baselines` |

## 3. Core Algorithm Or Method

The experiment does not yet implement a full final SCH optimizer. It tests candidate pieces of the method.

The basic loop is:

1. Maintain a relaxed sequence distribution over amino-acid tokens.
2. Evaluate multiple oracle losses and gradients.
3. Choose an update direction by one of several rules.
4. Project the update back to the simplex.
5. Score terminal sequences as soft, argmax, or sampled top-k candidates.

The important method variants are:

| ID | Method | Purpose |
|---|---|---|
| M1 | Single primary oracle | Checks what happens when only Protenix contact is optimized. |
| M3 | Naive weighted | Main baseline: sum all losses with equal weights. |
| M4 | Normalized weighted | Controls for gradient scale without cone constraints. |
| M6 | Soft cone correction | Searches for directions with less oracle harm. |
| M7a/M7c | Contact-preserving soft cone | Preserves primary Protenix contact descent while limiting auxiliary harms. |
| M7d/M7e | Entropy-annealed contact-preserving | Tests whether post-update sharpening can make M7c discretization-ready. |

## 4. Metrics

| Metric | Definition | Direction | Why it matters |
|---|---|---|---|
| Oracle harm rate | Fraction of oracle gradients with positive directional derivative under the actual update | Lower | Tests whether an update locally harms some objectives. |
| Worst directional derivative | Worst per-oracle local derivative under the actual update | Lower | Measures worst-case local oracle harm. |
| Step norm | Norm of the actual update | Context-dependent | Detects overly small or overly disruptive updates. |
| Sequence entropy | Mean entropy of relaxed sequence distribution | Context-dependent | Lower means closer to discrete; too low can harm optimization. |
| Protenix contact loss | Binder-target contact loss | Lower | Primary reduced structure objective. |
| BT PAE | Binder-target predicted aligned error | Lower | Main interface-quality metric. |
| BT ipTM | Binder-target interface predicted TM-score | Higher | Main interface-confidence metric. |
| IPSAE min | Interface PAE-derived score | Higher | Secondary interface-quality metric. |
| pLDDT | Predicted local confidence | Higher | Candidate structural confidence. |
| Trigram loss / hydrophobicity / charge | Sequence sanity losses | Lower or target range | Detects obviously unnatural or problematic candidates. |

## 5. Results By Subquestion

### Q1. Do multi-oracle update conflicts exist?

Goal: Show whether Mosaic's multi-oracle setting has real local gradient conflicts, rather than assuming the method motivation.

Answer: Yes, in both proxy and reduced Protenix runs.

How we found it:

- EVD-008 sequence proxy diagnostic measured oracle harm for single-oracle, naive weighted, normalized weighted, and soft-cone updates.
- EVD-010 Protenix A100/H100 follow-up repeated the update-harm test with a structure oracle.

Key results:

| Evidence | Setting | Main observation |
|---|---|---|
| EVD-008 | Sequence proxy oracles | Harm rates: single 0.622, naive 0.331, normalized 0.279, soft-cone 0.245. |
| EVD-010 | Protenix reduced runs | Single-Protenix and naive weighted had positive harm; normalized/soft-cone often reduced measured harm to 0. |

Are the results correct?

Yes for the mechanism claim. The same instrumentation was exercised on proxy losses and ProtenixMini. The limitation is scale: these are reduced target/seed settings, not full benchmark evidence.

### Q2. Does safer update geometry automatically improve candidates?

Goal: Test whether lower oracle harm is enough to produce better final binder candidates.

Answer: No.

How we found it:

- EVD-011 added candidate-level argmax holdout scoring after update-geometry runs.

Key result:

Normalized weighted and soft-cone updates improved update safety and sequence sanity, but single/naive methods had better argmax interface metrics in the first candidate-level scoring runs.

Are the results correct?

Yes as a warning against overclaiming. This is one of the most important controls: update-level safety and candidate-level quality are separate gates.

### Q3. Can contact-preserving geometry recover interface quality?

Goal: Repair the failure in Q2 by preserving Protenix contact descent while still limiting auxiliary harm.

Answer: Partially.

How we found it:

- EVD-012 implemented a contact-preserving update.
- ACT-010 swept contact-preserving settings and compared soft terminal scoring against argmax.

Key results:

| Experiment | Main answer |
|---|---|
| EVD-012 | Contact-preserving updates improved candidate interface metrics over normalized/soft-cone while keeping lower update harm than naive, but still did not beat naive/single on final interface metrics. |
| ACT-010 | M7c was strongest under soft terminal scoring: BT PAE 8.72, BT ipTM 0.510, pLDDT 0.727. After argmax, most of this advantage disappeared. |

Are the results correct?

Yes for the relaxed-sequence mechanism. The soft-vs-argmax split directly shows that contact-preserving geometry can help the relaxed terminal sequence, but argmax can erase the gain.

### Q4. Is argmax discretization the bottleneck?

Goal: Determine whether M7c's soft advantage is recoverable as discrete candidates, or whether it is only an artifact of soft scoring.

Answer: Argmax is a bottleneck, but not the only issue.

How we found it:

- ACT-011 added matched-budget top-k sampling and BT PAE reranking.

Key result:

At `top_k=4`, four samples per method/seed, and BT PAE reranking, M7c top-k candidates achieved BT PAE 11.96, BT ipTM 0.431, pLDDT 0.595, and contact loss 2.81. This beat M7c argmax and naive weighted top-k/argmax in that run.

Are the results correct?

Yes, but initially not robust enough. ACT-011 established that argmax is not the only viable handoff and that M7c can produce recoverable discrete candidates. It did not prove general superiority.

### Q5. Is the ACT-011 top-k result robust to candidate budget and reranking metric?

Goal: Check whether the positive ACT-011 result survives a stricter sensitivity test.

Answer: No, not at sampling temperature `1.0`.

How we found it:

- ACT-012 reanalyzed top-k budgets `1,4,8` and rerank metrics BT PAE, BT ipTM, and contact.

Key result:

M3 naive weighted dominated across tested budgets and rerank metrics. Under BT PAE reranking at budget 8, M3 had BT PAE 11.81 and BT ipTM 0.443, while M7c had BT PAE 14.72 and BT ipTM 0.243.

Are the results correct?

Yes as a blocker for scale-up. ACT-012 prevents us from over-interpreting ACT-011. The limitation is that it is still a reduced two-seed ProtenixMini run, but as a go/no-go gate it is sufficient to block immediate scaling.

### Q6. Does colder terminal sampling recover the M7c discrete-candidate signal?

Goal: Test whether ACT-012 failed because terminal top-k sampling was too broad.

Answer: Yes, colder handoff is an important control knob.

How we found it:

- ACT-013 repeated the matched top-k analysis with sampling temperatures `0.5` and `0.25`, using ACT-012 temperature `1.0` as the reference.

Key results:

BT PAE rerank, top-k candidates:

| Temperature | Budget | M3 BT PAE | M7c BT PAE | Winner |
|---:|---:|---:|---:|---|
| 0.5 | 8 | 12.7383 | 11.9860 | M7c |
| 0.25 | 1 | 15.2888 | 14.2063 | M7c |
| 0.25 | 4 | 15.1737 | 11.5974 | M7c |
| 0.25 | 8 | 13.9437 | 11.5974 | M7c |

At temperature `0.25`, M7c also beat M3 at budget 8 under BT ipTM and contact reranking.

Are the results correct?

Yes for the handoff claim: the same analysis pipeline was run with a controlled temperature change. The caveat is important: soft terminal scores still favored M3, so this proves a cold discrete handoff effect, not a fully solved optimizer.

### Q7. Can we move the cold-handoff benefit inside the optimizer by entropy annealing?

Goal: Convert ACT-013's post-hoc cold handoff into an optimizer-side method.

Answer: Not with the naive post-update sharpening implementation.

How we found it:

- ACT-014 added M7d/M7e, which apply post-update sharpening to final temperatures `0.5` and `0.25`.
- The actual post-sharpening update `x_new - x` was used for update-harm logging.

Key results:

| Method | Harm rate | Final entropy | Final contact loss | Budget-8 BT PAE top-k |
|---|---:|---:|---:|---:|
| M7c | 0.0833 | 2.2999 | 2.7127 | 12.1202 |
| M7d | 0.1667 | 1.3308 | 2.9184 | 15.0673 |
| M3 | 0.2083 | 2.1928 | 2.3890 | 10.6019 |
| M7e | 0.2500 | 0.5259 | 3.0914 | 13.5866 |

The annealed methods were colder, but not better. M7e was especially too aggressive: lower entropy, higher harm, worse contact loss.

Are the results correct?

Yes as a negative result for this implementation. The result is internally consistent: entropy decreased, step norms increased, update safety worsened, and candidate quality did not improve. It rejects naive per-step sharpening, not all optimizer-side discreteness methods.

## 6. How To Read The Tables

There are no figures in this integrated report. The tables should be read as reduced-run decision evidence:

- update-level rows answer whether the optimizer step is locally safe;
- candidate rows answer whether terminal sequences score well after soft/argmax/top-k handoff;
- lower BT PAE and higher BT ipTM are the most relevant interface-quality signals;
- M3 is the main candidate-quality baseline, while M7c/M7d/M7e test increasingly geometry-aware or discretization-aware variants.

## 7. Interpretation

The main scientific interpretation is:

1. The project motivation is valid: multi-oracle conflict exists and naive scalarization is not the whole story.
2. The first version of geometry-aware updates helps update safety and can improve relaxed terminal interface metrics.
3. Candidate-level gains depend strongly on discretization.
4. Cold top-k handoff is a useful evaluation and candidate-generation rule.
5. The optimizer-side repair needs to be gentler than post-update entropy sharpening.

This means the project should not yet claim that SCH-style updates improve binder-design pass rates. The current defensible claim is narrower:

> In reduced Mosaic/Protenix experiments, geometry-aware contact-preserving updates expose a real relaxed-to-discrete handoff problem; cold top-k handoff can recover M7c candidates, but naive entropy annealing is not sufficient to make the optimizer itself discretization-ready.

## 8. Conclusion And Discussion

We got useful answers, including negative answers.

What is proven enough to keep going:

- Multi-oracle gradient conflict exists.
- Update-level safety can be improved.
- Contact-preserving updates are more promising than generic harm minimization.
- Argmax is too brittle as the only terminal handoff.
- Cold top-k handoff is a real, measurable control.

What is not proven:

- Robust candidate-level superiority over M3.
- Cross-oracle robustness.
- Benchmark-scale pass-rate gains.
- That entropy annealing is the right optimizer-side mechanism.

The project is now in Phase 0 method revision, not Phase 1 scaling. More targets/seeds would be premature until a revised method beats M3 under a fixed matched-budget discrete handoff.

## 9. Limitations And Caveats

- Reduced target and binder lengths are used.
- Most Protenix runs have two seeds.
- ProtenixMini is the only structure oracle in this chain.
- The current score metrics are in silico diagnostics, not wet-lab validation.
- Candidate pools vary across runs, so single-run wins are not sufficient.
- The top-k handoff uses additional oracle calls; comparisons must stay budget-matched.
- Official BindCraft/BoltzDesign1 baselines remain deferred.

## 10. Next Steps

1. Implement ACT-015:
   - straight-through hard candidate scoring, or
   - a weak terminal discreteness regularizer with explicit contact-descent constraints.
2. Keep M3 top-k as the main reduced-run baseline.
3. Keep M7c cold top-k handoff as the positive handoff reference.
4. Keep M7d/M7e as negative controls, not methods to scale.
5. Only after a revised method beats M3 under matched budget should we scale seeds, targets, lengths, or add cross-oracle Boltz/AF2 holdout.

## Reproducibility Notes

Main code branch:

- `baseline/phase0-mosaic-baselines`
- Latest result commit: `2b6fb58`

Main reports:

- `docs/reports/phase0_update_geometry_581f872_20260506T105002Z.md`
- `docs/reports/phase0_protenix_update_geometry_followup_2026-05-06.md`
- `docs/reports/phase0_protenix_candidate_holdout_2026-05-06.md`
- `docs/reports/phase0_contact_preserving_update_2026-05-06.md`
- `docs/reports/phase0_act010_contact_sweep_2026-05-07.md`
- `docs/reports/phase0_act011_topk_handoff_2026-05-07.md`
- `docs/reports/phase0_act012_topk_sensitivity_2026-05-07.md`
- `docs/reports/phase0_act013_terminal_temperature_2026-05-07.md`
- `docs/reports/phase0_act014_entropy_annealing_2026-05-07.md`

Main run IDs:

- `phase0_protenix_update_geometry_f092264_20260507T042412Z` for ACT-010
- `phase0_protenix_update_geometry_09ed71d_20260507T045421Z` for ACT-011
- `phase0_protenix_update_geometry_cc6864e_20260507T051752Z` for ACT-012
- `phase0_protenix_update_geometry_f89ee15_20260507T054055Z` and `phase0_protenix_update_geometry_f89ee15_20260507T054922Z` for ACT-013
- `phase0_protenix_update_geometry_22cacaf_20260507T061105Z` for ACT-014

Quest execution:

- Project path: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code-worktrees/baseline-phase0-mosaic-baselines`
- Shared cache: `/projects/p32572/Jieke/.cache`
- Latest H100 node used: `qgpu3019`
