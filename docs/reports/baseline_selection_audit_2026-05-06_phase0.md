# Phase 0 Baseline Selection Audit

Date: 2026-05-06

Venue target: ICLR 2027

## Claim Surface

We need to show that SCH-style optimization improves multi-oracle update quality inside Mosaic, without the result being explained by weak scalarization baselines, different oracle budgets, favorable target choice, or inconsistent filtering.

Phase 0 is narrower: establish the baseline protocol and detect whether Mosaic oracles create gradient conflicts that naive weighted scalarization handles poorly.

## Candidate Baselines

| Candidate | Role | Requirement | Why reviewers expect it | Comparison form | Source/status |
|---|---|---|---|---|---|
| Random or unoptimized sequence scoring | control-baseline | must-have | Shows filters and holdout metrics are not trivially passed. | Same target, length, seeds, and scoring stack. | local implementation planned |
| Mosaic Protenix single-oracle hallucination | nearest-previous-method | must-have | Directly tests the codebase we forked before SCH changes. | Matched target, binder length, seeds, and post-hoc scoring. | repo example available |
| Mosaic Boltz or AF2 single-oracle hallucination | symmetric single-oracle baseline | should-have | Checks that observations are not Protenix-specific. | Same target, binder length, seeds, and post-hoc scoring. | blocked on oracle setup if Boltz is used |
| Mosaic weighted composite objective | ablation-baseline | must-have | Distinguishes SCH constraints from ordinary weighted multi-objective optimization. | Same oracle pool and budget, fixed weights. | planned |
| Normalized or clipped weighted objective | ablation-baseline | must-have | Tests whether simple gradient-scale fixes are sufficient. | Same oracle pool and budget, normalized gradients or clipped terms. | planned |
| Post-hoc filter-only reranking | ablation-baseline | should-have | Tests whether SCH gains are just better filtering after optimization. | Apply same filters to M1/M3 candidates without changing updates. | planned |
| BoltzGen plus Boltz2 ranking | not-comparable | excluded | It is a generation/ranking workflow rather than a matched update-rule baseline. | Do not run in current stage. | excluded |
| Official BindCraft | citation-only for current stage | deferred | Relevant to broader binder design, but not needed for Mosaic-internal update geometry. | Revisit for paper-scale external comparison. | deferred |
| Official BoltzDesign1 | citation-only for current stage | deferred | Relevant to broader binder design, but not needed for Mosaic-internal update geometry. | Revisit for paper-scale external comparison. | deferred |

## Fairness Ledger

| Baseline | Data | Capacity | Compute | Tuning | Protocol | Metric | Implementation | Verdict |
|---|---|---|---|---|---|---|---|---|
| Random scoring | same target and length | no trainable model | minimal; report separately | none | same scoring stack | same thresholds and raw metrics | local script | fair |
| Protenix single-oracle | same target and length | Mosaic Protenix model | match steps, seeds, and oracle calls | fixed config first | same post-hoc scoring | same raw metrics and pass rates | Mosaic example adaptation | fair-with-caveat |
| Boltz or AF2 single-oracle | same target and length | second Mosaic structure oracle | match steps, seeds, and oracle calls | fixed config first | same post-hoc scoring | same raw metrics and pass rates | Mosaic implementation | blocked/unverified |
| Weighted composite Mosaic | same target and length | same oracles as SCH where possible | match oracle calls and steps | weight tuning must be capped | same update logging and post-hoc scoring | update metrics plus final metrics | local implementation | needs-matched-run |
| Normalized/clipped weighted Mosaic | same target and length | same oracles as SCH | match oracle calls and steps | normalization/clipping rules fixed before run | same update logging and post-hoc scoring | update metrics plus final metrics | local implementation | planned |
| Post-hoc reranking | same candidate pool | no new optimizer | no extra generation budget | fixed scoring rule | same post-hoc scoring | final metrics only | local implementation | planned |
| BoltzGen plus Boltz2 | not matched | generator/ranker differs | not comparable to update-rule budget | not applicable | generation/ranking workflow | not an update-rule metric | Mosaic example | excluded |

## Reviewer Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Reviewer says SCH only beats a weak weighted baseline. | major | Include normalized/clipped weighted losses and cap tuning fairly. |
| Weighted composite baseline is under-tuned relative to SCH. | major | Cap and log tuning trials; add sensitivity analysis for weights. |
| Second structure oracle setup silently changes the protocol. | medium | Treat oracle model, checkpoint, recycling, and cache provenance as config fields. |
| A100 and H100 runs have different performance behavior. | medium | Record GPU class; do not compare raw runtime without hardware labels. |
| Filter thresholds are quoted as official without source verification. | medium | Complete ACT-006 and label thresholds as source-backed or project working values. |

## Phase 0 Decision Value

The minimum useful Phase 0 result is a complete M0/M1/M3 run with provenance, update-level geometry metrics, and final scoring. That is enough to decide whether the metric stack and trajectory capture are worth scaling. External baselines are not needed for the first Mosaic-internal update-geometry claim.
