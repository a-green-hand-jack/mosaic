# Phase 0 Baseline Selection Audit

Date: 2026-05-06

Venue target: ICLR 2027

## Claim Surface

We need to show that SCH-style optimization improves cross-oracle robustness for binder hallucination candidates, without the result being explained by weak baselines, different candidate budgets, favorable target choice, or inconsistent filtering.

Phase 0 is narrower: establish the baseline protocol and detect whether Mosaic baselines create the expected train-oracle versus holdout-oracle gap.

## Candidate Baselines

| Candidate | Role | Requirement | Why reviewers expect it | Comparison form | Source/status |
|---|---|---|---|---|---|
| Random or unoptimized sequence scoring | control-baseline | must-have | Shows filters and holdout metrics are not trivially passed. | Same target, length, seeds, and scoring stack. | local implementation planned |
| Mosaic Protenix single-oracle hallucination | nearest-previous-method | must-have | Directly tests the codebase we forked before SCH changes. | Matched target, binder length, seeds, and post-hoc scoring. | repo example available |
| Mosaic weighted composite objective | ablation-baseline | must-have for SCH paper | Distinguishes SCH constraints from ordinary weighted multi-objective optimization. | Same oracle pool and budget, fixed hand-tuned weights. | planned |
| BoltzGen plus Boltz2 ranking | direct-competitor / standard-benchmark-baseline | should-have | Current Mosaic examples include this generation-ranking path; it is close to BoltzDesign-style workflows. | Matched target and candidate budget when cache is available. | repo example available; checkpoint blocked |
| Official BindCraft | domain-required-baseline | must-have for paper-scale claims | The protein binder design community expects BindCraft for de novo binder design comparisons. | Official code, matched target set, candidate budget, and filters where possible. | later external baseline |
| Official BoltzDesign1 | direct-competitor | must-have for paper-scale claims | Closest Boltz-inversion baseline for differentiable binder design. | Official or faithful reproduction, matched target and candidate budget. | later external baseline |
| Post-hoc filter-only reranking | ablation-baseline | should-have | Tests whether SCH gains are just better filtering after generation. | Apply same filters to B1/B2 candidates without changing generation. | planned |

## Fairness Ledger

| Baseline | Data | Capacity | Compute | Tuning | Protocol | Metric | Implementation | Verdict |
|---|---|---|---|---|---|---|---|---|
| Random scoring | same target and length | no trainable model | minimal; report separately | none | same scoring stack | same thresholds and raw metrics | local script | fair |
| Protenix single-oracle | same target and length | Mosaic Protenix model | match steps, seeds, and oracle calls | fixed config first | same post-hoc scoring | same raw metrics and pass rates | Mosaic example adaptation | fair-with-caveat |
| Weighted composite Mosaic | same target and length | same oracles as SCH where possible | match budget to SCH | weight tuning must be capped | same post-hoc scoring | same raw metrics and pass rates | local implementation | needs-matched-run |
| BoltzGen plus Boltz2 | same target where possible | generator plus ranker differs from optimizer | match candidate count and report runtime | checkpoint/posttraining choice must be fixed | same post-hoc scoring | same raw metrics and pass rates | Mosaic example adaptation | blocked |
| BindCraft | target-compatible subset | AF2-based pipeline differs | match candidates or report compute-normalized results | official defaults first | official outputs plus common post-hoc scoring | common holdout metrics | external official code | unverified |
| BoltzDesign1 | target-compatible subset | Boltz-1 inversion differs | match candidates or report compute-normalized results | official defaults first | official outputs plus common post-hoc scoring | common holdout metrics | external official/fidelity TBD | unverified |

## Reviewer Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Reviewer says the baseline is only a weak Mosaic example, not a real binder-design baseline. | major | Include official BindCraft and BoltzDesign1 before paper-scale claims. |
| Weighted composite baseline is under-tuned relative to SCH. | major | Cap and log tuning trials; add sensitivity analysis for weights. |
| BoltzGen posttraining weights silently change the protocol. | medium | Treat posttraining as a config field and record checkpoint provenance. |
| A100 and H100 runs have different performance behavior. | medium | Record GPU class; do not compare raw runtime without hardware labels. |
| Filter thresholds are quoted as official without source verification. | medium | Complete ACT-006 and label thresholds as source-backed or project working values. |

## Phase 0 Decision Value

The minimum useful Phase 0 result is a complete B0/B1 run with provenance and holdout scoring. That is enough to decide whether the metric stack and trajectory capture are worth scaling. External baselines are needed for ICLR 2027 claims, but they should not block the first executable baseline pilot.
