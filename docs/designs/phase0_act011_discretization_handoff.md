# Phase 0 ACT-011 Discretization-Aware Handoff

## Design Context

ACT-010 showed that contact-preserving geometry can improve the relaxed terminal sequence, but naive argmax decoding loses most of that gain. The next question is whether a simple matched-budget handoff can recover discrete candidates without changing the update rule.

## Target Claim

CLM-002 remains unproven for final discrete binders. ACT-011 tests the narrower mechanism claim: if the relaxed terminal sequence contains useful interface information, then sampling and reranking discrete candidates from that distribution should recover better argmax-level candidates than single argmax decoding.

## Method Specification

Keep optimization unchanged. After each method produces a terminal relaxed sequence `P`, score:

- `soft`: the relaxed terminal sequence itself.
- `argmax`: one-hot argmax at each residue.
- `topk_sample`: sample one-hot residues from the top-k amino acids under `P`, then score each sampled candidate with the same ProtenixMini candidate scorer.

For the first implementation:

- `top_k = 4`
- `samples_per_method_seed = 4`
- rerank metric: binder-target PAE, lower is better
- methods: `M3`, `M4`, `M7a`, `M7c`
- controls: same target, seed list, update steps, ProtenixMini settings, and candidate scoring code path

## Assumptions and Invariants

- A1: The relaxed sequence probabilities contain useful alternatives that argmax discards.
- A2: A small number of samples is enough to detect whether the handoff direction is promising.
- I1: The update trajectory and losses are unchanged from ACT-010.
- I2: Every method gets the same number of sampled candidates and the same reranking metric.
- I3: Report both all-candidate averages and best-per-seed selected candidates.

## Failure Modes

- If top-k samples are worse than argmax for all methods, the relaxed distribution may be too diffuse or poorly calibrated for naive sampling.
- If all methods improve equally, the gain is from extra candidate budget, not SCH-specific geometry.
- If M7c improves under best-of-k but naive still wins, the method needs a stronger discretization-aware objective or terminal refinement.
- If M7c wins only by using more samples, paper claims must be cost-normalized.

## Implementation Handoff

- File: `scripts/run_protenix_update_geometry_smoke.py`
- New flags:
  - `--candidate-topk-samples`
  - `--candidate-sample-top-k`
  - `--candidate-sample-temperature`
  - `--candidate-best-metric`
  - `--method-ids`
- Job script: `jobs/quest_protenix_act011_topk_handoff_h100.slurm`
- First run: H100, reduced IL7RA, `steps=3`, `num_seeds=2`, `topk_samples=4`, methods `M3,M4,M7a,M7c`
- Acceptance check: decide whether best-of-top-k discrete candidates let M7c close or reverse the argmax gap to naive weighted under matched sample budget.
