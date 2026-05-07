# Phase 0 ACT-011 Top-K Discretization Handoff

## Question

ACT-010 showed that contact-preserving geometry can improve soft terminal interface metrics, but argmax decoding loses most of the gain. ACT-011 tests whether a simple matched-budget top-k sampling and reranking handoff can recover better discrete candidates.

## Run

- Run ID: `phase0_protenix_update_geometry_09ed71d_20260507T045421Z`
- Implementation commit: `09ed71d`
- Server: Quest H100 `qgpu3019`
- Runtime: 10:12 wall-clock
- Setup: IL7RA reduced target length 48, binder length 24, `steps=3`, `num_seeds=2`, ProtenixMini, methods `M3,M4,M7a,M7c`
- Handoff: `top_k=4`, `samples_per_method_seed=4`, rerank by binder-target PAE
- Artifacts:
  - `docs/reports/phase0_protenix_update_geometry_09ed71d_20260507T045421Z.md`
  - `docs/results/phase0_protenix_update_geometry_09ed71d_20260507T045421Z.json`
  - `docs/results/phase0_protenix_update_geometry_09ed71d_20260507T045421Z_steps.csv`
  - `docs/results/phase0_protenix_update_geometry_09ed71d_20260507T045421Z_candidates.csv`

## Update-Level Context

| Method | Harm rate | Worst derivative | Final contact loss |
|---|---:|---:|---:|
| M4 normalized weighted | 0.000 | -0.0086 | 2.7998 |
| M7a strict contact-preserving | 0.000 | -0.0046 | 2.8597 |
| M7c aggressive contact-preserving | 0.167 | 0.0088 | 2.8918 |
| M3 naive weighted | 0.208 | 0.0148 | 2.6145 |

M7c remains safer than naive weighted by update harm, but in this repeat its soft terminal metrics are not stronger than naive. This means the run should be interpreted as a handoff diagnostic, not as a direct replication of the ACT-010 soft-ranking result.

## Discrete Candidate Result

Best-per-seed discrete candidates selected by binder-target PAE:

| Method / mode | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss |
|---|---:|---:|---:|---:|---:|
| M7c top-k sample | 0.5953 | 11.9610 | 0.4313 | 0.1931 | 2.8101 |
| M7a top-k sample | 0.5417 | 14.0909 | 0.3045 | 0.2241 | 2.7869 |
| M3 top-k sample | 0.5475 | 14.8959 | 0.3663 | 0.2048 | 2.9420 |
| M3 argmax | 0.5142 | 15.1363 | 0.2715 | 0.1927 | 3.1093 |
| M7a argmax | 0.4660 | 15.5488 | 0.2049 | 0.1662 | 3.1162 |
| M4 top-k sample | 0.4981 | 15.9520 | 0.2228 | 0.0991 | 3.2629 |
| M7c argmax | 0.4710 | 16.5451 | 0.2141 | 0.1045 | 3.3326 |
| M4 argmax | 0.4279 | 17.6580 | 0.1483 | 0.0813 | 3.5159 |

M7c top-k sampling improves strongly over M7c argmax:

- BT PAE: `16.5451 -> 11.9610`
- BT ipTM: `0.2141 -> 0.4313`
- pLDDT: `0.4710 -> 0.5953`
- contact loss: `3.3326 -> 2.8101`

It also beats naive weighted under the same top-k sample budget and beats naive argmax on BT PAE, BT ipTM, pLDDT, and contact loss.

## Interpretation

This is the first evidence that the argmax bottleneck is actionable. A small top-k sampling and reranking handoff recovers high-quality discrete candidates from M7c that were not visible under single argmax decoding.

The result does not yet prove SCH candidate-level superiority, because the setting is still reduced and the best-of-k reranking adds oracle calls. It does support the next design direction: treat decoding as part of the method and report candidate quality under a matched discrete-candidate budget.

## Next Step

Run a controlled ACT-011 follow-up that compares:

- `topk_samples=1,4,8`
- rerank metrics `bt_pae` and `bt_iptm`
- methods `M3,M4,M7a,M7c`

If M7c remains best under matched sample budgets, move to more seeds/targets. If the gain disappears, test entropy annealing or straight-through hard scoring during optimization.
