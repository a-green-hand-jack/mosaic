# Phase 0 ACT-012 Top-K Sensitivity

## Question

ACT-011 showed a positive M7c top-k handoff result at `topk_samples=4`. ACT-012 tests whether that result survives larger sample budgets and different reranking metrics.

## Run

- Run ID: `phase0_protenix_update_geometry_cc6864e_20260507T051752Z`
- Implementation commit: `cc6864e`
- Server: Quest H100 `qgpu3019`
- Runtime: 10:42 wall-clock
- Setup: IL7RA reduced target length 48, binder length 24, `steps=3`, `num_seeds=2`, ProtenixMini, methods `M3,M4,M7a,M7c`
- Candidate pool: `top_k=4`, `samples_per_method_seed=8`
- Sensitivity: budgets `1,4,8`; rerank metrics `bt_pae`, `bt_iptm`, `contact`
- Artifacts:
  - `docs/reports/phase0_protenix_update_geometry_cc6864e_20260507T051752Z.md`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z.json`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z_candidates.csv`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z_topk_sensitivity.md`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z_topk_sensitivity.csv`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z_topk_sensitivity.json`

## Main Result

The ACT-011 M7c top-k advantage did not replicate in this new candidate pool. M3 naive weighted is best across all tested budgets and rerank metrics.

With BT PAE reranking:

| Budget | Best method | BT PAE | BT ipTM | pLDDT | Contact loss |
|---:|---|---:|---:|---:|---:|
| 1 | M3 | 11.8081 | 0.4425 | 0.6329 | 2.7006 |
| 4 | M3 | 11.8081 | 0.4425 | 0.6329 | 2.7006 |
| 8 | M3 | 11.8081 | 0.4425 | 0.6329 | 2.7006 |

M7c still improves as budget increases, but remains behind M3:

| Budget | M7c BT PAE | M7c BT ipTM | M7c pLDDT | M7c contact loss |
|---:|---:|---:|---:|---:|
| 1 | 17.3080 | 0.1492 | 0.4242 | 3.6130 |
| 4 | 16.1557 | 0.1670 | 0.4666 | 3.2998 |
| 8 | 14.7162 | 0.2427 | 0.4921 | 3.1527 |

The same qualitative pattern holds for BT ipTM and contact reranking: M3 remains strongest; M7c improves with budget but does not overtake M3.

## Interpretation

Top-k handoff is useful, but the M7c advantage is not robust yet. The best current interpretation is:

- ACT-011 established that argmax is not the only viable discrete handoff and that M7c can produce recoverable candidates.
- ACT-012 shows this is not stable enough to claim SCH candidate-level superiority.
- Naive weighted remains a very strong reduced-run candidate baseline, especially after giving every method the same top-k candidate budget.

The next method revision should not simply increase sample budget. It should change the terminal distribution or objective so the contact-preserving method consistently places good residues in high-probability positions. Plausible next checks are entropy annealing, lower sampling temperature, or straight-through hard candidate scoring during optimization.

## Claim Impact

- CLM-002 remains mixed rather than supported at candidate level.
- RSK-010 remains active: the decoding bottleneck is actionable but not solved.
- ACT-012 blocks immediate scale-up; run a smaller design revision before spending on more targets.
