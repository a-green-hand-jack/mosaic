# Phase 0 ACT-010 Contact Sweep

## Question

This diagnostic tests whether the candidate-level gap after contact-preserving safe updates is caused by:

- contact-preserving updates being too conservative,
- argmax discretization destroying a good relaxed sequence,
- or both.

## Run

- Run ID: `phase0_protenix_update_geometry_f092264_20260507T042412Z`
- Commit: `f092264`
- Server: Quest H100 `qgpu3019`
- Runtime: 16:24 wall-clock
- Setup: IL7RA reduced target length 48, binder length 24, `steps=3`, `num_seeds=2`, ProtenixMini, `candidate_score_mode=both`
- Artifacts:
  - `docs/reports/phase0_protenix_update_geometry_f092264_20260507T042412Z.md`
  - `docs/results/phase0_protenix_update_geometry_f092264_20260507T042412Z.json`
  - `docs/results/phase0_protenix_update_geometry_f092264_20260507T042412Z_steps.csv`
  - `docs/results/phase0_protenix_update_geometry_f092264_20260507T042412Z_candidates.csv`

## Variants

- `M7a`: strict contact-preserving cone, no auxiliary harms, `aux_slack=0.0`, minimum primary descent ratio `0.25`.
- `M7b`: previous contact-preserving setting, `aux_slack=0.02`, at most one auxiliary harm.
- `M7c`: more aggressive contact descent, `aux_slack=0.08`, at most one auxiliary harm, minimum primary descent ratio `0.60`.

## Key Results

### Update-Level Safety

| Method | Harm rate | Worst derivative | Final contact loss |
|---|---:|---:|---:|
| normalized weighted | 0.000 | -0.0087 | 2.7916 |
| soft-cone correction | 0.000 | -0.0087 | 2.7916 |
| M7a strict contact-preserving | 0.000 | -0.0046 | 2.7701 |
| M7c aggressive contact-preserving | 0.125 | 0.0099 | 2.7187 |
| M7b previous contact-preserving | 0.167 | 0.0009 | 2.7165 |
| naive weighted | 0.208 | 0.0111 | 2.5494 |
| single Protenix contact | 0.333 | 0.0295 | 2.8228 |

M7a is the cleanest safe contact-preserving variant: it keeps zero measured update harm while slightly improving final contact loss over normalized weighted and soft-cone. M7c and M7b buy more contact movement at the cost of nonzero harm, but still remain safer than naive weighted.

### Candidate-Level Soft Scoring

| Method | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss |
|---|---:|---:|---:|---:|---:|
| M7c aggressive contact-preserving | 0.7266 | 8.7199 | 0.5104 | 0.2393 | 2.4296 |
| naive weighted | 0.6989 | 9.5244 | 0.4834 | 0.2515 | 2.3242 |
| single Protenix contact | 0.7054 | 10.1270 | 0.4274 | 0.2140 | 2.4100 |
| M7a strict contact-preserving | 0.6435 | 10.4727 | 0.4373 | 0.2069 | 2.5198 |
| normalized weighted | 0.6342 | 11.0390 | 0.3745 | 0.2294 | 2.5172 |
| soft-cone correction | 0.6342 | 11.0390 | 0.3745 | 0.2294 | 2.5172 |

Under soft terminal scoring, M7c is the strongest interface-quality variant by BT PAE, BT ipTM, and pLDDT. This supports the mechanism claim that contact-aware geometry can improve the relaxed optimized sequence beyond normalized weighted and naive scalarization.

### Candidate-Level Argmax Scoring

| Method | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss |
|---|---:|---:|---:|---:|---:|
| single Protenix contact | 0.5186 | 14.8712 | 0.2268 | 0.1858 | 2.9805 |
| naive weighted | 0.5091 | 15.4820 | 0.2665 | 0.1775 | 3.1018 |
| M7a strict contact-preserving | 0.4437 | 16.4103 | 0.1959 | 0.1881 | 3.1935 |
| M7c aggressive contact-preserving | 0.4483 | 16.4140 | 0.1350 | 0.0679 | 3.3825 |
| normalized weighted | 0.4256 | 17.5593 | 0.1499 | 0.0816 | 3.5190 |
| soft-cone correction | 0.4256 | 17.5593 | 0.1499 | 0.0816 | 3.5190 |

After argmax, M7a and M7c still improve over normalized weighted and soft-cone, but they do not beat naive weighted or single-Protenix on interface metrics. The relaxed-sequence advantage is therefore not surviving discretization.

## Interpretation

This run answers the immediate ACT-010 question. Contact-aware geometry is not merely over-regularizing: when evaluated as a soft terminal sequence, the aggressive contact-preserving variant is clearly stronger than normalized weighted and naive weighted on key interface metrics. The candidate-level failure is mainly introduced by argmax discretization.

The next method change should not only tune cone constraints. It should add a discretization-aware handoff from relaxed sequence to candidate sequence, such as entropy annealing, straight-through argmax scoring, top-k sampling/reranking, or terminal ProteinMPNN/inverse-folding refinement under the same oracle budget.

## Claim Impact

- Strengthens CLM-001: geometric update conflicts and safer update rules remain real under structure-oracle diagnostics.
- Strengthens the mechanism part of CLM-002 only for soft terminal candidates.
- Keeps CLM-002 unproven for final discrete binders because argmax candidates still trail naive/single baselines.
