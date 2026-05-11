# Phase 0 Wave 1 Cross-Target Conflict Map

## Summary

- Question: whether Protenix, Boltz2, and sequence-oracle gradients show target-specific conflicts, and which update rule is safe enough to gate later candidate-level tests.
- Setup: 5 targets (`il7ra`, `pdl1`, `pd1`, `cd47`, `il6`), 4 seeds, 1 update step, binder length 24, target length 48, methods `M3,M4,M7c,M8a`.
- Headline result: Protenix-Boltz2 gradient cosine is weakly positive, not strongly opposed, across the target panel (`0.0005` to `0.2113`).
- Interpretation: the conflict is not a simple "two structure oracles point in opposite directions" story; it is a multi-oracle update-harm story where aggressive updates harm auxiliary objectives.
- Decision: keep `M4` normalized weighting as the current update-level baseline. Do not scale `M8a`; test cleaner conflict-balancing rules against `M4` before candidate-level promotion.

## 1. Experiment Motivation

Earlier Phase 0 runs showed real oracle harm in reduced Protenix-only and Protenix/Boltz2 settings, but the initial Boltz2 gradient-conflict run was IL7RA-only. This run tests whether the same mechanism appears across a broader Wave 1 target slice before spending compute on candidate-level or 24-target benchmarks.

The specific question was not whether a current SCH variant already wins final binder quality. The experiment was designed as a method gate: first confirm cross-target oracle misalignment and identify a safe update-level baseline.

## 2. Experiment Setup

| Item | Value |
|---|---|
| Run mode | Detached manual run on Quest `qgpu2013`, GPU0 |
| Code version | `23bc3a1` for launcher/driver; results committed at `9981422` |
| Launcher | `jobs/launch_boltz2_oracle_conflict_map_wave1.sh` |
| Driver | `scripts/run_boltz2_oracle_gradient_conflict.py` |
| Targets | `il7ra`, `pdl1`, `pd1`, `cd47`, `il6` |
| Methods | `M3` naive weighted, `M4` normalized weighted, `M7c` contact-preserving soft cone, `M8a` QP grid |
| Oracles | Protenix contact, Boltz2 distogram contact, solubility, charge, trigram naturalness |
| Seeds / steps | 4 seeds, 1 step |
| Binder / target length | 24 / 48 |
| Key controls | Protenix recycling 1, Boltz2 sampling 1, Boltz2 recycling 1, step size 0.25 |
| Main log | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/manual-boltz2-oracle-conflict-map-wave1-qgpu2013-20260510T213909Z.out` |
| Resource log | `/projects/p32572/Jieke/Projects/SCH-BinderDesign/logs/resource-boltz2-oracle-conflict-map-wave1-qgpu2013-20260510T213909Z.out` |

Target mapping:

| Target | Structure | Chain | Target length |
|---|---|---:|---:|
| `il7ra` | `IL7RA.cif` | `A` | 48 |
| `pdl1` | `4ZQK.cif` | `A` | 48 |
| `pd1` | `4ZQK.cif` | `B` | 48 |
| `cd47` | `2JJS.cif` | `C` | 48 |
| `il6` | `1ALU.cif` | `A` | 48 |

## 3. Core Algorithm or Method

The driver initializes a relaxed binder sequence distribution and measures oracle gradients for Protenix contact, Boltz2 distogram contact, and sequence-level sanity oracles. For each method and seed it applies one proposed update and records:

- the Protenix-Boltz2 gradient cosine before the update;
- each oracle's directional derivative under the chosen update;
- oracle harm rate, defined by how often auxiliary directional derivatives become positive;
- worst oracle directional derivative and step norm.

The compared update rules differ only in how they combine or constrain oracle gradients. This keeps the experiment focused on update geometry rather than candidate generation or decoding.

## 4. Metrics

| Metric | Definition | Direction | Why it matters |
|---|---|---|---|
| Protenix-Boltz2 cosine | Cosine similarity between the two structure-oracle gradients | More negative means stronger direct opposition | Tests the simple "opposed oracle" hypothesis |
| Oracle harm rate | Fraction of auxiliary oracle derivatives that are positive after the update | Lower is better | Measures whether improving the primary objective harms other oracle terms |
| Worst directional derivative | Maximum directional derivative across non-primary oracles | More negative is safer | Captures the most harmed oracle in each update |
| Protenix directional derivative | Directional derivative for Protenix contact | More negative is stronger descent | Measures primary structure-oracle descent |
| Boltz2 directional derivative | Directional derivative for Boltz2 distogram contact | More negative is stronger descent | Measures holdout structure-oracle agreement |
| Step norm | Norm of the update step | Should remain nontrivial | Guards against "safe because no movement" |

## 5. Results

Aggregate over 5 targets and 4 seeds per target:

| Method | Avg harm | Avg worst dir. | Avg Protenix dir. | Avg Boltz2 dir. | Avg step norm | Zero-harm targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M4` | 0.000 | -0.0135 | -1.8417 | -0.0719 | 0.5023 | 5/5 |
| `M7c` | 0.080 | 0.0047 | -5.3903 | -0.0669 | 0.8103 | 1/5 |
| `M3` | 0.250 | 0.0353 | -7.7743 | -0.0439 | 1.0425 | 0/5 |
| `M8a` | 0.390 | 0.0354 | -7.2694 | -0.0301 | 0.9464 | 0/5 |

Per-target harm summary:

| Target | Protenix-Boltz2 cosine | M4 harm | M7c harm | M3 harm | M8a harm | Best safe method |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `il7ra` | 0.0005 | 0.00 | 0.05 | 0.20 | 0.40 | `M4` |
| `pdl1` | 0.0337 | 0.00 | 0.10 | 0.40 | 0.45 | `M4` |
| `pd1` | 0.0469 | 0.00 | 0.20 | 0.20 | 0.45 | `M4` |
| `cd47` | 0.1024 | 0.00 | 0.05 | 0.15 | 0.30 | `M4` |
| `il6` | 0.2113 | 0.00 | 0.00 | 0.30 | 0.35 | `M4` / `M7c` tied on harm |

Runtime and resource observations:

| Item | Observed value |
|---|---:|
| Total wall time | 2h 32m 50s |
| Per-target wall time | about 30-31 min |
| GPU memory, GPU0 | peak `8715 MiB`, average `8205 MiB` across 60s samples |
| GPU utilization snapshots, GPU0 | average `0.32%`, peak `14%` |
| Power snapshots, GPU0 | average `74.65 W`, peak `86.72 W` |

The low sampled utilization does not mean the job was idle. The process held JAX/Boltz2 memory throughout the run and likely had bursty GPU kernels plus CPU/JAX/Python overhead that 60s `nvidia-smi` snapshots under-sampled.

## 6. How to Read the Tables

The aggregate table should be read as an update-geometry gate, not as candidate-quality evidence. Negative directional derivatives are good because they indicate local descent for the corresponding oracle. A method can look strong on Protenix descent but still fail if it produces positive worst-oracle derivatives or high harm rates.

The per-target table shows that `M4` was the only method with zero harm on all five targets. `M7c` was close on several targets and tied `M4` on IL6, but it did not pass the across-target safety gate. `M3` and `M8a` both moved more aggressively but produced consistent auxiliary harm.

## 7. Interpretation

The Wave 1 conflict map supports the Phase 0 problem statement but narrows its mechanism:

- The Protenix and Boltz2 structure gradients are not strongly opposed in this reduced setup. They are mostly near-orthogonal or weakly aligned.
- Naive scalarization (`M3`) still causes real auxiliary harm, so the lack of negative Protenix-Boltz2 cosine does not falsify the multi-oracle conflict claim.
- `M4` is not merely a no-op: it keeps a nontrivial step norm (`0.5023`) and maintains negative Protenix and Boltz2 directional derivatives while achieving zero measured harm.
- `M8a` should be treated as a negative reference, not a scale-up candidate. It is more aggressive than `M4`, but its average harm rate is the worst of the tested methods.

## 8. Conclusion and Discussion

EXP-A is complete. It shows cross-target update harm and identifies `M4` normalized weighting as the update-level baseline to beat. It does not show the originally tempting story that Protenix and Boltz2 are directly adversarial oracles.

The correct next step is therefore EXP-B: implement or test a cleaner conflict-balancing rule against `M4` on the same update-level metrics. Candidate-level Boltz2 holdout should remain gated, because none of the current SCH-style variants beats `M4` on safety.

## 9. Limitations and Caveats

- This is a reduced diagnostic: 5 targets, 4 seeds, 1 step, binder length 24, target length 48.
- Several targets use raw multi-chain RCSB structures with explicit chain selection rather than canonical target-only assets.
- Metrics are local update-geometry measurements, not final binder pass rates.
- `nvidia-smi` utilization was sampled every 60 seconds and can miss short GPU kernels.
- Boltz2 coordinate-level losses remain excluded; this run uses finite distogram/contact-style diagnostics.

## 10. Next Steps

1. Add a cleaner update-rule candidate for EXP-B, such as PCGrad-style projection, min-norm gradient combination, or a CAGrad-style compromise.
2. Run EXP-B on the same 5-target panel with `M3`, `M4`, `M7c`, `M8a`, and the new candidate rule.
3. Promote a method to candidate-level Boltz2 holdout only if it matches `M4` on harm and worst derivative while improving primary or Boltz2 descent without shrinking the step into irrelevance.
4. Keep `M4` as the required baseline in all near-term Phase 0 comparisons.

## Reproducibility Notes

Run record:

- `docs/runs/2026-05-10-phase0-conflict-map-wave1.md`

Per-target reports:

- `docs/reports/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T220950Z.md`
- `docs/reports/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T224007Z.md`
- `docs/reports/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T231100Z.md`
- `docs/reports/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T234048Z.md`
- `docs/reports/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260511T001153Z.md`

Raw results:

- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T220950Z.json`
- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T220950Z_steps.csv`
- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T224007Z.json`
- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T224007Z_steps.csv`
- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T231100Z.json`
- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T231100Z_steps.csv`
- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T234048Z.json`
- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260510T234048Z_steps.csv`
- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260511T001153Z.json`
- `docs/results/phase0_boltz2_oracle_gradient_conflict_23bc3a1_20260511T001153Z_steps.csv`
