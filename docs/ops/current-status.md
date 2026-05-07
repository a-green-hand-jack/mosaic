# Code Ops Status

Last updated: 2026-05-07

## State

Official Mosaic cloned locally. Upstream remote is `upstream`. Writable fork remote `origin` is `git@github.com:a-green-hand-jack/mosaic.git`.

Quest clone exists at `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code`. Verify the exact synchronized commit with `git rev-parse --short HEAD` before running jobs.

Quest environment uses uv:

- uv: `/home/zpt6685/.local/bin/uv`
- cache: `/projects/p32572/Jieke/.cache/uv`
- uv-managed Python root: `/projects/p32572/Jieke/Envs/uv-python`
- venv: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code/.venv`
- Python: 3.12.12
- sync command: `uv sync --group jax-cuda`
- GPU policy: use one uv environment across A100 and H100, then smoke-test each GPU family.
- smoke script: `jobs/quest_jax_cuda_smoke.slurm`

## Current Focus

Phase 0 method revision after ACT-010 through ACT-017. Current evidence supports gradient-guided relaxed optimization plus cold top-k handoff as useful, but rules out naive hard-sample CEM from a random relaxed distribution and simple QP threshold tuning as scalable paths. ACT-017 found that M8c/M8d/M8e do not reduce harm, while M8b gives the best hard top-k BT PAE/ipTM in the reduced run but still exceeds M3 update harm. The next priority is ACT-018: revise QP fallback/candidate search around M8b or implement a hard-candidate optimizer slice.

## Latest Run

- Commit: `f092264`
- Run ID: `phase0_protenix_update_geometry_f092264_20260507T042412Z`
- Server/node: Quest H100 `qgpu3019`
- Runtime: 16:24 wall-clock, exit status 0
- Report: `docs/reports/phase0_act010_contact_sweep_2026-05-07.md`
- Raw artifacts:
  - `docs/reports/phase0_protenix_update_geometry_f092264_20260507T042412Z.md`
  - `docs/results/phase0_protenix_update_geometry_f092264_20260507T042412Z.json`
  - `docs/results/phase0_protenix_update_geometry_f092264_20260507T042412Z_steps.csv`
  - `docs/results/phase0_protenix_update_geometry_f092264_20260507T042412Z_candidates.csv`

Result summary: aggressive contact-preserving update `M7c` beats naive/normalized baselines under soft terminal scoring, but the advantage is mostly lost after argmax discretization. The next method step should target discretization-aware candidate handoff rather than only tuning cone slack.

## ACT-011 Result

- Commit: `09ed71d`
- Run ID: `phase0_protenix_update_geometry_09ed71d_20260507T045421Z`
- Server/node: Quest H100 `qgpu3019`
- Runtime: 10:12 wall-clock, exit status 0
- Report: `docs/reports/phase0_act011_topk_handoff_2026-05-07.md`
- Raw artifacts:
  - `docs/reports/phase0_protenix_update_geometry_09ed71d_20260507T045421Z.md`
  - `docs/results/phase0_protenix_update_geometry_09ed71d_20260507T045421Z.json`
  - `docs/results/phase0_protenix_update_geometry_09ed71d_20260507T045421Z_steps.csv`
  - `docs/results/phase0_protenix_update_geometry_09ed71d_20260507T045421Z_candidates.csv`

Result summary: with matched `top_k=4`, `samples_per_method_seed=4`, and BT PAE reranking, M7c top-k samples beat M7c argmax and naive weighted top-k/argmax on discrete candidate interface metrics. This supports discretization-aware handoff as a viable next method component.

## ACT-012 Result

- Commit: `cc6864e`
- Run ID: `phase0_protenix_update_geometry_cc6864e_20260507T051752Z`
- Server/node: Quest H100 `qgpu3019`
- Runtime: 10:42 wall-clock, exit status 0
- Report: `docs/reports/phase0_act012_topk_sensitivity_2026-05-07.md`
- Raw artifacts:
  - `docs/reports/phase0_protenix_update_geometry_cc6864e_20260507T051752Z.md`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z.json`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z_candidates.csv`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z_topk_sensitivity.md`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z_topk_sensitivity.csv`
  - `docs/results/phase0_protenix_update_geometry_cc6864e_20260507T051752Z_topk_sensitivity.json`

Result summary: top-k handoff improves M7c as sample budget increases, but the ACT-011 M7c advantage did not replicate. M3 naive weighted remains best across budgets `1,4,8` and rerank metrics `bt_pae`, `bt_iptm`, and `contact`. The next step should revise the terminal distribution/objective rather than simply increase top-k budget.

## ACT-015A Result

- Commit: `1ea2072`
- Run ID: `phase0_protenix_cem_1ea2072_20260507T080401Z`
- Server/node: Quest H100 `qgpu3019`
- Runtime: 6:29 wall-clock, exit status 0
- Report: `docs/reports/phase0_act015a_cem_2026-05-07.md`
- Raw artifacts:
  - `docs/reports/phase0_protenix_cem_1ea2072_20260507T080401Z.md`
  - `docs/results/phase0_protenix_cem_1ea2072_20260507T080401Z.json`
  - `docs/results/phase0_protenix_cem_1ea2072_20260507T080401Z_candidates.csv`
  - `docs/results/phase0_protenix_cem_1ea2072_20260507T080401Z_cem_rounds.csv`
  - `docs/results/phase0_protenix_cem_1ea2072_20260507T080401Z_topk_sensitivity.md`

Result summary: naive CEM/elite-sampling successfully lowers entropy but does not improve hard candidate quality. Under matched 24-candidate BT PAE reranking, M3 reaches 10.7001, M7c reaches 10.8029, CEMp reaches 14.9997, and CEMc reaches 15.8457. Do not scale this CEM variant; prioritize QP-style constrained gradient updates or only test warm-start CEM as a diagnostic.

## ACT-015B Result

- Commit: `ba8c855`
- Run ID: `phase0_protenix_update_geometry_ba8c855_20260507T082915Z`
- Server/node: Quest H100 `qgpu3019`
- Runtime: 10:37 wall-clock, exit status 0
- Report: `docs/reports/phase0_act015b_qp_2026-05-07.md`
- Raw artifacts:
  - `docs/reports/phase0_protenix_update_geometry_ba8c855_20260507T082915Z.md`
  - `docs/results/phase0_protenix_update_geometry_ba8c855_20260507T082915Z.json`
  - `docs/results/phase0_protenix_update_geometry_ba8c855_20260507T082915Z_candidates.csv`
  - `docs/results/phase0_protenix_update_geometry_ba8c855_20260507T082915Z_steps.csv`
  - `docs/results/phase0_protenix_update_geometry_ba8c855_20260507T082915Z_topk_sensitivity.md`

Result summary: QP-grid M8a is promising but not accepted. M8a soft terminal reaches BT PAE 7.0264 and BT ipTM 0.5880, better than M3 soft 8.7660/0.5183. Under budget-8 BT ipTM reranking, M8a top-k reaches 0.4800 versus M3 0.4576. However, M3 still wins budget-8 BT PAE and contact reranking, and M8a has higher update harm than M3. M8b has better harm control but weaker candidates.

## ACT-017 Result

- Commit: `65947b0` for implementation/run; `eb0ca34` for committed results
- Run ID: `phase0_protenix_update_geometry_65947b0_20260507T091217Z`
- Server/node: Quest H100 `qgpu3019`
- Runtime: 16:50 wall-clock, exit status 0
- Report: `docs/reports/phase0_act017_qp_tuning_2026-05-07.md`
- Raw artifacts:
  - `docs/reports/phase0_protenix_update_geometry_65947b0_20260507T091217Z.md`
  - `docs/results/phase0_protenix_update_geometry_65947b0_20260507T091217Z.json`
  - `docs/results/phase0_protenix_update_geometry_65947b0_20260507T091217Z_candidates.csv`
  - `docs/results/phase0_protenix_update_geometry_65947b0_20260507T091217Z_steps.csv`
  - `docs/results/phase0_protenix_update_geometry_65947b0_20260507T091217Z_topk_sensitivity.md`

Result summary: simple QP threshold tuning is negative. M8c/M8d/M8e all have harm rate 0.3333 and collapse to the same trajectory. M8b is the best hard top-k method in this run: at budget 8, BT PAE reranking gives M8b 9.5652 versus M3 12.8342, and BT ipTM reranking gives M8b 0.4657 versus M3 0.4533. However, M8b update harm is 0.2917 versus M3 0.2083 and includes an infeasible fallback step, so it is not ready for scale-up.

## Blockers

- Torch is currently CPU-only because Mosaic's `pyproject.toml` routes torch through the PyTorch CPU wheel index.
- ACT-018 needs a method choice before the next run: revised QP fallback/candidate search versus hard-candidate optimizer such as CEM-lite or gradual position-wise hardening.
