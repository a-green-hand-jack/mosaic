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

Phase 0 method revision after ACT-010 through ACT-015A. Current evidence supports gradient-guided relaxed optimization plus cold top-k handoff as useful, but rules out naive hard-sample CEM from a random relaxed distribution as the next scalable path. The next priority is ACT-015B QP-style constrained update, with optional warm-start CEM only as a diagnostic.

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

## Blockers

- Torch is currently CPU-only because Mosaic's `pyproject.toml` routes torch through the PyTorch CPU wheel index.
