# ACT-017 QP Tuning Sweep

Date: 2026-05-07

## Goal

ACT-015B showed that QP-grid updates are promising but not yet acceptable:

- `M8a` improved soft terminal interface metrics and top-k BT ipTM, but had higher update harm and lost BT PAE/contact against M3.
- `M8b` improved auxiliary harm control, but candidate quality was weaker.

ACT-017 tests whether a smaller sweep over QP tolerance and contact descent ratio can keep the M8a signal while reducing harm.

## Variants

| Method ID | Aux tolerance | Contact descent ratio | Intent |
|---|---:|---:|---|
| `M8a` | 0.08 | 0.60 | ACT-015B contact-aggressive reference. |
| `M8b` | 0.04 | 0.60 | ACT-015B strict auxiliary reference. |
| `M8c` | 0.06 | 0.40 | Middle slack, gentler contact ratio. |
| `M8d` | 0.06 | 0.50 | Middle slack, middle contact ratio. |
| `M8e` | 0.08 | 0.40 | M8a slack, gentler contact ratio. |

## Added Diagnostics

Each `contact_qp_grid` step logs:

- best primary contact derivative available in the candidate set;
- required minimum primary derivative;
- selected primary ratio;
- selected auxiliary violation;
- selected contact violation;
- selected distance to contact-only update;
- whether the selected update is feasible under the local constraints.

These diagnostics distinguish true constrained updates from fallback-selected updates.

## Reduced Gate

Run on reduced IL7RA with:

- target length 48;
- binder length 24;
- steps 3;
- seeds 2;
- top-k candidate handoff with 8 samples;
- sampling temperature 0.25;
- methods `M3,M7c,M8a,M8b,M8c,M8d,M8e`.

## Acceptance Check

A tuned QP variant is positive only if it beats M3 under matched top-k budget on BT PAE or contact while not exceeding M3 update harm. BT ipTM-only wins remain useful but are not enough for scale-up.
