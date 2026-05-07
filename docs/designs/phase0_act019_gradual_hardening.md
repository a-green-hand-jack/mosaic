# ACT-019 Gradual Position-Wise Hardening

Date: 2026-05-07

## Goal

ACT-014 showed that global entropy sharpening is too disruptive. ACT-017 and ACT-018 showed that QP-grid tuning can improve hard candidates only by accepting higher update harm. ACT-019 tests a narrower repair:

> convert relaxed sequences to hard sequences gradually by freezing high-confidence positions during optimization.

The first slice isolates the hardening mechanism by using the same weighted update direction as M3, then adding different position-freeze schedules.

## Variants

| Method ID | Method | Freeze rule | Intent |
|---|---|---|---|
| `M9a` | `position_hardening_probability` | freeze positions with max probability >= 0.42 | Simple confidence threshold. |
| `M9b` | `position_hardening_consensus` | freeze positions whose top amino acid appears in >= 75% of 8 samples | Cold-sample stability. |
| `M9c` | `position_hardening_margin_lowsensitivity` | freeze positions with top1-top2 margin >= 0.12 and below-average gradient sensitivity | Avoid hardening locally sensitive positions. |

Each method freezes at most 20% of positions per step.

## Controls

- `M3`: primary naive weighted baseline.
- `M7c`: contact-preserving cold-handoff baseline.
- `M8a/M8g`: contact-aggressive QP controls that win some hard top-k metrics but have high harm.

## Acceptance Check

A positive ACT-019 result should:

1. beat M3 under matched hard top-k budget on BT PAE/contact or BT PAE/BT ipTM;
2. not exceed M3 update harm;
3. reduce the soft-to-hard gap relative to M3;
4. freeze a nontrivial but not complete fraction of positions.

If M9 improves hard candidates but raises harm, the next slice should make hardening conditional on update harm. If M9 freezes too few positions, lower thresholds. If it freezes too many positions and collapses quality, add a trust region or lower the per-step freeze fraction.
