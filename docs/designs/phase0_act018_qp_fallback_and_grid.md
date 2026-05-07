# ACT-018 QP Fallback And Grid Revision

Date: 2026-05-07

## Goal

ACT-017 showed that simple QP threshold tuning is not enough:

- `M8c/M8d/M8e` did not reduce harm and collapsed to the same trajectory.
- `M8b` gave the best hard top-k BT PAE/ipTM in the reduced run, but still exceeded M3 update harm and had one infeasible fallback step with large contact violation.

ACT-018 tests whether the useful `M8b` hard-candidate signal can be kept while making the QP behavior cleaner.

## Variants

| Method ID | Selector | Aux tolerance | Contact descent ratio | Grid denominator | Intent |
|---|---|---:|---:|---:|---|
| `M8b` | `contact_qp_grid` | 0.04 | 0.60 | 10 | ACT-017 strict QP hard-candidate reference. |
| `M8f` | `contact_qp_grid_contact_first` | 0.04 | 0.60 | 10 | Same strict constraints as M8b, but infeasible fallback prioritizes contact violation before auxiliary violation. |
| `M8g` | `contact_qp_grid_contact_first` | 0.06 | 0.60 | 10 | Slightly looser auxiliary tolerance with contact-first fallback. |
| `M8h` | `contact_qp_grid` | 0.04 | 0.60 | 20 | Same strict constraints as M8b, but with a higher-resolution candidate grid. |

## Acceptance Check

A positive ACT-018 result should:

1. beat M3 under matched hard top-k budget on BT PAE/contact or BT PAE/BT ipTM;
2. not exceed M3 update harm;
3. avoid large QP contact-violation fallback events;
4. retain acceptable sequence sanity under trigram loss and pLDDT.

If no variant meets this gate, the next method should move away from QP threshold/fallback tuning and start gradual hardening or a warm-started hard-candidate optimizer.
