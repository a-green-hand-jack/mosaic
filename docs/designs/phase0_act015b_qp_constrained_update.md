# ACT-015B QP-Style Constrained Update

Date: 2026-05-07

## Goal

ACT-015B tests whether the M7c contact-preserving update can be made more systematic. Instead of choosing a cone direction by heuristic sorting, the update approximates a small constrained optimization problem:

```text
minimize_d ||d - d_contact||^2
subject to g_contact^T d <= -epsilon
           g_i^T d <= tau_i for each auxiliary oracle
```

The intent is to stay close to the contact oracle's desired update while explicitly bounding auxiliary harm.

## Implementation

The first prototype avoids a solver dependency. It reuses the existing cone candidate set:

- normalized weighted direction;
- raw weighted direction;
- single-oracle normalized directions;
- single-oracle raw directions;
- simplex-grid mixtures of negative normalized gradients.

For every candidate, the diagnostic computes the actual projected simplex update. The QP-grid selector then:

1. computes the projected contact-only update;
2. filters candidates that keep contact descent and keep each auxiliary directional derivative below `aux_slack`;
3. chooses the feasible candidate whose actual projected update is closest to the contact-only update;
4. if no candidate is feasible, falls back to the candidate with the smallest auxiliary violation, then the smallest contact-descent violation.

## Variants

| Method ID | Method | Contact descent ratio | Aux tolerance |
|---|---|---:|---:|
| `M8a` | `contact_qp_grid` | 0.60 | 0.08 |
| `M8b` | `contact_qp_grid` | 0.60 | 0.04 |

`M8a` matches M7c's auxiliary slack. `M8b` tests whether stricter auxiliary tolerance improves candidate behavior without losing too much contact descent.

## Reduced Gate

Run on reduced IL7RA with:

- target length 48;
- binder length 24;
- steps 3;
- seeds 2;
- top-k candidate handoff with 8 samples;
- sampling temperature 0.25;
- methods `M3,M7c,M8a,M8b`.

## Acceptance Check

The prototype is positive only if M8a or M8b beats M3 top-k under matched candidate budget on BT PAE or BT ipTM while preserving better update safety than M3.

If M8 only improves update safety but loses candidate quality, then the next step should be gradual position-wise hardening or a warm-started CEM diagnostic rather than scaling QP.
