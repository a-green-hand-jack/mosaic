# ACT-015 Method Menu After ACT-014

Date: 2026-05-07

## Context

ACT-013 showed that cold top-k handoff can recover strong M7c discrete candidates. ACT-014 showed that naive post-update entropy annealing is not the right optimizer-side repair: it lowers entropy, but worsens update safety and candidate quality.

The next step should not be another aggressive entropy schedule. The useful design space is broader:

- discrete candidate expected-quality optimization;
- constrained multi-objective updates;
- gradual discretization rather than terminal argmax;
- adaptive use of M3 and M7c;
- oracle reliability controls.

## Design Decision

Prioritize methods that directly address the relaxed-to-discrete gap while keeping the comparison against M3 budget-matched.

The recommended order is:

1. `ACT-015A`: CEM / elite-sampling optimizer.
2. `ACT-015B`: QP-style constrained multi-objective update.
3. `ACT-015C`: gradual position-wise hardening.

Straight-through and weak discreteness regularization remain viable, but they are now secondary to the CEM direction because CEM most directly internalizes the successful ACT-013 cold top-k handoff.

## ACT-015A: CEM / Elite-Sampling Optimizer

### Goal

Optimize expected hard-candidate quality rather than relying on relaxed loss to transfer to discrete candidates.

### Mechanism

At each update round:

1. Sample `N` hard sequences from the current relaxed distribution with cold top-k or categorical sampling.
2. Score each hard candidate with the same oracle metrics used for candidate evaluation.
3. Select the top `rho` elite candidates by a scalar or constrained ranking rule.
4. Update the relaxed distribution toward the empirical elite distribution.
5. Repeat for a small number of rounds.

This turns the ACT-013 cold top-k handoff into the optimizer instead of applying it only at the terminal step.

### First Prototype

- Methods: `M3` top-k baseline, `M7c` cold handoff, `CEM-contact`, `CEM-constrained`.
- Candidate budget: fixed and matched; for example `N=8` hard sequences per round, with the M3 comparison receiving the same oracle-call budget.
- Elite fraction: `rho=0.25` or top 2 of 8.
- Ranking metric: start with BT PAE, then repeat with BT ipTM/contact.
- Distribution update: convex update toward elite amino-acid frequencies with a trust coefficient.

### Acceptance Check

CEM beats M3 top-k under matched oracle-call budget at temperature `0.25` on BT PAE or BT ipTM without producing worse sequence sanity metrics.

### Failure Modes

- Too expensive if every update requires many Protenix calls.
- Overfits to one reranking metric.
- Collapses diversity too early without a trust-region update.

## ACT-015B: QP-Style Constrained Update

### Goal

Make the M7c contact-preserving rule formal and controllable, with per-oracle harm tolerances.

### Mechanism

Choose an update direction `d` close to the contact direction while satisfying local descent constraints:

```text
minimize_d ||d - d_contact||^2
subject to g_contact^T d <= -epsilon
           g_i^T d <= tau_i for auxiliary oracle i
```

This is a constrained multi-objective optimizer rather than a heuristic cone search.

### First Prototype

Use a small candidate-set approximation before adding a true QP solver:

- enumerate the existing cone candidate directions;
- score each by distance to contact direction;
- filter by contact descent and per-oracle tolerances;
- select the feasible direction closest to contact.

This keeps the implementation local to the current diagnostic code and avoids new solver dependencies.

### Acceptance Check

The constrained update preserves or improves M7c update safety while improving top-k candidate quality against M3 under matched budget.

### Failure Modes

- Constraints may be infeasible.
- Contact descent may dominate and recreate single-oracle failure modes.
- Per-oracle tolerances can become hidden tuning knobs.

## ACT-015C: Gradual Position-Wise Hardening

### Goal

Avoid a single terminal argmax event by committing only stable, low-risk positions during optimization.

### Mechanism

Every few steps:

1. Identify positions with high top-1 probability or stable top-1 identity across cold samples.
2. Test whether hardening the position worsens contact or auxiliary losses.
3. Freeze safe positions to hard amino acids.
4. Continue relaxed optimization over the remaining positions.

### First Prototype

Start without per-position Protenix rescoring, using cheap criteria:

- top-1 probability threshold;
- top-1 vs top-2 margin;
- consistency across cold samples;
- exclude positions whose hardening would strongly increase sequence sanity losses.

Only after this passes proxy checks should per-position Protenix rescoring be added.

### Acceptance Check

Gradual hardening improves argmax or top-k candidate quality over M7c without the ACT-014 step-norm and harm increase.

### Failure Modes

- Early wrong commitments are hard to recover from.
- Interface-critical positions may be hardened too early.
- Position-wise decisions may ignore pairwise residue dependencies.

## Secondary Directions

| Direction | Status | Reason |
|---|---|---|
| Straight-through hard scoring | Keep as fallback | Directly targets hard candidates but may be unstable with Protenix gradients. |
| Weak discreteness regularizer | Keep as fallback | Safer than ACT-014 but less directly tied to ACT-013 than CEM. |
| Trust-region updates | Add to CEM/QP/hardening | ACT-014 suggests drift control is required. |
| Margin regularization | Add later | More targeted than entropy but needs position sensitivity rules. |
| M3/M7c hybrid schedule | Useful control | M3 is a strong discrete baseline and may be a method component. |
| Soft-to-hard gap surrogate | Later diagnostic | Useful after more trajectories exist; too early now. |
| Oracle reliability weighting | Later extension | Valuable once more oracles and targets are included. |
| Pareto candidate pool | Later scale-up | Promising but changes compute budget and system complexity. |

## Recommended Immediate Implementation

Implement `ACT-015A` first as a small CEM prototype.

Minimum viable run:

- reduced IL7RA;
- `num_seeds=2`;
- `rounds=3`;
- `samples_per_round=8`;
- elite count `2`;
- matched M3 top-k oracle budget;
- report BT PAE, BT ipTM, pLDDT, contact loss, sequence sanity, and candidate diversity.

Decision after first CEM run:

- If CEM beats M3 under matched budget, scale CEM to more seeds and test QP as a formal update baseline.
- If CEM only wins with too much budget, revise budget matching before continuing.
- If CEM fails but preserves diversity, test QP constrained update.
- If CEM collapses or worsens all metrics, move to gradual position-wise hardening.
