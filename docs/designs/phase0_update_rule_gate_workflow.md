# Phase 0 Update-Rule Gate Workflow

Date: 2026-05-13

## Purpose

Phase 0 should now evaluate dynamic oracle-arbitration rules through a fixed
gate before spending candidate-level compute. The immediate baseline is `M4`
normalized weighting. Any new rule must first match or beat `M4` on update-level
metrics before it is promoted to Boltz2 candidate holdout.

## Gate Metrics

Primary gate:

- oracle harm rate no worse than `M4`;
- worst-oracle directional derivative no worse than `M4`;
- Protenix and Boltz2 directional derivatives negative on average;
- step norm at least half of `M4` unless the run explicitly changes the
  threshold;
- per-target zero-harm count reported, not hidden inside only an average.

## Result Aggregation

Use the gate summarizer after a Wave 1 run produces per-target JSON/CSV outputs:

```bash
python scripts/summarize_oracle_balance_results.py \
  docs/results/<target1>.json \
  docs/results/<target2>.json \
  docs/results/<target3>.json \
  docs/results/<target4>.json \
  docs/results/<target5>.json \
  --output-prefix docs/reports/phase0_<run_name>_gate_summary
```

The script writes:

- `docs/reports/phase0_<run_name>_gate_summary.md`
- `docs/reports/phase0_<run_name>_gate_summary.csv`
- `docs/reports/phase0_<run_name>_gate_summary.json`

The current sanity check on the completed Wave 1 conflict map reproduces the
known conclusion: `M4` is the baseline, while `M7c`, `M3`, and `M8a` all fail
the configured gate.

## Gradient Snapshot Export

Future expensive oracle runs should include:

```bash
--snapshot-dir docs/results/gradient_snapshots/<run_name>
```

Each snapshot stores:

- relaxed sequence state `x_t`;
- per-oracle gradient stack;
- oracle names and weights;
- source method, seed, step, and target metadata.

This keeps Protenix/Boltz2 compute separate from cheap update-rule iteration.

## Offline Replay

Replay saved snapshots before launching a new Quest run:

```bash
uv run python scripts/replay_gradient_snapshots.py \
  docs/results/gradient_snapshots/<run_name>/*.npz \
  --method-ids M3,M4,M7c,M8a,M10a,M11a \
  --step-size 0.25
```

Replay is not paper evidence because it reuses saved gradients. Its role is to
kill weak update rules before they consume structure-oracle compute.

## Current Candidate Rules

- `M10a=balanced_zero_harm_cone`: implemented before this workflow; EXP-B result
  still needs Quest verification.
- `M11a=pcgrad_normalized`: deterministic PCGrad-style projection over
  normalized oracle gradients. This is a cheap arbitration baseline to test
  against `M4`; it is not a claim until it passes replay and real oracle gates.

## Promotion Rule

Only promote a method to candidate-level Boltz2 holdout if it passes:

1. offline replay on saved snapshots;
2. Wave 1 one-step update gate against `M4`;
3. manual inspection for nontrivial step norm and per-target stability.
