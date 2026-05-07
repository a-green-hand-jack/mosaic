# ACT-013 Terminal Sampling Temperature Sweep

Date: 2026-05-07

## Goal

Test whether ACT-012's weak M7c top-k result is caused by a terminal handoff that samples too broadly from the relaxed sequence distribution.

The specific question is:

> Under a matched discrete candidate budget, does lowering top-k sampling temperature recover better M7a/M7c candidates relative to the current strongest reduced-run baseline, M3 naive weighted?

## Motivation

ACT-010 showed that contact-preserving geometry can improve the relaxed terminal sequence. ACT-011 showed that M7c can produce useful discrete candidates when top-k handoff is used. ACT-012 showed that this advantage is not robust: M3 remained best across budgets and reranking metrics.

This points to a terminal distribution problem rather than a pure update-geometry win. A lower sampling temperature is the smallest intervention because it changes only the discrete handoff while leaving optimization, oracle calls, seeds, and method definitions unchanged.

## Design

- Target: reduced IL7RA smoke setting.
- Methods: `M3`, `M7a`, `M7c`.
- Candidate mode: `both` plus top-k samples.
- Candidate budget: 8 top-k samples per method/seed.
- Top-k amino acids per residue: 4.
- Sampling temperatures: `0.5`, `0.25`.
- Reranking metrics: BT PAE, BT ipTM, contact.

ACT-012 at temperature `1.0` is the reference condition.

## Interpretation Rules

- If M7c beats M3 at lower temperature, then the next method change should focus on colder terminal distributions or entropy annealing.
- If M7c improves but still loses to M3, top-k temperature helps but does not solve the main candidate-level claim.
- If M7c does not improve, the next intervention should move inside optimization, for example straight-through hard candidate scoring or an explicit terminal discreteness regularizer.

## Acceptance Gate

Do not scale to more seeds or targets unless a geometry-aware method beats M3 under a matched discrete candidate budget on BT PAE or BT ipTM.
