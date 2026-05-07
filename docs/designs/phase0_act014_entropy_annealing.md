# ACT-014 Entropy-Annealed Contact-Preserving Update

Date: 2026-05-07

## Goal

Move the ACT-013 cold terminal handoff signal inside the optimizer.

ACT-013 showed that M7c can beat M3 under matched top-k budget when terminal sampling is colder, especially at temperature `0.25`. ACT-014 tests the next mechanism question:

> Can a contact-preserving update produce a colder, more discretization-ready relaxed sequence during optimization while retaining the update-safety advantage over M3?

## Method

Add entropy-annealed variants of M7c:

- `M7d`: M7c contact-preserving direction plus per-step terminal sharpening to final temperature `0.5`
- `M7e`: M7c contact-preserving direction plus per-step terminal sharpening to final temperature `0.25`

The sharpening schedule is linear over update steps. After each projected simplex update, the relaxed sequence is sharpened by:

```text
x_new = softmax(log(clip(x_projected)) / tau_step)
```

The actual update used for directional-derivative and oracle-harm logging is `x_new - x`, not the pre-sharpened projection. This keeps the safety metrics honest.

## Controls

- Keep `M3` naive weighted as the main reduced-run candidate baseline.
- Keep `M7c` as the non-annealed contact-preserving control.
- Use matched candidate budget across all methods.
- Evaluate `soft`, `argmax`, and top-k candidates.

## First Gate

Reduced IL7RA ProtenixMini:

- methods: `M3,M7c,M7d,M7e`
- steps: 3
- seeds: 2
- top-k samples: 8
- top-k amino acids per residue: 4
- candidate sampling temperature: `0.25`
- rerank metrics: BT PAE, BT ipTM, contact

## Interpretation

- If `M7d` or `M7e` beats M3 top-k while preserving lower update harm than M3, then entropy annealing is a viable optimizer-side repair.
- If annealed variants only win by post-hoc top-k but lose update safety, the sharpening is too aggressive.
- If annealed variants do not improve over M7c, the next repair should be straight-through hard candidate scoring rather than post-update entropy annealing.
