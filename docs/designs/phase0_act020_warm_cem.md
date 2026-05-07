# Phase 0 ACT-020 Warm-Started Hard-Candidate Optimization

## Motivation

ACT-015A showed that naive CEM from a random relaxed distribution collapses entropy around poor hard candidates. ACT-019 showed that irreversible early hardening also collapses flexibility too soon. ACT-020 tests the middle path:

> start hard-candidate optimization only after a useful relaxed trajectory has already found a basin.

This directly tests whether the cold top-k handoff can become an optimizer stage rather than a terminal-only reranking trick.

## Variants

| Method | Init source | Elite metric | Purpose |
|---|---|---|---|
| `WCEMp_M3` | M3 terminal distribution | BT PAE | Can M3's strong discrete behavior be improved by elite updates? |
| `WCEMc_M3` | M3 terminal distribution | contact loss | Does contact-focused elite selection help M3 terminals? |
| `WCEMp_M7c` | M7c terminal distribution | BT PAE | Can the cold-handoff M7c signal be internalized? |
| `WCEMc_M7c` | M7c terminal distribution | contact loss | Does contact-focused elite selection rescue M7c terminals? |
| `WCEMp_M8a` | M8a terminal distribution | BT PAE | Can contact-aggressive QP soft quality be converted into hard candidates? |
| `WCEMc_M8a` | M8a terminal distribution | contact loss | Does contact selection repair M8a's high-harm but strong-contact basin? |

Each warm-started CEM run:

1. runs the source baseline trajectory;
2. mixes the terminal distribution with a small uniform component;
3. samples hard top-k candidates at temperature `0.25`;
4. scores hard candidates with the reduced ProtenixMini candidate oracle;
5. updates the distribution toward elite hard candidates;
6. repeats for 3 rounds.

## Controls

- `M3` top-k with the same 24-candidate budget.
- `M7c` cold top-k handoff.
- `M8a` contact-aggressive QP control.
- ACT-015A random CEM and ACT-019 hardening are historical negative controls.

## Gate

Positive only if a warm-started method beats M3 under matched hard-candidate budget on BT PAE/contact or BT PAE/BT ipTM, while the interpretation is clear about whether the improvement is a post-hoc candidate optimizer or a safer update rule.

If the warm-started method wins only by using extra hard-candidate scoring, report it as an oracle-budget-matched candidate optimizer, not as an update-safety claim.
