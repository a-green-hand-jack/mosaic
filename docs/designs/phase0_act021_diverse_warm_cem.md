# Phase 0 ACT-021 Diversity-Preserving Warm CEM

## Motivation

ACT-020 showed that warm-started CEM can repair terminal argmax quality, but did not beat the source cold top-k pools at the matched 24-candidate budget. The likely failure mode is over-commitment: elite updates pull the relaxed distribution toward a few sampled candidates, which helps argmax but can erase the diversity that made source top-k reranking strong.

ACT-021 tests whether CEM should be a conservative pool refiner rather than a replacement for the source terminal distribution.

## Hypothesis

If ACT-020 lost because elite updates removed too much source diversity, then a diversity-preserving warm CEM should:

1. keep or improve the ACT-020 argmax repair;
2. improve best-of-24 BT PAE/contact over the standard warm CEM variants;
3. close the gap to source M7c/M8a top-k reranking under the same hard-candidate budget.

If it only improves argmax but not best-of-24, then the issue is not just source-diversity loss. The harder conclusion would be that fixed terminal top-k already samples a stronger local candidate pool than the current iterative elite updates.

## Variants

| Method | Init source | Elite metric | Update rule | Purpose |
|---|---|---|---|---|
| `WCEMp_*` | baseline terminal | BT PAE | standard warm CEM | ACT-020 control |
| `WCEMc_*` | baseline terminal | contact loss | standard warm CEM | ACT-020 contact control |
| `DWCEMp_*` | baseline terminal | BT PAE | lower elite rate + source mixture | Test whether preserving source diversity improves BT PAE reranking |
| `DWCEMc_*` | baseline terminal | contact loss | lower elite rate + source mixture | Test whether preserving source diversity improves contact reranking |

`DWCEM*` uses:

- `--diverse-cem-update-rate 0.3`
- `--diverse-source-mix 0.35`

After each elite update, the distribution is mixed back toward the warm-start source distribution. This makes the CEM step a local refinement around the source terminal basin instead of a one-way collapse toward elites.

## Controls

- Source `M3`, `M7c`, and `M8a` top-k candidate pools with 24 hard-candidate scores.
- Standard ACT-020 `WCEMp` and `WCEMc` in the same run.
- Top-k sensitivity budgets `1,8,24` for `bt_pae`, `bt_iptm`, and `contact`.

## Decision Gate

Call ACT-021 positive only if a `DWCEM*` variant improves over the corresponding `WCEM*` variant and either:

- beats the best source baseline under matched budget on BT PAE/contact; or
- materially closes the source-top-k gap while preserving ACT-020's argmax repair.

If `DWCEM*` still trails source top-k, report warm CEM as an argmax repair and diagnostic tool, not as the current best candidate selection strategy.
