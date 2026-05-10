# Boltz2 Candidate Holdout Scoring

Run ID: `boltz2_candidate_holdout_act021_m3_m7c_m8a_24_20260510T134114Z`

## Scope

This post-hoc check scores Phase 0 Protenix candidates with Boltz2 finite outputs. It intentionally uses distogram, pLDDT, and PAE metrics only; coordinate-level structure output is recorded but not treated as a pass condition.

## Summary

- Candidates scored: 24
- Finite distogram/pLDDT/PAE for all candidates: True
- Any finite structure coordinates: False
- Mean Boltz2 inter-PAE: 19.4981
- Best Boltz2 sequence: `RFKIRATSKVNSPEGTFITISFWK`
- Pearson Protenix bt_PAE vs Boltz2 inter-PAE: 0.39632985643654145
- Pearson Protenix contact vs Boltz2 contact@12A: 0.33327934499002626

## Top Boltz2 Candidates

| Rank | Method | Mode | Sequence | Protenix bt PAE | Boltz2 inter PAE | Boltz2 contact@12A | Boltz2 pLDDT |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | M3 | topk_sample | `RFKIRATSKVNSPEGTFITISFWK` | 13.604653358459473 | 13.9913 | 0.2263 | 0.4742 |
| 2 | M7c | topk_sample | `LRWPRPHYKPDSVNGNFITIPFWP` | 14.352824211120605 | 15.4656 | 0.2258 | 0.4948 |
| 3 | M3 | topk_sample | `RISFRANYKVNSDDGQFITIWFPD` | 12.35933780670166 | 15.7529 | 0.2345 | 0.4643 |
| 4 | M7c | topk_sample | `LRWPWPDIKPDGVNGTFITYSWWP` | 12.505476951599121 | 17.3908 | 0.2195 | 0.4921 |
| 5 | M8a | topk_sample | `LRWPWPHYKPDSDNGNMITYDFPD` | 13.373279571533203 | 18.1562 | 0.2147 | 0.4833 |
| 6 | M8a | soft | `LRWPRPHYKPDGDNGNFITYPFWD` | 13.253604888916016 | 18.6866 | 0.2188 | 0.4618 |
| 7 | M7c | topk_sample | `LRFPRGNVKPDGDDGNFIEYDYWP` | 14.78978157043457 | 18.7359 | 0.2293 | 0.4624 |
| 8 | M8a | topk_sample | `LRFPMPMYKPDADNGNFIEYPYWD` | 19.3974552154541 | 18.7418 | 0.2205 | 0.4866 |
| 9 | M8a | topk_sample | `LRIPWPMIKPDGDNGTFITYSWWP` | 14.8756103515625 | 18.8414 | 0.2285 | 0.4866 |
| 10 | M3 | topk_sample | `RIKIRATYKVNSPEGNFITIRFWK` | 13.897786140441895 | 19.0175 | 0.2435 | 0.4477 |

## Outputs

- CSV: `docs/results/boltz2_candidate_holdout_act021_m3_m7c_m8a_24_20260510T134114Z.csv`
- JSON: `docs/results/boltz2_candidate_holdout_act021_m3_m7c_m8a_24_20260510T134114Z.json`
