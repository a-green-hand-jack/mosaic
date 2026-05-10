# Boltz2 Candidate Holdout Scoring

Run ID: `boltz2_candidate_holdout_cem2dd3965_top5_20260510T132026Z`

## Scope

This post-hoc check scores Phase 0 Protenix candidates with Boltz2 finite outputs. It intentionally uses distogram, pLDDT, and PAE metrics only; coordinate-level structure output is recorded but not treated as a pass condition.

## Summary

- Candidates scored: 5
- Finite distogram/pLDDT/PAE for all candidates: True
- Any finite structure coordinates: False
- Mean Boltz2 inter-PAE: 18.3960
- Best Boltz2 sequence: `RIHVRATYEVNGDNGNFITRPFWP`
- Pearson Protenix bt_PAE vs Boltz2 inter-PAE: 0.28994315675342636
- Pearson Protenix contact vs Boltz2 contact@12A: -0.44246496206524216

## Top Boltz2 Candidates

| Rank | Method | Mode | Sequence | Protenix bt PAE | Boltz2 inter PAE | Boltz2 contact@12A | Boltz2 pLDDT |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | M3 | topk_sample | `RIHVRATYEVNGDNGNFITRPFWP` | 16.61977767944336 | 15.1898 | 0.2383 | 0.4912 |
| 2 | M3 | soft | `RVWFRPTYEVNGKNGNFNTRPFWK` | 13.170269966125488 | 18.1063 | 0.2205 | 0.5010 |
| 3 | M3 | topk_sample | `RIWFRANYKVNGKGCTFNTKPYPD` | 15.550691604614258 | 19.1335 | 0.2325 | 0.4448 |
| 4 | M3 | topk_sample | `RFRIRPSYEVNAKNGNFNTRPYWK` | 18.527721405029297 | 19.4788 | 0.2338 | 0.4642 |
| 5 | M3 | topk_sample | `RVHFRPTYEVNSDNGQFNTKPFPD` | 18.157028198242188 | 20.0716 | 0.2201 | 0.4646 |

## Outputs

- CSV: `docs/results/boltz2_candidate_holdout_cem2dd3965_top5_20260510T132026Z.csv`
- JSON: `docs/results/boltz2_candidate_holdout_cem2dd3965_top5_20260510T132026Z.json`
