# Phase 0 Contact-Preserving Update

Date: 2026-05-06

## Scope

This run tests a first contact-preserving safe update variant, `contact_preserving_soft_cone`. The method treats the first oracle as primary, which is `protenix_contact` in the Protenix diagnostic. It searches candidate directions that preserve primary contact descent while allowing small slack on non-primary sequence sanity oracles.

Run ID: `phase0_protenix_update_geometry_36ec419_20260506T201747Z`

Setup: H100, `steps=3`, `num_seeds=2`, target length 48, binder length 24, ProtenixMini, argmax candidate scoring.

## Update Geometry

| Method | Harm rate | Worst derivative | Final contact loss | Final trigram loss | Final solubility loss |
|---|---:|---:|---:|---:|---:|
| normalized_weighted | 0.000 | -0.0090 | 2.856 | 3.077 | 0.006 |
| soft_cone_correction | 0.000 | -0.0090 | 2.856 | 3.077 | 0.006 |
| contact_preserving_soft_cone | 0.083 | 0.0005 | 2.686 | 3.219 | 0.059 |
| naive_weighted | 0.167 | 0.0110 | 2.440 | 3.280 | 0.067 |
| single_protenix_contact | 0.333 | 0.0301 | 2.536 | 3.366 | 0.104 |

## Candidate Holdout

| Method | pLDDT | BT PAE | BT ipTM | IPSAE min | Contact loss | Trigram loss | Hydrophobic mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| single_protenix_contact | 0.573 | 14.066 | 0.402 | 0.178 | 2.923 | 3.215 | 0.313 |
| naive_weighted | 0.542 | 14.281 | 0.314 | 0.186 | 2.938 | 3.131 | 0.354 |
| contact_preserving_soft_cone | 0.476 | 15.899 | 0.254 | 0.127 | 3.270 | 3.223 | 0.333 |
| normalized_weighted | 0.470 | 16.434 | 0.186 | 0.087 | 3.512 | 2.770 | 0.229 |
| soft_cone_correction | 0.470 | 16.434 | 0.186 | 0.087 | 3.512 | 2.770 | 0.229 |

## Interpretation

The contact-preserving variant moved in the intended direction. Relative to normalized weighted and soft-cone correction, it improved candidate-level interface metrics:

- binder-target PAE decreased from 16.43 to 15.90
- binder-target ipTM increased from 0.186 to 0.254
- IPSAE increased from 0.087 to 0.127
- contact loss decreased from 3.51 to 3.27

It did this while keeping update harm lower than naive weighted: 0.083 versus 0.167. This suggests the basic design direction is useful.

The variant is still not strong enough. Naive weighted and single-Protenix remain better on candidate interface metrics in this reduced run. The next iteration should tune the primary-preservation slack or explicitly constrain a minimum contact descent ratio, rather than only allowing small non-primary slack.
