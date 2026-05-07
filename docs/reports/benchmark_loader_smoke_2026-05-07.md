# Benchmark Loader Smoke Check

Date: 2026-05-07

Scope: CPU-only validation that the currently downloaded benchmark source structures can pass the existing Mosaic data-loading / feature-construction entry points. This did not instantiate or run AF2, Protenix, or Boltz2 model weights.

## Environment

- Server: Quest
- Code path: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/code`
- Code commit: `2c42ede`
- Raw source path: `/projects/p32572/Jieke/Projects/SCH-BinderDesign/data/benchmarks/targets/raw/rcsb`
- Boltz2 cache: `/projects/p32572/Jieke/.cache/boltz`
- Protenix cache: `/projects/p32572/Jieke/.cache/protenix`
- CPU-only setting: `JAX_PLATFORMS=cpu`

## Cache Fix

Protenix featurization initially failed before reading any target-specific content because the installed package hardcodes `~/.protenix` for CCD/component data:

```text
FileNotFoundError: /home/zpt6685/.protenix/components.v20240608.cif
```

I fixed this by exposing the shared Quest cache at the package's default location:

```bash
mkdir -p /projects/p32572/Jieke/.cache/protenix
if [ ! -e ~/.protenix ]; then ln -s /projects/p32572/Jieke/.cache/protenix ~/.protenix; fi
uv run python -c "from protenix.backend import download_data; download_data()"
```

After this, the Protenix data cache contained `components.v20240608.cif`, `components.v20240608.cif.rdkit_mol.pkl`, and `clusters-by-entity-40.txt`.

## Loader Entrypoints

- AF2: `mosaic.models.af2.make_af_features(chains)`
- Protenix: `protenix.data.template.featurize([...ChainInput(..., compute_msa=False, template=...)])`
- Boltz2: `mosaic.models.boltz2` YAML/template helpers plus `mosaic.losses.boltz2.load_features_and_structure_writer(..., cache=/projects/p32572/Jieke/.cache/boltz)`

The smoke script stripped non-polymer residues from the template chains before constructing `TargetChain(sequence=..., use_msa=False, template_chain=...)`. This avoids hetero-residue noise while still validating the selected polymer chains and template coordinate paths.

## Results

| Target slice | Chains | AF2 loader | Protenix loader | Boltz2 loader |
|---|---:|---:|---:|---:|
| `T01_il7ra_7opb_A` | `A:196` | OK, `aatype=(196,)` | OK, `token_index=(196,)` | OK, `token_index=(1, 196)` |
| `T02_pdl1_4zqk_A` | `A:115` | OK, `aatype=(115,)` | OK, `token_index=(115,)` | OK, `token_index=(1, 115)` |
| `T03_pd1_4zqk_B` | `B:106` | OK, `aatype=(106,)` | OK, `token_index=(106,)` | OK, `token_index=(1, 106)` |
| `T07_cd47_2jjs_C` | `C:115` | OK, `aatype=(115,)` | OK, `token_index=(115,)` | OK, `token_index=(1, 115)` |
| `T11_vegfa_1vpf_AB` | `A:94,B:94` | OK, `aatype=(188,)` | OK, `token_index=(188,)` | OK, `token_index=(1, 188)` |
| `T13_il6_1alu_A` | `A:157` | OK, `aatype=(157,)` | OK, `token_index=(157,)` | OK, `token_index=(1, 157)` |
| `T17_il1b_9ilb_A` | `A:153` | OK, `aatype=(153,)` | OK, `token_index=(153,)` | OK, `token_index=(1, 153)` |
| `T19_pcsk9_2p4e_PA` | `P:92,A:494` | OK, `aatype=(586,)` | OK, `token_index=(586,)` | OK, `token_index=(1, 586)` |

## Interpretation

The currently downloaded Wave 1 structures are loader-ready for AF2, Protenix, and Boltz2 under CPU-only feature construction. Remaining checks before using these as model-backed oracles are model-load/model-forward checks on compute nodes, especially Boltz2 no-download model initialization against `/projects/p32572/Jieke/.cache/boltz`.
