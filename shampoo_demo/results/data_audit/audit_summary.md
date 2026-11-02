# Shampoo raw-data audit

- Generated (UTC): 2026-11-02T06:56:12+00:00
- Observational unit: one shampoo formulation in the source JSON
- Parsed formulations: 812
- Raw SHA-256: `3A195870782E6FD87BDD7499CBB4FE201D025B2CDA05F9EEE9FB9477F34B58BD`
- Provenance hash check: **PASS**

## Safeguards

- Raw input is read-only; no row deletion, imputation, or composition renormalization occurred.
- Numeric zero and an absent field remain distinct.
- No split, threshold selection, or predictive performance result is produced here.

## Top-level field availability

| Field | Present records | Completeness |
|---|---:|---:|
| `Arlypon F` | 812 | 100.0% |
| `Arlypon TT` | 812 | 100.0% |
| `Dehyquart A-CA` | 812 | 100.0% |
| `Dehyquart CC6` | 812 | 100.0% |
| `Dehyquart CC7 Benz` | 812 | 100.0% |
| `Dehyton AB 30` | 812 | 100.0% |
| `Dehyton MC` | 812 | 100.0% |
| `Dehyton ML` | 812 | 100.0% |
| `Dehyton PK 45` | 812 | 100.0% |
| `ID` | 812 | 100.0% |
| `Luviquat Excellence` | 812 | 100.0% |
| `Plantacare 2000` | 812 | 100.0% |
| `Plantacare 818` | 812 | 100.0% |
| `Plantapon ACG 50` | 812 | 100.0% |
| `Plantapon Amino KG-L` | 812 | 100.0% |
| `Plantapon Amino SCG-L` | 812 | 100.0% |
| `Plantapon LC 7` | 812 | 100.0% |
| `Rheology_Data` | 812 | 100.0% |
| `Rheology_Type` | 812 | 100.0% |
| `Salcare Super 7` | 812 | 100.0% |
| `Stability_Test` | 812 | 100.0% |
| `Texapon SB 3 KC` | 812 | 100.0% |
| `Turbidity_Error` | 812 | 100.0% |
| `Turbidity_NTU` | 812 | 100.0% |
| `Viscosity` | 812 | 100.0% |

## Automatic mapping candidates

- Stability: `Stability_Test`
- Inputs: 18 wide numeric formulation columns after excluding seven documented outcomes/metadata columns.
- BASF reference CSV: 12 of those columns are explicitly documented surfactants; the other six remain composition inputs and are retained.
- Nested ingredient list: absent (expected for this wide-format source).
- Figures: `figures/stability_label_distribution.png`, `figures/ingredients_per_formulation.png`, `figures/ingredient_usage_frequency.png`.

## Nested scalar field coverage (top 30)

| Field path | Present records |
|---|---:|
| `Texapon SB 3 KC` | 812 |
| `Plantapon ACG 50` | 812 |
| `Plantapon LC 7` | 812 |
| `Plantacare 818` | 812 |
| `Plantacare 2000` | 812 |
| `Dehyton MC` | 812 |
| `Dehyton PK 45` | 812 |
| `Dehyton ML` | 812 |
| `Dehyton AB 30` | 812 |
| `Plantapon Amino SCG-L` | 812 |
| `Plantapon Amino KG-L` | 812 |
| `Dehyquart A-CA` | 812 |
| `Luviquat Excellence` | 812 |
| `Dehyquart CC6` | 812 |
| `Dehyquart CC7 Benz` | 812 |
| `Salcare Super 7` | 812 |
| `Arlypon F` | 812 |
| `Arlypon TT` | 812 |
| `Stability_Test` | 812 |
| `Turbidity_NTU` | 812 |
| `Turbidity_Error` | 812 |
| `Viscosity` | 812 |
| `Rheology_Type` | 812 |
| `ID` | 812 |
| `Rheology_Data` | 518 |
| `Rheology_Data[]<object>` | 294 |

## Next decision

Confirm mappings for stability, turbidity, viscosity, ingredient identity, and input quantity. Then define formulation-combination groups before group-aware model evaluation.
