# Shampoo raw-data audit

- Generated (UTC): 2026-11-02T07:06:48.953980+00:00
- Observational unit: one shampoo formulation in the source JSON
- Parsed formulations: 812
- Raw SHA-256: `3A195870782E6FD87BDD7499CBB4FE201D025B2CDA05F9EEE9FB9477F34B58BD`
- Provenance hash check: **PASS**

## Safeguards

- Raw input is read-only; no row deletion, imputation, or composition renormalization occurred.
- Numeric zero and an absent field remain distinct.
- No split, threshold selection, or predictive performance result is produced here.

## Automatic mapping candidates

- Stability: `Stability_Test`
- Inputs: 18 wide numeric formulation columns after excluding seven documented outcomes/metadata columns.
- BASF reference CSV: 12 documented surfactant columns; the other six composition inputs are retained.
- Figures: `figures/stability_label_distribution.png`, `figures/ingredients_per_formulation.png`, `figures/ingredient_usage_frequency.png`.

## Next decision

Confirm mappings for stability, turbidity, viscosity, ingredient identity, and input quantity. Then define formulation-combination groups before group-aware model evaluation.
