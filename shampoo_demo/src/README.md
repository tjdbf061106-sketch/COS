# Analysis scripts

Each script uses `# %%` cells, so it can be read and run like a lightweight notebook in VS Code/Jupyter while also running from a terminal.

| Order | Script | Purpose | Raw-data mutation |
|---:|---|---|---|
| 01 | `01_audit_raw_dataset.ipynb` | Verify the frozen source, inspect aggregate schema/completeness, and save useful distribution figures. | None |

Run from the repository root:

```powershell
jupyter lab shampoo_demo/src/01_audit_raw_dataset.ipynb
```

The script writes only derived files under `../results/data_audit/`. It does not train a model, choose a threshold, impute values, or merge skin/eye datasets.
