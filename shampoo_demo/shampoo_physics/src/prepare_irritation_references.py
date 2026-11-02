"""Audit public single-chemical skin/eye labels against shampoo ingredient representatives.

The outputs are reference evidence only. They do not predict irritation of a shampoo mixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from common import ROOT, sha256, write_csv, write_json

SOURCE_COMMIT = "1e5de1a4310feb17c576de7a54431f1ea56e8f9f"
SOURCE_URL = "https://github.com/yxliu0907/SkinEye"
ENDPOINT_FILES = {
    "skin": ROOT / "sources/skin_eye_chemical_skin.csv",
    "eye": ROOT / "sources/skin_eye_chemical_eye.csv",
}


def canonicalize(smiles: object) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


def load_references(path):
    frame = pd.read_csv(path)
    if set(frame.columns) != {"multi", "SMILES", "class"}:
        raise ValueError(f"Unexpected source columns in {path.name}: {list(frame.columns)}")
    if not frame["class"].isin([0, 1]).all():
        raise ValueError(f"Non-binary labels in {path.name}")
    frame["canonical_smiles"] = frame.SMILES.map(canonicalize)
    valid = frame.dropna(subset=["canonical_smiles"]).copy()
    grouped = valid.groupby("canonical_smiles", sort=True)["class"]
    references = grouped.agg([("label_min", "min"), ("label_max", "max"), ("rows", "size")]).reset_index()
    references["label_status"] = np.where(
        references.label_min.eq(references.label_max),
        references.label_min.astype(str),
        "conflicting",
    )
    return frame, valid, references


def run():
    registry = pd.read_csv(ROOT / "data/ingredient_registry.csv")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    audit_rows = []
    endpoint_summary = {}

    for endpoint, path in ENDPOINT_FILES.items():
        raw, valid, references = load_references(path)
        molecules = [Chem.MolFromSmiles(s) for s in references.canonical_smiles]
        fingerprints = [generator.GetFingerprint(m) for m in molecules]
        reference_index = {s: i for i, s in enumerate(references.canonical_smiles)}

        endpoint_summary[endpoint] = {
            "rows": len(raw),
            "valid_smiles_rows": len(valid),
            "invalid_smiles_rows": len(raw) - len(valid),
            "unique_valid_structures": len(references),
            "positive_rows": int(raw["class"].sum()),
            "negative_rows": int((raw["class"] == 0).sum()),
            "conflicting_unique_structures": int((references.label_status == "conflicting").sum()),
        }

        for ingredient in registry.itertuples(index=False):
            result = {
                "endpoint": endpoint,
                "ingredient_id": ingredient.ingredient_id,
                "inci": ingredient.inci,
                "representative_only": True,
                "exact_structure_match": False,
                "exact_reference_class": "",
                "nearest_tanimoto": np.nan,
                "nearest_reference_smiles": "",
                "nearest_reference_class": "",
                "use": "reference_only_not_shampoo_prediction",
            }
            mol = (
                Chem.MolFromSmiles(str(ingredient.representative_smiles))
                if pd.notna(ingredient.representative_smiles)
                else None
            )
            if mol is not None:
                canonical = Chem.MolToSmiles(mol)
                fingerprint = generator.GetFingerprint(mol)
                similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
                nearest = int(np.argmax(similarities))
                result.update(
                    exact_structure_match=canonical in reference_index,
                    nearest_tanimoto=float(similarities[nearest]),
                    nearest_reference_smiles=references.iloc[nearest].canonical_smiles,
                    nearest_reference_class=references.iloc[nearest].label_status,
                )
                if canonical in reference_index:
                    exact = references.iloc[reference_index[canonical]]
                    result["exact_reference_class"] = exact.label_status
            audit_rows.append(result)

    audit = pd.DataFrame(audit_rows)
    summary = {
        "source": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "license": "MIT",
        "source_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in ENDPOINT_FILES.values()},
        "endpoints": endpoint_summary,
        "shampoo_representative_exact_matches": {
            endpoint: int(group.exact_structure_match.sum()) for endpoint, group in audit.groupby("endpoint")
        },
        "intended_use": "Ingredient-level chemical reference audit only",
        "not_validated_for": [
            "shampoo mixture irritation prediction",
            "commercial ingredient distribution prediction",
            "product safety or regulatory decisions",
        ],
        "label_note": "The source provides binary chemical-level class labels. Nearest-neighbor labels are not predictions.",
    }
    write_csv(ROOT / "data/skin_eye_chemical_reference_audit.csv", audit)
    write_json(ROOT / "data/skin_eye_chemical_reference_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    run()
