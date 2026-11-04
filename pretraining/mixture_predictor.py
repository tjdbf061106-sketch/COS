"""Portable inference for audited mixture inputs from the mixture-003 package.

MLM learns ingredient SMILES without irritation labels. The four exported
primary heads predict formulation-level EPA classes from pooled ingredient
embeddings plus 11 composition/uncertainty features. They require all declared
amounts to be known. This module does not resolve ingredient identities or
complete a partially disclosed recipe.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch
from rdkit import Chem
from transformers import AutoModelForMaskedLM, AutoTokenizer

USABLE_STATUSES = {"usable_exact_structure", "usable_unspecified_stereochemistry"}
UNCERTAINTY_COLUMNS = [
    "declared_count", "represented_count", "unrepresented_count",
    "declared_pct_sum", "represented_pct", "unrepresented_declared_pct",
    "undeclared_pct", "zero_amount_count", "represented_unspecified_stereo_count",
    "has_missing_amount", "amount_unknown_count",
]
ALLOWED_ATOMS = {1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
MAX_TOKENS = 128


def audit_components(payload):
    """Check preconsolidated audit rows and calculate the training covariates.

Every declared row is required, including unrepresented and explicit-zero
ingredients. A missing structure is not a missing ingredient amount. The
caller must already have resolved source identity/duplicate/label conflicts;
these cannot be reconstructed from SMILES alone.
    """
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Input requires a nonempty audited components list")
    amounts, represented, ids = [], [], set()
    stereo_count = 0
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError(f"Component {index} must be an object")
        if any(k in component for k in ["label", "epa_class", "binary_label", "final_label"]):
            raise ValueError("Ingredient rows must not contain mixture toxicity targets")
        if not isinstance(component.get("structure_status"), str) or not component["structure_status"]:
            raise ValueError(f"Component {index} needs an audited structure_status")
        cid = component.get("component_id")
        if cid is not None:
            if cid in ids:
                raise ValueError("Duplicate component_id: provide consolidated ingredient rows")
            ids.add(cid)
        amount = component.get("amount_pct")
        if amount is None or isinstance(amount, bool):
            raise ValueError("Primary models require every declared amount; unknown is not zero")
        try:
            amount = float(amount)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Component {index} has an invalid reported amount") from exc
        if not math.isfinite(amount) or not 0 <= amount <= 100:
            raise ValueError("Reported amounts must be finite percentages in [0, 100]")
        amounts.append(amount)
        if amount == 0 or component["structure_status"] not in USABLE_STATUSES:
            continue
        smiles = component.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("A represented ingredient requires canonical_smiles")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("Represented structures must be valid connected molecular SMILES")
        if not any(a.GetAtomicNum() == 6 for a in molecule.GetAtoms()):
            raise ValueError("The primary encoder contract requires a discrete organic structure")
        if any(a.GetAtomicNum() not in ALLOWED_ATOMS for a in molecule.GetAtoms()):
            raise ValueError("Represented structure has atoms outside the audited scope")
        canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
        if canonical != smiles:
            raise ValueError("Use the audited canonical isomeric SMILES; no silent structural rewriting")
        unspecified = any(str(s.specified) == "Unspecified" for s in Chem.FindPotentialStereo(molecule))
        if unspecified != (component["structure_status"] == "usable_unspecified_stereochemistry"):
            raise ValueError("Structure status disagrees with the reported stereochemical information")
        stereo_count += int(unspecified)
        represented.append({"canonical_smiles": smiles, "amount_pct": amount,
                            "component_id": cid if cid is not None else str(index)})
    total = sum(amounts)
    if total > 100.01:
        raise ValueError("Declared percentage sum exceeds the primary eligibility threshold")
    if not represented:
        raise ValueError("At least one positive-amount usable ingredient is required")
    represented_pct = sum(c["amount_pct"] for c in represented)
    declared_count = sum(a > 0 for a in amounts)
    coverage = {
        "declared_count": declared_count,
        "represented_count": len(represented),
        "unrepresented_count": declared_count - len(represented),
        "declared_pct_sum": total,
        "represented_pct": represented_pct,
        "unrepresented_declared_pct": max(0.0, total - represented_pct),
        "undeclared_pct": max(0.0, 100.0 - total),
        "zero_amount_count": sum(a == 0 for a in amounts),
        "represented_unspecified_stereo_count": stereo_count,
        "has_missing_amount": 0,
        "amount_unknown_count": 0,
    }
    return represented, coverage


def pool_mixture(component_vectors, represented, coverage, pooling):
    vectors = np.asarray(component_vectors, dtype=np.float32)
    if vectors.shape != (len(represented), 384) or not np.isfinite(vectors).all():
        raise ValueError("Expected aligned finite 384-dimensional component embeddings")
    amounts = np.array([c["amount_pct"] for c in represented], dtype=float)
    if pooling == "mean":
        pooled = vectors.mean(axis=0)
    elif pooling == "weighted":
        pooled = np.average(vectors, axis=0, weights=amounts)
    else:
        raise ValueError("pooling must be mean or weighted")
    auxiliary = np.array([coverage[name] for name in UNCERTAINTY_COLUMNS], dtype=np.float32)
    return np.concatenate([pooled, auxiliary]).astype(np.float32)


class MixturePredictor:
    def __init__(self, package_dir=None, threads=4):
        self.package_dir = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        checkpoint = self.package_dir / "best"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        # Loading the trained MLM backbone avoids creating an unused random pooler.
        masked_model = AutoModelForMaskedLM.from_pretrained(checkpoint, local_files_only=True)
        self.encoder = masked_model.roberta.cpu().eval()
        if self.encoder.config.hidden_size != 384:
            raise ValueError("Unexpected encoder hidden size")
        self.heads = {}

    def encode_components(self, smiles, batch_size=16):
        if not smiles:
            raise ValueError("Empty represented ingredient collection")
        encoded = []
        special_ids = set(self.tokenizer.all_special_ids)
        for text in smiles:
            row = self.tokenizer(text, truncation=False, return_attention_mask=True)
            ids = row["input_ids"]
            if len(ids) > MAX_TOKENS:
                raise ValueError("Ingredient exceeds 128 tokens; truncation is not permitted")
            if self.tokenizer.unk_token_id is not None and self.tokenizer.unk_token_id in ids:
                raise ValueError("Ingredient contains tokens absent from the trained vocabulary")
            if not any(i not in special_ids for i in ids):
                raise ValueError("Ingredient has no non-special tokens")
            encoded.append(row)
        output = []
        for start in range(0, len(encoded), batch_size):
            batch = self.tokenizer.pad(encoded[start:start + batch_size], padding=True, return_tensors="pt")
            special = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
            for token in special_ids:
                special |= batch["input_ids"].eq(token)
            keep = batch["attention_mask"].bool() & ~special
            with torch.no_grad():
                hidden = self.encoder(**batch).last_hidden_state
                pooled = (hidden * keep.unsqueeze(-1).to(hidden.dtype)).sum(1) / keep.sum(1).unsqueeze(-1).to(hidden.dtype)
            output.append(pooled.cpu().numpy())
        values = np.concatenate(output)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite component representation")
        return values

    def vectorize(self, payload, pooling="weighted"):
        represented, coverage = audit_components(payload)
        embeddings = self.encode_components([c["canonical_smiles"] for c in represented])
        return pool_mixture(embeddings, represented, coverage, pooling), coverage

    def predict(self, payload, endpoint, pooling="weighted"):
        if endpoint not in {"eye_epa", "skin_epa"}:
            raise ValueError("endpoint must be eye_epa or skin_epa")
        vector, coverage = self.vectorize(payload, pooling)
        key = (endpoint, pooling)
        if key not in self.heads:
            path = self.package_dir / "mixture_heads" / f"{endpoint}__{pooling}.joblib"
            self.heads[key] = joblib.load(path)
        head = self.heads[key]
        if head.n_features_in_ != 395:
            raise ValueError("Unexpected mixture head input dimension")
        positive = list(head.classes_).index(1)
        score = float(head.predict_proba(vector.reshape(1, -1))[0, positive])
        return {
            "mixture_id": payload.get("mixture_id"), "endpoint": endpoint, "pooling": pooling,
            "probability_epa_i_ii": score, "predicted_binary_class": int(score >= 0.5),
            "threshold": 0.5, "label_definition": "1: EPA I/II; 0: EPA III/IV",
            "input_scope": "audited disclosed ingredients; incomplete formulations are explicitly represented",
            "coverage": coverage,
            "validation_scope": "internal FIFRA mixture research; not independently validated on cosmetic finished products",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON with all audited declared component rows")
    parser.add_argument("--endpoint", required=True, choices=["eye_epa", "skin_epa"])
    parser.add_argument("--pooling", default="weighted", choices=["mean", "weighted"])
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = MixturePredictor(args.package_dir).predict(payload, args.endpoint, args.pooling)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
