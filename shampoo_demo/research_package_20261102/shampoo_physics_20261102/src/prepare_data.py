"""Download the published CC0 records, preserve stock percentages, audit source domains."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from common import REPO, ROOT, ingredient_group, interpolate_viscosity, read_json, sha256, write_csv, write_json

RDLogger.DisableLog("rdApp.*")
FILES = {
    25451878: (45187180, "LiquidFormulationsDataset_2023.json"),
    25451926: (45187306, "surfactants_information.csv"),
    25451929: (45187309, "ingredient_structures.pptx"),
}
META = {"Stability_Test", "Turbidity_NTU", "Turbidity_Error", "Viscosity", "Rheology_Type", "Rheology_Data", "ID"}


def download_sources():
    records = []
    for article, (file_id, name) in FILES.items():
        meta_path = ROOT / "sources" / f"article_{article}_v1.json"
        if meta_path.exists():
            metadata = read_json(meta_path)
        else:
            response = requests.get(f"https://api.figshare.com/v2/articles/{article}/versions/1", timeout=60)
            response.raise_for_status()
            metadata = response.json()
            write_json(meta_path, metadata)
        entry = next(f for f in metadata["files"] if f["id"] == file_id)
        path = ROOT / "sources" / name
        if not path.exists():
            response = requests.get(entry["download_url"], timeout=90)
            response.raise_for_status()
            path.write_bytes(response.content)
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        expected = entry.get("computed_md5") or entry.get("supplied_md5")
        if expected and expected.lower() != md5:
            raise ValueError(f"Published checksum mismatch: {name}")
        records.append({"file": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "md5": md5,
                        "published_md5": expected, "article_id": article, "version": 1, "url": metadata["url_public_html"],
                        "license": metadata["license"]["name"]})
    write_json(ROOT / "sources/manifest.json", {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "records": records})


def trade_key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def bounds(value):
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    return (float(numbers[0]), float(numbers[-1])) if numbers else (np.nan, np.nan)


def make_registry(ingredients):
    surf = pd.read_csv(ROOT / "sources/surfactants_information.csv")
    lookup = {trade_key(row.trade_name): row for row in surf.itertuples(index=False)}
    rows = []
    polymers = {
        "Luviquat Excellence": ("Polyquaternium-16", "approximately 40000 g/mol; distribution and charge density not fully specified"),
        "Dehyquart CC6": ("Polyquaternium-6", "molecular weight unknown"),
        "Dehyquart CC7 Benz": ("Polyquaternium-7", "approximately 500000 g/mol; source slide gives copolymer fractions"),
        "Salcare Super 7": ("Polyquaternium-7", "100000-200000 g/mol; different copolymer fractions from CC7"),
    }
    for name in ingredients:
        row = {"ingredient_id": name, "representative_smiles": "", "molecular_composition_complete": False,
               "active_min_pct": np.nan, "active_max_pct": np.nan, "water_min_pct": np.nan, "water_max_pct": np.nan,
               "source_note": "Commercial stock ID; preserve grade, not only INCI"}
        if trade_key(name) in lookup:
            # Column names containing punctuation are easier to address by position.
            source = surf[surf.trade_name.map(trade_key) == trade_key(name)].iloc[0]
            row.update(role="surfactant", inci=source["INCI"], representative_smiles=source["SMILES_C0"], surfactant_type=source["surf_type"])
            row["active_min_pct"], row["active_max_pct"] = bounds(source["active_matter (%)"])
            row["water_min_pct"], row["water_max_pct"] = bounds(source["water_content (%)"])
            row["other_content"] = source["other_content (%)"]
            row["source_note"] = "Representative chain only; commercial chain-length distribution retained in source CSV"
            if name == "Dehyquart A-CA":
                row["source_note"] += "; published INCI contains 'Potassium' but SMILES has chloride and no potassium: identity review required"
            if name == "Dehyton AB 30":
                row["source_note"] += "; published representative SMILES includes a net +1 charge: do not silently correct"
        elif name in polymers:
            row.update(role="conditioning_polymer", inci=polymers[name][0], source_note=polymers[name][1])
        elif name == "Arlypon F":
            row.update(role="thickener", inci="Laureth-2", representative_smiles="CCCCCCCCCCCCCCOCCOCCO", source_note="Representative SMILES transcribed from source slide 26; not a full commercial distribution")
        elif name == "Arlypon TT":
            row.update(role="thickener", inci="PEG/PPG-120/10 Trimethylolpropane Trioleate and Laureth-2", source_note="Source slide 25: commercial blend, each component 40-50%; polymer distribution unresolved")
        else:
            raise ValueError(f"Unmapped ingredient: {name}")
        mol = Chem.MolFromSmiles(row["representative_smiles"]) if row["representative_smiles"] else None
        row["representative_smiles_valid"] = mol is not None
        row["representative_formal_charge"] = Chem.GetFormalCharge(mol) if mol is not None else np.nan
        rows.append(row)
    registry = pd.DataFrame(rows)
    write_csv(ROOT / "data/ingredient_registry.csv", registry)
    return registry


def prepare_formulations():
    raw = read_json(ROOT / "sources/LiquidFormulationsDataset_2023.json")
    frame = pd.DataFrame(raw)
    ingredients = [c for c in frame if c not in META]
    if len(ingredients) != 18 or len(frame) != 812 or frame.ID.nunique() != 812:
        raise ValueError("Published dataset schema/count changed; review before proceeding")
    if not frame.Stability_Test.map(lambda x: isinstance(x, bool)).all():
        raise ValueError("Stability labels must be explicit booleans")
    registry = make_registry(ingredients)
    amounts = frame[ingredients].astype(float)
    if not np.isfinite(amounts.to_numpy()).all() or (amounts < 0).any().any() or (amounts.sum(axis=1) > 100).any():
        raise ValueError("Invalid stock percentages")
    polymers = registry.loc[registry.role == "conditioning_polymer", "ingredient_id"].tolist()
    thickeners = registry.loc[registry.role == "thickener", "ingredient_id"].tolist()
    data = amounts.copy()
    data.insert(0, "sample_id", frame.ID.astype(int))
    data["stable"] = frame.Stability_Test.astype(int)
    data["ingredient_combination"] = amounts.apply(lambda r: ingredient_group(r, ingredients), axis=1)
    data["polymer_thickener_group"] = amounts.apply(lambda r: ingredient_group(r, polymers) + " / " + ingredient_group(r, thickeners), axis=1)
    data["stock_sum_pct"] = amounts.sum(axis=1)
    data["added_water_balance_pct_estimate"] = 100 - data.stock_sum_pct
    data["water_basis_note"] = "Pre-titration balance estimate; excludes water inside commercial stocks; acid/base additions not in JSON"
    data["turbidity_ntu"] = pd.to_numeric(frame.Turbidity_NTU, errors="coerce")
    data["turbidity_error_as_reported"] = pd.to_numeric(frame.Turbidity_Error, errors="coerce")
    data["viscosity_class"] = frame.Viscosity.replace("NA", np.nan)
    data["rheology_type"] = frame.Rheology_Type.replace("NA", np.nan)
    data["measurement_temperature_C"] = 25
    data["target_pH"] = 5.8
    data["target_pH_tolerance"] = 0.2
    data["rheometer"] = np.where(frame.ID.between(619, 689), "Anton Paar MCR702e", "TA DHR30")
    eta, flow_rows, curve_issues = [], [], []
    for row in frame.itertuples(index=False):
        if not row.Rheology_Data:
            eta.append(np.nan)
            continue
        curve = {key: value for part in row.Rheology_Data for key, value in part.items()}
        try:
            eta.append(interpolate_viscosity(curve["shear_rate"], curve["avg_viscosity"]))
        except ValueError as exc:
            eta.append(np.nan)
            curve_issues.append({"sample_id": row.ID, "reason": str(exc)})
        sd = curve.get("std_dev", [np.nan] * len(curve["shear_rate"]))
        if not len(curve["shear_rate"]) == len(curve["avg_viscosity"]) == len(sd):
            raise ValueError(f"Mismatched curve arrays: {row.ID}")
        for x, y, s in zip(curve["shear_rate"], curve["avg_viscosity"], sd):
            flow_rows.append({"sample_id": row.ID, "shear_rate_s_inv": x, "viscosity_mPa_s": y, "std_dev_mPa_s": s})
    data["viscosity_100_mPa_s"] = eta
    data["log10_viscosity_100"] = np.log10(data.viscosity_100_mPa_s)
    data["log10p1_turbidity"] = np.log10(1 + data.turbidity_ntu)
    if data.loc[data.stable == 0, ["turbidity_ntu", "viscosity_100_mPa_s"]].notna().any().any():
        raise ValueError("Unexpected measurements on unstable samples")
    write_csv(ROOT / "data/formulations.csv", data)
    write_csv(ROOT / "data/flow_curves.csv", pd.DataFrame(flow_rows))
    write_json(ROOT / "data/feature_names.json", ingredients)
    write_json(ROOT / "data/qc.json", {
        "rows": len(data), "stable": int(data.stable.sum()), "unstable": int((1-data.stable).sum()),
        "ingredient_count": len(ingredients), "combination_groups": data.ingredient_combination.nunique(),
        "subsystem_groups": data.polymer_thickener_group.nunique(),
        "nonzero_ingredient_counts": (amounts > 0).sum(axis=1).value_counts().to_dict(),
        "viscosity_100_targets": data.viscosity_100_mPa_s.notna().sum(), "turbidity_targets": data.turbidity_ntu.notna().sum(),
        "curve_issues": curve_issues, "non_four_ingredient_ids": data.loc[(amounts > 0).sum(axis=1) != 4, "sample_id"].tolist(),
        "duplicate_exact_recipes": int(amounts.duplicated().sum()),
        "physical_conditions": "35 min mixing at 360 rpm and 25 C; pH 5.8 +/- 0.2; final stability at 36 h; rheology 25 C",
        "not_present": ["skin irritation", "eye irritation", "long-term stability", "exact acid/base titrant mass", "complete molecular composition"],
    })
    return data, registry


def molecular_references():
    skin = pd.read_csv(REPO / "skin irritation/skin_vectorized.csv")
    eye = pd.read_csv(REPO / "eye irritation/eye_vectorized.csv", low_memory=False)
    raw_skin = pd.read_csv(REPO / "skin irritation/skin_filtered.csv")
    cache = pd.read_csv(REPO / "skin irritation/casrn_smiles_cache.csv")
    names = raw_skin.dropna(subset=["casrn"]).drop_duplicates("casrn").set_index("casrn").chemical_name.to_dict()
    skin_ref = cache[cache.casrn.isin(names)].copy()
    skin_ref["name"] = skin_ref.casrn.map(names)
    raw_eye = pd.read_csv(REPO / "eye irritation/eye_filtered.csv")
    eye_train = eye[(eye.substance_type == "Mixture") & eye.binary_label.notna()]
    eye_ref = raw_eye[(raw_eye.substance_type == "Mixture") & raw_eye.formulation_id.isin(eye_train.formulation_id)].copy()
    eye_ref = eye_ref.rename(columns={"chemical_name": "name"})
    summary = {
        "skin": {"rows": len(skin), "label_counts": skin.binary_label.value_counts().to_dict(),
                 "undeclared_over_1pct": (skin.undeclared_pct > 1).sum(), "median_undeclared_pct": skin.undeclared_pct.median(),
                 "label_meaning": "1=EPA I/II, 0=EPA III/IV; 0 does not mean absence of skin irritation"},
        "eye": {"all_rows": len(eye), "mixture_labeled_rows": len(eye_train), "label_counts": eye_train.binary_label.value_counts().to_dict(),
                "label_meaning": "1=EPA I/II, 0=EPA III/IV; outcome distinct from skin and from physical properties"},
        "source_commit": read_json(ROOT / "protocol.json")["source_commit"],
        "use": "Read-only reference ingredient audit, not physical-property training or toxicity prediction",
        "limitations": "Only reference SMILES actually present in these inputs are compared; unresolved source structures are not imputed. Molecular similarity is not a validated AD threshold.",
    }
    source_paths = ["skin irritation/skin_vectorized.csv", "skin irritation/skin_filtered.csv", "skin irritation/casrn_smiles_cache.csv",
                    "eye irritation/eye_vectorized.csv", "eye irritation/eye_filtered.csv"]
    summary["input_hashes"] = {p: sha256(REPO / p) for p in source_paths}
    return {"skin": skin_ref, "eye": eye_ref}, summary


def audit_domains(data, registry):
    references, summary = molecular_references()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    audit = []
    for endpoint, refs in references.items():
        molecules, seen = [], set()
        for row in refs.dropna(subset=["smiles"]).itertuples(index=False):
            mol = Chem.MolFromSmiles(str(row.smiles))
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol)
            if canonical in seen:
                continue
            seen.add(canonical)
            parent = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
            molecules.append((canonical, generator.GetFingerprint(parent), str(row.casrn), str(row.name)))
        summary[endpoint]["reference_rows"] = len(refs)
        summary[endpoint]["reference_rows_with_smiles"] = int(refs.smiles.notna().sum())
        summary[endpoint]["unique_valid_reference_structures"] = len(molecules)
        fps = [r[1] for r in molecules]
        for row in registry.itertuples(index=False):
            entry = {"endpoint": endpoint, "ingredient_id": row.ingredient_id, "representative_only": True,
                     "reference_structure_count": len(molecules), "exact_full_smiles_match": False,
                     "nearest_parent_tanimoto": np.nan, "nearest_casrn": "", "nearest_name": "",
                     "ad_status": "not_validated", "identity_review_note": row.source_note}
            mol = Chem.MolFromSmiles(row.representative_smiles) if row.representative_smiles else None
            if mol is not None and fps:
                canonical = Chem.MolToSmiles(mol)
                parent = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
                sims = DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(parent), fps)
                ix = int(np.argmax(sims))
                entry.update(exact_full_smiles_match=canonical in seen, nearest_parent_tanimoto=sims[ix],
                             nearest_casrn=molecules[ix][2], nearest_name=molecules[ix][3])
            audit.append(entry)
    write_csv(ROOT / "data/skin_eye_ingredient_audit.csv", pd.DataFrame(audit))
    write_json(ROOT / "data/skin_eye_source_audit.json", summary)
    links = data[["sample_id"]].copy()
    represented = registry.loc[registry.representative_smiles_valid, "ingredient_id"].tolist()
    links["representative_structure_stock_pct"] = data[represented].sum(axis=1)
    links["unrepresented_structure_stock_pct"] = data.stock_sum_pct - links.representative_structure_stock_pct
    links["complete_molecular_composition"] = False
    links["skin_measured_label_available"] = False
    links["eye_measured_label_available"] = False
    links["skin_prediction_status"] = "not_validated_for_shampoo"
    links["eye_prediction_status"] = "not_validated_for_shampoo"
    links["toxicity_transfer_status"] = "not_performed"
    write_csv(ROOT / "data/endpoint_links.csv", links)


if __name__ == "__main__":
    download_sources()
    data, registry = prepare_formulations()
    audit_domains(data, registry)
    print(read_json(ROOT / "data/qc.json"))
