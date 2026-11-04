"""Read-only audit for the proposed ingredient onboarding protocol. No new models are fitted."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os
import sys
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
DOC = Path(os.environ.get("SHAMPOO_PROTOCOL_DOCX", str(OUT / "original_protocol.docx")))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from app_core import applicability, build_resolver, project_inputs, resolve_ingredient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_inventory():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob("*"))
            if p.is_file() and not p.is_relative_to(OUT) and "__pycache__" not in p.parts}


def run():
    before = source_inventory()
    document_hash = sha(DOC)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(DOC) as z:
        doc = ET.fromstring(z.read("word/document.xml"))
        paragraphs = ["".join(t.text or "" for t in p.findall(".//w:t", ns))
                      for p in doc.findall(".//w:p", ns)]
        media = {i.filename: hashlib.sha256(z.read(i.filename)).hexdigest()
                 for i in z.infolist() if i.filename.startswith("word/media/")}
    (OUT / "original_protocol_text.txt").write_text("\n".join(p for p in paragraphs if p) + "\n", encoding="utf-8")
    data, features, registry = project_inputs()
    counts = data.groupby("ingredient_combination").size()
    assert len(data) == 812 and data.sample_id.is_unique and len(features) == 18
    records = []
    for ingredient in features:
        present = data[data[ingredient] > 0]
        absent = data[data[ingredient] == 0]
        records.append({"ingredient": ingredient, "present_n": len(present),
                        "stable_n": int(present.stable.sum()), "unstable_n": int((present.stable == 0).sum()),
                        "viscosity_n": int(present.viscosity_100_mPa_s.notna().sum()),
                        "turbidity_n": int(present.turbidity_ntu.notna().sum()),
                        "combination_groups": present.ingredient_combination.nunique(),
                        "absent_n": len(absent), "absent_stable_n": int(absent.stable.sum()),
                        "absent_unstable_n": int((absent.stable == 0).sum()),
                        "positive_stock_min_pct": present[ingredient].min(),
                        "positive_stock_max_pct": present[ingredient].max()})
    support = pd.DataFrame(records)
    support.to_csv(OUT / "ingredient_support.csv", index=False, encoding="utf-8-sig")
    assert (support.present_n + support.absent_n).eq(812).all()
    assert support.present_n.eq(support.stable_n + support.unstable_n).all()
    oof = pd.read_csv(ROOT / "results/oof_predictions.csv", float_precision="round_trip")
    rows = []
    for (scheme, seed, model), group in oof[oof.target == "stability"].groupby(["scheme", "seed", "model"]):
        assert len(group) == 812 and group.sample_id.is_unique
        y = group.observed.to_numpy(int)
        pred = group.predicted.to_numpy() >= 0.5
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({"scheme": scheme, "seed": int(seed), "model": model, "positive_class": "stable",
                     "threshold": 0.5, "n": 812, "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
                     "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
                     "MCC": matthews_corrcoef(y, pred), "ROC-AUC": roc_auc_score(y, group.predicted)})
    repeat = pd.DataFrame(rows)
    prior = pd.read_csv(ROOT / "results/repeat_metrics.csv", float_precision="round_trip").query("target == 'stability'")
    aligned = repeat.merge(prior, on=["scheme", "seed", "model"], validate="one_to_one")
    for new, old in zip(["Sensitivity", "Specificity", "MCC", "ROC-AUC"], ["recall", "specificity", "mcc", "roc_auc"]):
        np.testing.assert_allclose(aligned[new], aligned[old], atol=1e-12, rtol=0)
    repeat.to_csv(OUT / "classification_metrics_by_repeat.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for (scheme, model), group in repeat.groupby(["scheme", "model"]):
        row = {"scheme": scheme, "model": model, "positive_class": "stable", "threshold": 0.5,
               "n_independent_recipes": 812, "n_repeats": len(group)}
        for metric in ["Sensitivity", "Specificity", "MCC", "ROC-AUC"]:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_sd"] = group[metric].std(ddof=1) if len(group) > 1 else np.nan
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "classification_metrics_summary.csv", index=False, encoding="utf-8-sig")
    resolver = build_resolver(registry)
    ingredient = registry[registry.ingredient_id == "Plantacare 2000"].iloc[0]
    resolution = resolve_ingredient("NEW_SUPPLIER_UNVERIFIED", ingredient.representative_smiles, resolver, registry)
    assert resolution.status == "resolved" and resolution.ingredient_id == "Plantacare 2000"
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'blockers = details.loc[~details.status.eq("resolved")]' in app_text
    observed = data[data.sample_id == 38].iloc[0]
    x = observed[features].astype(float).to_frame().T
    changed = x.copy()
    changed.loc[:, "Texapon SB 3 KC"] = float(data["Texapon SB 3 KC"].max())
    domain = applicability(changed, data, features)
    assert counts[observed.ingredient_combination] == 1
    assert domain["status"] == "interpolation" and not np.allclose(x, changed)
    case = {"case": "Global individual ranges do not establish joint interpolation",
            "base_sample_id": 38, "same_combination_observations": 1,
            "changed_ingredient": "Texapon SB 3 KC", "observed_pct": float(x["Texapon SB 3 KC"].iloc[0]),
            "new_pct": float(changed["Texapon SB 3 KC"].iloc[0]), "app_result": domain,
            "measured_outcome_for_changed_recipe": None,
            "interpretation": "Outside the convex hull of the one observed recipe in this exact combination; no error or outcome inferred."}
    folds = pd.read_csv(ROOT / "results/outer_folds.csv")
    coverage = []
    for (scheme, seed, fold), test in folds[folds.scheme == "combination_holdout"].groupby(["scheme", "seed", "fold"]):
        train_ids = set(data.sample_id) - set(test.sample_id)
        train = data[data.sample_id.isin(train_ids)]
        held = data[data.sample_id.isin(test.sample_id)]
        supported = held.ingredient_combination.isin(set(train.ingredient_combination))
        assert not supported.any()
        coverage.append({"scheme": scheme, "seed": int(seed), "fold": int(fold), "n_test": len(held),
                         "seen_combination_n": int(supported.sum()),
                         "coverage_if_exact_seen_combination_required": float(supported.mean())})
    pd.DataFrame(coverage).to_csv(OUT / "hypothetical_gate_coverage.csv", index=False, encoding="utf-8-sig")
    counterexamples = {"unverified_supplier_with_known_structure": asdict(resolution),
                      "global_range_false_interpolation": case,
                      "scope": "Core functions reproduced read-only. No UI change, new model fit or new-formulation performance estimate.",
                      "hypothetical_gate_note": "Current app still displays predictions outside combinations; zero coverage describes an exact-seen-combination gate applied within the saved grouped folds, not current app rejection frequency."}
    save("app_counterexamples.json", counterexamples)
    for path, expected in json.loads((ROOT / "training_lock.json").read_text())["hashes"].items():
        assert sha(ROOT / path) == expected, path
    assert source_inventory() == before and sha(DOC) == document_hash
    save("source_inventory.json", {"project_root": str(ROOT), "original_protocol": str(DOC),
                                   "original_protocol_sha256": document_hash, "project_files": before,
                                   "document_media_sha256": media, "source_files_unchanged": True})
    result = {"status": "PASS", "analysis": "Feasibility and implementation audit; no new performance experiment",
              "rows": len(data), "stable": int(data.stable.sum()), "unstable": int((data.stable == 0).sum()),
              "viscosity_n": int(data.viscosity_100_mPa_s.notna().sum()),
              "turbidity_n": int(data.turbidity_ntu.notna().sum()), "ingredients": len(features),
              "exact_combination_groups": len(counts), "singleton_groups": int((counts == 1).sum()),
              "singleton_group_fraction": float((counts == 1).mean()),
              "ingredient_heldout_support_min_n": int(support.present_n.min()),
              "ingredient_heldout_support_max_n": int(support.present_n.max()),
              "ingredient_stable_min_n": int(support.stable_n.min()),
              "ingredient_stable_max_n": int(support.stable_n.max()),
              "existing_oof_metrics_reproduced": True, "training_lock_matches": True,
              "project_files_unchanged": True, "original_docx_unchanged": True,
              "new_models_trained": False, "external_validation_performed": False,
              "real_new_ingredient_experiment_performed": False,
              "paper_url": "https://pubmed.ncbi.nlm.nih.gov/38961122/"}
    save("audit_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
