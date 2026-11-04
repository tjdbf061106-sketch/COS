"""Conservative identity and joint-composition checks, without substituting unknown stocks."""
import re
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from rdkit import Chem


def normalized(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def known_text(value):
    return isinstance(value, str) and value.strip().casefold() not in {"", "unknown", "not_reported", "none", "nan", "미확인"}


def canonical(value):
    if pd.isna(value) or not str(value).strip(): return None
    mol = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(mol) if mol is not None else None


def resolve_verified_formula(rows, registry, features, evidence_records, context="external"):
    if context not in {"external", "study_assumption"}: raise ValueError("Unknown identity context")
    if not {"ingredient", "wt_pct"}.issubset(rows): raise ValueError("ingredient and wt_pct columns required")
    vector = pd.Series(0.0, index=features)
    details = []
    by_name = {normalized(v): v for v in registry.ingredient_id}
    records = {r["evidence_id"]: r for r in evidence_records}
    if len(records) != len(evidence_records): raise ValueError("Duplicate evidence_id")
    for number, (_, row) in enumerate(rows.iterrows(), 1):
        raw_name = row.get("ingredient", "")
        name = "" if pd.isna(raw_name) else str(raw_name).strip()
        try: amount = float(row.get("wt_pct"))
        except (TypeError, ValueError): amount = np.nan
        if not name and pd.isna(amount): continue
        ingredient = by_name.get(normalized(name))
        status, message = "verified", "원료·공급사·등급 확인 기록과 일치"
        if not np.isfinite(amount) or amount < 0 or amount > 100:
            status, message = "invalid_amount", "투입량은 유한한 0~100 질량%여야 합니다."
        elif ingredient is None:
            status, message = "unverified_identity", "등록 상용 원료 ID가 필요합니다. INCI·대표 구조로 자동 대체하지 않습니다."
        else:
            reference = registry[registry.ingredient_id == ingredient].iloc[0]
            supplied_smiles = row.get("smiles", "")
            if pd.notna(supplied_smiles) and str(supplied_smiles).strip():
                a, b = canonical(supplied_smiles), canonical(reference.representative_smiles)
                if a is None or b is None or a != b:
                    status, message = "structure_conflict", "입력 구조를 등록 대표 구조와 확인할 수 없습니다."
            evidence_id = str(row.get("evidence_id", ""))
            record = records.get(evidence_id)
            allowed_status = "documented_source_assumption" if context == "study_assumption" else "verified_external_grade"
            if status == "verified" and (record is None or record.get("status") != allowed_status or not known_text(record.get("evidence_reference"))):
                status, message = "missing_identity_evidence", "공급사·등급 확인 근거가 없어 예측을 보류합니다."
            if status == "verified":
                matches = record.get("ingredient_id") == ingredient and all(
                    normalized(row.get(field, "")) == normalized(record.get(field, "")) and known_text(record.get(field))
                    for field in ["supplier", "grade"])
                if context == "external":
                    matches &= known_text(row.get("lot_id")) and known_text(record.get("lot_id")) and row.get("lot_id") == record.get("lot_id")
                if not matches:
                    status, message = "identity_mismatch", "원료·공급사·등급·로트가 확인 기록과 일치하지 않습니다."
        if status == "verified": vector[ingredient] += float(amount)
        details.append({"row": number, "input_name": name, "ingredient_id": ingredient,
                        "wt_pct": float(amount) if np.isfinite(amount) else None, "status": status, "reason": message})
    detail = pd.DataFrame(details)
    if detail.empty: return None, detail, ["처방 입력이 비어 있습니다."]
    blockers = detail.loc[detail.status != "verified", "reason"].tolist()
    if blockers: return None, detail, blockers
    if vector.sum() <= 0 or vector.sum() > 100 + 1e-8:
        return None, detail, ["전체 원료 투입량 합계가 유효하지 않습니다."]
    return vector.to_frame().T, detail, []


def joint_domain(formula, training, features, tolerance=1e-8):
    if len(formula) != 1 or list(formula.columns) != list(features):
        raise ValueError("One recipe in the exact model feature order is required")
    target = formula.to_numpy(float)[0]
    if not np.isfinite(target).all() or (target < 0).any() or target.sum() <= 0 or target.sum() > 100 + tolerance:
        return {"status": "invalid", "allow_research_prediction": False, "reason": "유효하지 않은 투입량"}
    x = training[features].to_numpy(float)
    same_combination = ((x > 0) == (target > 0)).all(axis=1)
    reference = training.loc[same_combination]
    common = {"support_recipes": len(reference), "geometry_is_validated_error_bound": False,
              "stock_sum_pct": float(target.sum()), "added_water_balance_estimate_pct": float(100-target.sum())}
    if reference.empty:
        return {**common, "status": "unseen_combination", "allow_research_prediction": False,
                "reason": "동일 원료 조합의 관측 처방이 없어 예측을 보류합니다."}
    values = reference[features].to_numpy(float)
    exact = np.max(np.abs(values-target), axis=1) <= tolerance
    if exact.any():
        ids = reference.loc[exact, "sample_id"].astype(int).tolist()
        return {**common, "status": "observed_recipe", "allow_research_prediction": False,
                "observed_sample_ids": ids, "reason": "이미 측정된 처방입니다. 예측 대신 실측 결과를 표시합니다."}
    unique, selected = np.unique(values, axis=0, return_index=True)
    if len(unique) < 2:
        return {**common, "status": "insufficient_joint_support", "allow_research_prediction": False,
                "reason": "동일 조합의 서로 다른 관측점이 한 개뿐이므로 농도 변경을 보간으로 인정할 수 없습니다."}
    if (target < unique.min(axis=0)-tolerance).any() or (target > unique.max(axis=0)+tolerance).any():
        return {**common, "status": "outside_joint_ranges", "allow_research_prediction": False,
                "reason": "동일 조합에서 관측한 투입량 범위를 벗어납니다."}
    active = target > 0
    scale = np.maximum(np.ptp(unique[:, active], axis=0), 1.0)
    z = unique[:, active] / scale
    query = target[active] / scale
    solution = linprog(np.zeros(len(unique)), A_eq=np.vstack([z.T, np.ones(len(unique))]),
                       b_eq=np.r_[query, 1], bounds=(0, None), method="highs")
    if not solution.success or np.max(abs(solution.x @ unique-target)) > tolerance or abs(solution.x.sum()-1) > tolerance:
        return {**common, "status": "outside_observed_convex_hull", "allow_research_prediction": False,
                "reason": "개별 범위 안이어도 관측 처방들의 공동 조성 보간 영역 밖입니다."}
    supports = [{"sample_id": int(reference.iloc[selected[i]].sample_id), "weight": float(w)}
                for i, w in enumerate(solution.x) if w > tolerance]
    return {**common, "status": "within_observed_convex_hull", "allow_research_prediction": True,
            "convex_support": supports, "reason": "관측 처방의 볼록결합으로 표현됩니다. 기하학적 지지이며 실제 오차·안정성을 보장하지 않습니다."}


def verify_conditions(conditions, context="external"):
    missing = []
    expected = {"temperature_C": 25.0, "observation_hours": 36.0, "shear_rate_s_inv": 100.0}
    for key, value in expected.items():
        try:
            if not np.isfinite(float(conditions[key])) or abs(float(conditions[key])-value) > 1e-8: missing.append(key)
        except (KeyError, TypeError, ValueError): missing.append(key)
    try:
        if not 5.6 <= float(conditions["pH"]) <= 6.0: missing.append("pH")
    except (KeyError, TypeError, ValueError): missing.append("pH")
    if context == "external" and (conditions.get("process_matches_source") is not True or not conditions.get("process_evidence_reference")):
        missing.append("manufacturing_process_evidence")
    return missing


def guarded_prediction(formula, identity_blockers, domain, conditions, models, features, context="external", regression_domains=None):
    if identity_blockers: return {"status": "HOLD", "reasons": identity_blockers, "predictions": None}
    condition_issues = verify_conditions(conditions, context)
    if condition_issues: return {"status": "HOLD", "reasons": ["시험·제조 조건 미확인 또는 불일치: " + ", ".join(condition_issues)], "predictions": None}
    if not domain.get("allow_research_prediction"):
        return {"status": "OBSERVED" if domain.get("status") == "observed_recipe" else "HOLD",
                "reasons": [domain["reason"]], "predictions": None}
    x = formula[features]
    p = float(models["stability"].predict_proba(x)[0, 1])
    values = {"uncalibrated_stability_score": p, "stability_threshold": 0.5,
              "predicted_stable": p >= 0.5, "viscosity_100_mPa_s": None, "turbidity_ntu": None}
    property_holds = []
    regression_domains = regression_domains or {}
    if p >= 0.5:
        for target, output in [("viscosity_100", "viscosity_100_mPa_s"), ("turbidity", "turbidity_ntu")]:
            if not regression_domains.get(target, {}).get("allow_research_prediction"):
                property_holds.append(target + ": 실제 안정·해당 물성 측정 처방의 공동 조성 지지가 부족해 수치를 보류합니다.")
                continue
            prediction = float(models[target].predict(x)[0])
            values[output] = float(10 ** prediction) if target == "viscosity_100" else float(10 ** max(prediction, 0) - 1)
    return {"status": "RESEARCH_ONLY", "predictions": values, "real_stability_confirmed": False,
            "regression_domains": regression_domains,
            "reasons": ["동일 상용 원료·시험 조건을 가정한 연구용 후보입니다.",
                        "점도·탁도는 안정성 점수가 0.5 이상일 때만 표시하며 실제 안정성이 확인된 값은 아닙니다.",
                        "기하학적 보간과 높은 모델 점수는 검증된 오차범위나 원료 편입 승인을 뜻하지 않습니다."] + property_holds}
