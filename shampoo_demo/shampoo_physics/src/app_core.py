"""Core input resolution, applicability checks, and predictions for the shampoo app."""
from __future__ import annotations

import re
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from common import ROOT, load_inputs


@dataclass(frozen=True)
class Resolution:
    ingredient_id: str | None
    status: str
    basis: str | None
    canonical_smiles: str | None
    message: str


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def canonicalize_smiles(value: object) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    return Chem.MolToSmiles(mol) if mol is not None else None


def build_resolver(registry: pd.DataFrame) -> dict:
    by_id: dict[str, list[str]] = {}
    by_inci: dict[str, list[str]] = {}
    by_smiles: dict[str, list[str]] = {}
    for row in registry.itertuples(index=False):
        by_id.setdefault(normalize_name(row.ingredient_id), []).append(row.ingredient_id)
        by_inci.setdefault(normalize_name(row.inci), []).append(row.ingredient_id)
        canonical = canonicalize_smiles(row.representative_smiles) if pd.notna(row.representative_smiles) else None
        if canonical:
            by_smiles.setdefault(canonical, []).append(row.ingredient_id)
    return {"by_id": by_id, "by_inci": by_inci, "by_smiles": by_smiles}


def resolve_ingredient(name: object, smiles: object, resolver: dict, registry: pd.DataFrame) -> Resolution:
    name_text = "" if pd.isna(name) else str(name).strip()
    smiles_text = "" if pd.isna(smiles) else str(smiles).strip()
    canonical = canonicalize_smiles(smiles_text) if smiles_text else None
    if smiles_text and canonical is None:
        return Resolution(None, "invalid", None, None, "SMILES 형식이 올바르지 않습니다.")

    matches: list[str] = []
    basis = None
    if name_text:
        key = normalize_name(name_text)
        matches = resolver["by_id"].get(key, [])
        basis = "commercial_ingredient_name"
        if not matches:
            matches = resolver["by_inci"].get(key, [])
            basis = "inci"
        if len(matches) > 1:
            return Resolution(None, "ambiguous", basis, canonical, "같은 INCI의 상업 원료가 여러 개이므로 제품명을 선택해야 합니다.")

    if not matches and canonical:
        matches = resolver["by_smiles"].get(canonical, [])
        basis = "representative_smiles"
        if len(matches) > 1:
            return Resolution(None, "ambiguous", basis, canonical, "같은 대표 구조에 여러 상업 원료가 연결되어 제품명이 필요합니다.")

    if not matches:
        if name_text or canonical:
            return Resolution(None, "unsupported", None, canonical, "등록되지 않은 원료입니다. 참고 구조 검색만 가능하고 물성 예측은 중단됩니다.")
        return Resolution(None, "empty", None, None, "빈 행")

    ingredient_id = matches[0]
    record = registry.loc[registry.ingredient_id.eq(ingredient_id)].iloc[0]
    registered = canonicalize_smiles(record.representative_smiles) if pd.notna(record.representative_smiles) else None
    if canonical and registered and canonical != registered:
        return Resolution(None, "conflict", basis, canonical, "원료명과 입력 SMILES가 등록된 대표 구조와 일치하지 않습니다.")
    note = "등록 상업 원료명으로 확인됨" if basis == "commercial_ingredient_name" else "대표 정보로 연결됨; 상업 등급 동일성 확인 필요"
    return Resolution(ingredient_id, "resolved", basis, canonical or registered, note)


def resolve_formula(rows: pd.DataFrame, registry: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"ingredient", "smiles", "wt_pct"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Input rows require columns: {sorted(required)}")
    resolver = build_resolver(registry)
    vector = pd.Series(0.0, index=features, dtype=float)
    details = []
    for index, row in rows.iterrows():
        name = row.ingredient
        smiles = row.smiles
        amount = pd.to_numeric(row.wt_pct, errors="coerce")
        if (pd.isna(name) or not str(name).strip()) and (pd.isna(smiles) or not str(smiles).strip()) and pd.isna(amount):
            continue
        resolution = resolve_ingredient(name, smiles, resolver, registry)
        amount_valid = bool(pd.notna(amount) and np.isfinite(amount) and 0 <= float(amount) <= 100)
        if resolution.status == "resolved" and amount_valid:
            vector[resolution.ingredient_id] += float(amount)
        details.append({
            "row": int(index) + 1,
            "input_name": "" if pd.isna(name) else str(name),
            "resolved_ingredient": resolution.ingredient_id or "",
            "status": resolution.status if amount_valid else "invalid_amount",
            "mapping_basis": resolution.basis or "",
            "wt_pct": amount,
            "canonical_smiles": resolution.canonical_smiles or "",
            "message": resolution.message if amount_valid else "투입량은 0~100 사이의 숫자여야 합니다.",
        })
    return vector.to_frame().T, pd.DataFrame(details)


def applicability(formula: pd.DataFrame, training: pd.DataFrame, features: list[str]) -> dict:
    values = formula.iloc[0]
    total = float(values.sum())
    active = tuple(sorted(values.index[values.gt(0)]))
    observed_sets = {
        tuple(sorted(row.index[row.gt(0)])) for _, row in training[features].iterrows()
    }
    outside = []
    for feature in active:
        observed = training.loc[training[feature].gt(0), feature]
        if observed.empty or values[feature] < observed.min() or values[feature] > observed.max():
            outside.append(feature)
    if total <= 0:
        status = "invalid"
        message = "한 개 이상의 원료 투입량이 필요합니다."
    elif total > 100:
        status = "invalid"
        message = "등록 원료 투입량 합계가 100%를 초과합니다."
    elif active not in observed_sets:
        status = "outside_combination"
        message = "학습자료에 없는 원료 조합입니다. 예측값은 외삽 경고와 함께 참고만 가능합니다."
    elif outside:
        status = "outside_range"
        message = "일부 투입량이 해당 원료의 관측 범위를 벗어납니다."
    else:
        status = "interpolation"
        message = "학습자료에서 관측된 원료 조합과 개별 투입량 범위 안입니다."
    return {
        "status": status,
        "message": message,
        "stock_sum_pct": total,
        "water_balance_pct_estimate": 100 - total,
        "active_ingredients": list(active),
        "outside_range_ingredients": outside,
    }


def load_models() -> dict:
    return {target: joblib.load(ROOT / "models" / f"{target}.joblib") for target in ["stability", "viscosity_100", "turbidity"]}


def predict_properties(formula: pd.DataFrame, models: dict, features: list[str]) -> dict:
    x = formula[features]
    stability = float(models["stability"].predict_proba(x)[0, 1])
    viscosity_log = float(models["viscosity_100"].predict(x)[0])
    turbidity_log = max(float(models["turbidity"].predict(x)[0]), 0.0)
    return {
        "stability_score_uncalibrated": stability,
        "viscosity_100_mPa_s": 10 ** viscosity_log,
        "turbidity_ntu": 10 ** turbidity_log - 1,
    }


def load_reference_fingerprints() -> dict:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = {}
    for endpoint in ["skin", "eye"]:
        frame = pd.read_csv(ROOT / f"sources/skin_eye_chemical_{endpoint}.csv")
        frame["canonical_smiles"] = frame.SMILES.map(canonicalize_smiles)
        frame = frame.dropna(subset=["canonical_smiles"])
        grouped = frame.groupby("canonical_smiles")["class"]
        refs = grouped.agg([("label_min", "min"), ("label_max", "max")]).reset_index()
        refs["reference_class"] = np.where(refs.label_min.eq(refs.label_max), refs.label_min.astype(str), "conflicting")
        refs["fingerprint"] = refs.canonical_smiles.map(lambda s: generator.GetFingerprint(Chem.MolFromSmiles(s)))
        result[endpoint] = refs
    return result


def lookup_references(details: pd.DataFrame, references: dict) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    output = []
    for row in details.itertuples(index=False):
        if not row.canonical_smiles:
            continue
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        fingerprint = generator.GetFingerprint(mol)
        for endpoint, refs in references.items():
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, refs.fingerprint.tolist())
            nearest = int(np.argmax(similarities))
            match = refs.iloc[nearest]
            output.append({
                "input_row": row.row,
                "ingredient": row.resolved_ingredient or row.input_name,
                "endpoint": endpoint,
                "exact_structure_match": bool(similarities[nearest] == 1.0),
                "nearest_tanimoto": float(similarities[nearest]),
                "nearest_reference_class": match.reference_class,
                "interpretation": "화학구조 참고자료; 샴푸 자극성 예측 아님",
            })
    return pd.DataFrame(output)


def project_inputs():
    training, features = load_inputs()
    registry = pd.read_csv(ROOT / "data/ingredient_registry.csv")
    return training, features, registry
