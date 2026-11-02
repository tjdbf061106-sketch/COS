from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from app_core import (
    applicability,
    load_models,
    load_reference_fingerprints,
    lookup_references,
    predict_properties,
    project_inputs,
    resolve_formula,
)

RDLogger.DisableLog("rdApp.error")
st.set_page_config(page_title="샴푸 물성 설계", layout="wide")


@st.cache_resource
def cached_assets():
    training, features, registry = project_inputs()
    return training, features, registry, load_models(), load_reference_fingerprints()


training, features, registry, models, references = cached_assets()

st.title("샴푸 물성 설계 앱")
st.caption("등록 원료의 물성 예측 + 임의 SMILES의 Skin/Eye 구조 참고 검색")
st.warning(
    "물성 모델은 아래 18개 상업 원료로만 학습되었습니다. 등록되지 않은 원료는 SMILES를 입력해도 "
    "물성 예측에 사용할 수 없으며 Skin/Eye 구조 참고 검색만 수행합니다."
)

with st.sidebar:
    st.header("현재 지원 원료")
    st.dataframe(
        registry[["ingredient_id", "inci", "role"]],
        hide_index=True,
        use_container_width=True,
    )
    st.markdown(
        "**입력 원칙**\n"
        "- 상업 원료 제품명이 가장 신뢰도 높음\n"
        "- INCI는 제품이 여러 개면 선택 불가\n"
        "- SMILES는 대표 구조 일치에만 사용\n"
        "- 미등록 원료를 0%로 숨기지 않음"
    )

st.subheader("1. 처방 입력")
st.write("원료명 또는 SMILES 중 하나 이상과 공급제품 기준 투입량(wt%)을 입력하세요. 행은 자유롭게 추가할 수 있습니다.")
initial = pd.DataFrame(
    [
        {"ingredient": "Plantacare 2000", "smiles": "", "wt_pct": 8.0},
        {"ingredient": "Dehyton PK 45", "smiles": "", "wt_pct": 6.0},
        {"ingredient": "Arlypon TT", "smiles": "", "wt_pct": 1.5},
    ]
)
rows = st.data_editor(
    initial,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config={
        "ingredient": st.column_config.TextColumn("상업 원료명 또는 INCI"),
        "smiles": st.column_config.TextColumn("SMILES(선택)"),
        "wt_pct": st.column_config.NumberColumn("투입량 wt%", min_value=0.0, max_value=100.0, step=0.1),
    },
)

if st.button("처방 분석", type="primary", use_container_width=True):
    formula, details = resolve_formula(rows, registry, features)
    if details.empty:
        st.error("분석할 원료를 한 개 이상 입력하세요.")
        st.stop()

    st.subheader("2. 원료 확인 결과")
    st.dataframe(
        details[["row", "input_name", "resolved_ingredient", "status", "mapping_basis", "wt_pct", "message"]],
        hide_index=True,
        use_container_width=True,
    )

    reference_results = lookup_references(details, references)
    if not reference_results.empty:
        st.subheader("3. Skin/Eye 구조 참고 검색")
        st.info("가장 가까운 화학물질의 클래스는 참고자료이며 원료 또는 완제품의 자극성 예측값이 아닙니다.")
        st.dataframe(reference_results, hide_index=True, use_container_width=True)

    blockers = details.loc[~details.status.eq("resolved")]
    if not blockers.empty:
        st.error("미등록·모호·오류 원료가 있어 물성 예측을 중단했습니다. 지원 상업 원료명으로 수정해야 합니다.")
        st.stop()

    domain = applicability(formula, training, features)
    st.subheader("4. 물성 예측")
    a, b, c = st.columns(3)
    a.metric("등록 원료 합계", f"{domain['stock_sum_pct']:.2f} wt%")
    b.metric("물 잔량 추정", f"{domain['water_balance_pct_estimate']:.2f} wt%")
    c.metric("적용범위", domain["status"])

    if domain["status"] == "invalid":
        st.error(domain["message"])
        st.stop()
    if domain["status"] == "interpolation":
        st.success(domain["message"])
    else:
        st.warning(domain["message"])

    prediction = predict_properties(formula, models, features)
    p1, p2, p3 = st.columns(3)
    p1.metric("단기 상안정성 점수", f"{prediction['stability_score_uncalibrated']:.3f}")
    p2.metric("예상 점도 @100 s⁻¹", f"{prediction['viscosity_100_mPa_s']:.1f} mPa·s")
    p3.metric("예상 탁도", f"{prediction['turbidity_ntu']:.1f} NTU")
    st.caption("안정성 점수는 보정된 확률이 아닙니다. 점도·탁도는 실측 안정 처방에 조건부인 탐색적 추정값입니다.")

    result = {
        "resolved_formula_wt_pct": {key: float(value) for key, value in formula.iloc[0].items() if value > 0},
        "applicability": domain,
        "physical_property_prediction": prediction,
        "skin_eye_status": "reference_only_not_shampoo_prediction",
    }
    st.download_button(
        "분석 결과 JSON 다운로드",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="shampoo_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.markdown(
    "**해석 원칙:** 이 앱은 실험 후보를 좁히는 연구 보조 도구입니다. 장기 안정성, 실제 사용감, "
    "피부·안구 안전성, 규제 적합성 또는 제품 출시 결정을 대신하지 않습니다."
)
