"""Research demo with separate observations, hypothetical candidates and external intake."""
from pathlib import Path
import sys
import json
import hashlib

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_core import read_json, clean
from support_gate import resolve_verified_formula, joint_domain, guarded_prediction

st.set_page_config(page_title="샴푸 물성·원료 편입 연구", layout="wide")


@st.cache_resource
def assets():
    provenance = read_json(ROOT / "source_provenance.json")
    for name, expected in provenance["input_sha256"].items():
        assert hashlib.sha256((ROOT / "inputs" / name).read_bytes()).hexdigest() == expected, name
    models = {}
    for target in ["stability", "viscosity_100", "turbidity"]:
        path = PROJECT / "models" / (target + ".joblib")
        card = read_json(PROJECT / "models" / (target + ".card.json"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == card["model_sha256"]
        models[target] = joblib.load(path)
    return (pd.read_csv(ROOT / "inputs/formulations.csv"), read_json(ROOT / "inputs/feature_names.json"),
            pd.read_csv(ROOT / "inputs/ingredient_registry.csv"), models)


training, features, registry, models = assets()
st.title("샴푸 물성·원료 편입 연구")
st.caption("실측 근거와 연구용 예측을 구분하고, 미확인 원료·조건에서는 예측을 보류합니다.")
st.info("현재 예측기는 기존 18개 원료의 Extra Trees 기준선입니다. 입력 보강 비교 결과를 자동으로 적용하거나 신규 원료를 승인하지 않습니다.")
tabs = st.tabs(["실측 처방", "공개 원료 가정 후보", "실제 처방 확인", "연구 결과"])

with tabs[0]:
    st.subheader("공개 실측 처방 조회")
    sid = st.selectbox("제형 ID", training.sample_id.tolist(), key="observed_id")
    observed = training[training.sample_id == sid].iloc[0]
    st.write("단기 상안정성 실측:", "안정" if observed.stable == 1 else "불안정")
    st.dataframe(pd.DataFrame([{"상용 원료": f, "투입 질량%": observed[f]} for f in features if observed[f] > 0]), hide_index=True)
    st.dataframe(pd.DataFrame([{"점도 @100 s⁻¹ (mPa·s)": observed.viscosity_100_mPa_s, "탁도 (NTU)": observed.turbidity_ntu}]), hide_index=True)
    st.caption("빈 물성은 미측정 또는 QC 제외입니다. 이 처방들은 학습 자료에 포함돼 있어 모델의 외부 성능을 확인하는 용도로 사용할 수 없습니다.")

with tabs[1]:
    st.subheader("원 논문의 공급제품·제조 조건을 가정한 후보")
    st.write("현재 판매 원료의 동일성이나 실제 배치 상태가 확인된 처방은 아닙니다. 실제 제조 처방은 ‘실제 처방 확인’에서 검토합니다.")
    candidates = pd.read_csv(PROJECT / "results/candidate_predictions.csv")
    candidate_id = st.selectbox("기존 미측정 보간 후보", candidates.candidate_id.tolist(), key="candidate_id")
    candidate = candidates[candidates.candidate_id == candidate_id].iloc[0]
    initial = pd.DataFrame([{"ingredient": f, "wt_pct": float(candidate[f])} for f in features if candidate[f] > 0])
    rows = st.data_editor(initial, num_rows="dynamic", hide_index=True, key="study_recipe",
        column_config={"ingredient": st.column_config.TextColumn("원 논문의 상용 원료"),
                       "wt_pct": st.column_config.NumberColumn("투입 질량%", min_value=0, max_value=100)})
    if st.button("후보 근거 확인", key="check_study"):
        prepared = rows.copy()
        prepared["supplier"] = "BASF"
        prepared["grade"] = prepared.ingredient
        prepared["evidence_id"] = "source:" + prepared.ingredient.fillna("").astype(str)
        formula, detail, blockers = resolve_verified_formula(prepared, registry, features, read_json(ROOT / "config/source_material_evidence.json"), "study_assumption")
        st.dataframe(detail, hide_index=True)
        domain = joint_domain(formula, training, features) if formula is not None else {}
        regression_domains = {}
        if formula is not None:
            for target, column in [("viscosity_100", "viscosity_100_mPa_s"), ("turbidity", "turbidity_ntu")]:
                measured_stable = training[training.stable.eq(1) & training[column].notna()]
                regression_domains[target] = joint_domain(formula, measured_stable, features)
        conditions = {"pH": 5.8, "temperature_C": 25, "observation_hours": 36, "shear_rate_s_inv": 100}
        result = guarded_prediction(formula, blockers, domain, conditions, models, features, "study_assumption", regression_domains)
        result["domain"] = domain
        if result["status"] == "HOLD": st.warning("예측 보류: " + " / ".join(result["reasons"]))
        elif result["status"] == "OBSERVED":
            st.info("이미 측정된 처방입니다. 아래 실측값을 확인하세요.")
            st.dataframe(training[training.sample_id.isin(domain["observed_sample_ids"])][["sample_id", "stable", "viscosity_100_mPa_s", "turbidity_ntu"]], hide_index=True)
        else:
            st.write("관측 조성의 기하학적 보간: 실제 물성이나 오차를 보증하지 않습니다.")
            st.dataframe(pd.DataFrame(domain.get("convex_support", [])), hide_index=True)
            st.dataframe(pd.DataFrame([result["predictions"]]), hide_index=True)
            for message in result["reasons"]: st.caption(message)
        st.download_button("근거 기록 저장", json.dumps(clean(result), ensure_ascii=False, indent=2), "candidate_review.json", "application/json")

with tabs[2]:
    st.subheader("실제 공급 원료·처방 검토")
    st.write("제품명만으로 승인하지 않습니다. 공급사·등급·로트의 검토된 근거 기록이 필요합니다. 현재 외부 원료 승인 기록은 비어 있습니다.")
    upload = st.file_uploader("원료·투입량 CSV", type="csv")
    if upload is not None:
        try:
            actual = pd.read_csv(upload)
            formula, detail, blockers = resolve_verified_formula(actual, registry, features, read_json(ROOT / "config/external_material_evidence.json"), "external")
            st.dataframe(detail, hide_index=True)
            if blockers: st.warning("예측 보류: " + " / ".join(sorted(set(blockers))))
            else:
                st.info("원료 기록은 일치합니다. 제조·시험 조건 확인과 모델 적용범위 검토가 추가로 필요하여 이 화면에서는 수치를 표시하지 않습니다.")
                st.json(joint_domain(formula, training, features))
        except (ValueError, KeyError) as exc: st.error(str(exc))
    st.download_button("처방 입력 양식", (ROOT / "templates/external_formula.csv").read_bytes(), "external_formula.csv", "text/csv")
    st.caption("신규 실험의 A/B 분리는 별도 intake_split 도구로 수행합니다. 식별자가 없으면 독립성을 확인했다고 표시하지 않습니다.")

with tabs[3]:
    st.subheader("고정 계획으로 수행한 내부 비교")
    path = ROOT / "results/representation_repeat_metrics.csv"
    if path.exists():
        frame = pd.read_csv(path)
        metrics = ["Sensitivity", "Specificity", "MCC", "ROC-AUC"]
        st.write("양성=안정, 임계값 0.5. 아래는 3회 그룹 분할의 평균이며 신뢰구간이 아닙니다.")
        st.dataframe(frame[frame.target == "stability"].groupby(["features", "model"])[metrics].mean().reset_index(), hide_index=True)
        st.caption("민감도와 특이도의 수준과 차이를 함께 판단합니다. 한 지표가 높다는 이유로 모델을 채택하지 않습니다.")
    for name in ["representation_four_metrics.png", "onboarding_learning_curves.png", "selective_coverage.png"]:
        path = ROOT / "results" / name
        if path.exists(): st.image(str(path))
    st.write("모의 신규 원료 평가는 18개 원료를 차례로 가리고, 최종 처방을 고정한 채 개발 조합을 추가한 결과입니다. 실제 신규 원료·독립 외부검증은 아닙니다.")

st.caption("물성 연구용 도구입니다. 장기 안정성·자극성·제품 출시 적합성을 판정하지 않습니다.")
