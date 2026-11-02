"""Produce a standalone Korean research report, figures and repository usage notes."""
from __future__ import annotations

import base64
import html
import importlib.metadata
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from common import ROOT, read_json, sha256, write_json


def table(frame):
    return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}", na_rep="—", escape=True)


def run():
    qc = read_json(ROOT / "data/qc.json")
    audit = read_json(ROOT / "data/skin_eye_source_audit.json")
    candidate = read_json(ROOT / "results/candidate_summary.json")
    verification = read_json(ROOT / "results/verification.json")
    summary = pd.read_csv(ROOT / "results/metrics_summary.csv")
    oof = pd.read_csv(ROOT / "results/oof_predictions.csv", float_precision="round_trip")
    primary = summary[summary.scheme == "combination_holdout"]
    main_metrics = primary[primary.metric.isin(["roc_auc", "average_precision", "balanced_accuracy", "mae_log10", "r2_log10"])][["target", "model", "metric", "mean", "repeat_sd", "n_samples"]]
    stress = summary[(summary.scheme == "subsystem_holdout") & (summary.model == "extra_trees") & summary.metric.isin(["roc_auc", "mae_log10", "r2_log10"])][["target", "metric", "mean", "n_samples"]]
    def result(target, metric):
        return primary[(primary.model == "extra_trees") & (primary.target == target) & (primary.metric == metric)].iloc[0]
    stability = result("stability", "roc_auc")
    viscosity = result("viscosity_100", "r2_log10")
    turbidity = result("turbidity", "r2_log10")
    eta_mae = result("viscosity_100", "mae_log10")
    ingredient_audit = pd.read_csv(ROOT / "data/skin_eye_ingredient_audit.csv")
    audit_table = ingredient_audit[["endpoint", "ingredient_id", "exact_full_smiles_match", "nearest_parent_tanimoto", "ad_status"]]
    candidates = pd.read_csv(ROOT / "results/candidate_predictions.csv")
    candidate_table = candidates[["candidate_id", "parent_a", "parent_b", "stability_score_uncalibrated", "predicted_viscosity_100_mPa_s", "predicted_turbidity_ntu", "point_target_match", "conservative_error_band_target_match"]].head(10)
    measured = pd.read_csv(ROOT / "results/observed_starting_formulations.csv")
    measured_table = measured[["sample_id", "viscosity_100_mPa_s", "turbidity_ntu", "rheology_type"]].sort_values("turbidity_ntu").head(10)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    selected = oof[(oof.scheme == "combination_holdout") & (oof.seed == 20261102) & (oof.model == "extra_trees")]
    pred = selected[selected.target == "stability"]
    fpr, tpr, _ = roc_curve(pred.observed, pred.predicted)
    axes[0].plot(fpr, tpr, color="#167a74", lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="#a1a1aa")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Phase stability | grouped OOF")
    for ax, target, title in zip(axes[1:], ["viscosity_100", "turbidity"], ["Viscosity at 100 s^-1 | log10(mPa.s)", "Turbidity | log10(1 + NTU)"]):
        p = selected[selected.target == target]
        ax.scatter(p.observed, p.predicted, alpha=0.55, s=16, color="#2866a7")
        lo = min(p.observed.min(), p.predicted.min())
        hi = max(p.observed.max(), p.predicted.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#a1a1aa")
        ax.set(xlabel="Observed", ylabel="Out-of-fold predicted", title=title)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle("Extra Trees | first grouped CV repeat | no external validation", fontsize=12)
    figure_path = ROOT / "results/oof_diagnostics.png"
    fig.savefig(figure_path, dpi=160)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    image = base64.b64encode(buffer.getvalue()).decode("ascii")
    limitations = [
        "물성 모델의 학습 자료는 샴푸 실측자료입니다. skin·eye 라벨을 물성 라벨로 합치지 않았으며, 자극성 전이학습을 수행한 결과가 아닙니다.",
        "skin 544개·eye의 EPA 라벨 제형 606개는 성분 구조 비교와 적용범위 검토에 연결했습니다. 참조 구조 중 정확한 전체 SMILES가 일치하는 샴푸 대표 구조는 두 자료 모두 0개입니다. 이 결과는 사용 가능한 대표 SMILES에 한정하며, 성분 전체의 불일치나 적용범위 밖임을 확정하지 않습니다.",
        "상업용 원료의 대표 SMILES는 실제 분자 분포 전체가 아닙니다. 고분자와 Arlypon TT 혼합물은 완전한 구조 입력이 아니며, 원자료의 INCI/전하 표기 검토 항목도 보존했습니다.",
        "상안정성은 pH 5.8±0.2 조정 전후의 단기 판정이며, 최종 관찰은 36시간입니다. 장기 보관·가속 안정성이나 실제 사용 중 성능을 검증하지 않았습니다.",
        "점도·탁도 모델은 실측상 안정한 제형만으로 학습했습니다. 불안정 제형의 결측 물성은 0으로 채우지 않았습니다. 안정성 점수가 높은 후보라도 실제 물성 측정이 필요합니다.",
        "점도는 25℃, 전단속도 100 s⁻¹에서의 보간값입니다. 저전단 점도나 점도 등급과 동일하지 않습니다. S151은 비양수 값이 포함된 곡선이어서 이번 보수적 QC에서 점도 타깃만 제외했습니다.",
        "보고된 ±는 3회 분할에 따른 표준편차이며 신뢰구간이 아닙니다. 서로 다른 제품·시험에서 확보한 독립 외부검증은 수행하지 않았습니다.",
        "후보 오차대는 첫 반복의 OOF 절대 로그 오차 90백분위수입니다. 보정된 예측구간·동시 충족 확률·안전성 보증이 아닙니다. 안정성 점수도 별도로 확률 보정하지 않았습니다.",
        "skin의 EPA III·IV에는 중등도 또는 경미한 자극이 포함되므로 이를 무자극 라벨로 해석하지 않습니다. 샴푸 후보의 skin·eye 측정 라벨은 없습니다.",
    ]
    html_text = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>샴푸 물성 연구 · 첫 기준선</title><style>
body{{font-family:Malgun Gothic,Arial,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#172735;line-height:1.7;background:#fafbfc}}
h1,h2{{line-height:1.3}}h2{{margin-top:38px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{background:white;border:1px solid #d8e3e6;border-radius:12px;padding:18px}}.value{{font-size:28px;font-weight:700;color:#146e68}}.note{{border-left:4px solid #c18a21;background:#fff6df;padding:14px 18px}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #e0e5e9}}.scroll{{overflow-x:auto}}img{{width:100%}}a{{color:#176f9b}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}}}
</style><h1>샴푸 물성 설계 연구 · 첫 기준선</h1>
<p>상업용 원료 투입량 → 단기 상안정성 → 안정 제형의 탁도·점도. skin·eye 자료는 별도의 성분 비교·적용범위 검토에 연결했습니다.</p>
<div class="cards"><div class="card">샴푸 제형<div class="value">812</div>안정 294 / 불안정 518</div><div class="card">상안정성 ROC-AUC<div class="value">{stability['mean']:.3f} ± {stability.repeat_sd:.3f}</div>동일 원료 조합을 분리한 3회 5-fold</div><div class="card">현재 용도<div class="value">탐색 연구</div>독립 외부검증·자극성 검증 미수행</div></div>
<h2>결과 해석</h2><p>점도의 로그 척도 R²는 {viscosity['mean']:.3f}, 탁도의 로그 척도 R²는 {turbidity['mean']:.3f}입니다. 점도 MAE는 {eta_mae['mean']:.3f} log10 단위로, 10^MAE ≈ {10**eta_mae['mean']:.2f}의 배수 오차 척도에 해당합니다. 정밀 처방 확정보다 추가 실험 후보를 좁히는 단계입니다.</p>
<img alt="Grouped out-of-fold ROC and property parity plots" src="data:image/png;base64,{image}">
<h2>데이터와 평가 설계</h2><p>원료 18개의 실제 상업 원료 투입량을 그대로 입력합니다. 공개 성분끼리 100%로 재정규화하지 않습니다. 원료 조합 {qc['combination_groups']}개를 그룹으로 묶어 분할하고, 모든 모델·목표가 같은 외부 분할을 사용합니다. 점도 타깃 {qc['viscosity_100_targets']}개, 탁도 타깃 {qc['turbidity_targets']}개이며 다른 원료 {qc['subsystem_groups']}개 고분자·증점제 조합을 차례로 제외하는 추가 평가도 수행했습니다.</p>
<p>물 잔량은 원료 투입량 합에서 계산한 조정 전 추정값입니다. 상업 원료 안의 물과 pH 조정액 투입량을 완전히 복원한 최종 분자 조성이 아닙니다. pH·온도는 연구 조건이 고정되어 있어 학습 특성에 넣지 않았습니다.</p>
<h2>기준선 비교</h2><p>모델과 설정을 사전에 고정했습니다. 결과에 따라 더 좋은 모델만 골라 보고하지 않았습니다. 분류는 고정 임계값 0.5, 회귀는 로그 척도입니다. 단순 선형 모델도 비슷하거나 일부 지표에서 더 높은 결과를 보여, 복잡한 모델이 우수하다는 근거는 없습니다. 탁도 MAE 개선은 특히 작습니다.</p><div class="scroll">{table(main_metrics)}</div>
<h2>고분자·증점제 조합 제외 평가</h2><div class="scroll">{table(stress)}</div>
<h2>예시 목표와 후보</h2><p>예시 목표: η(100 s⁻¹) ≥ {candidate['min_viscosity_at_100_s_inv_mPa_s']:.0f} mPa·s, 탁도 ≤ {candidate['max_turbidity_ntu']:.0f} NTU, 미보정 상안정성 점수 ≥ {candidate['min_uncalibrated_stability_score']:.2f}. 사용자가 확정한 제품 사양이 아니며 CLI에서 바꿀 수 있습니다.</p>
<p>원료 조합이 같은 안정 제형 두 개 사이의 투입량을 보간해 미측정 후보 {candidate['candidate_count']}개를 생성했습니다. 점추정 기준 충족은 {candidate['point_target_matches']}개입니다. 실제 관측값으로 물성 목표를 충족하는 출발 제형은 {candidate['observed_starting_formulations']}개입니다.</p>
<div class="note"><strong>오차대까지 고려해 목표를 충족한 예측 후보: {candidate['conservative_error_band_target_matches']}개.</strong> 예측 후보를 성공 처방이나 저자극 제품으로 해석하지 않습니다. 우선 실측 출발 제형과 재현 조건을 확인하는 것이 적절합니다.</div>
<h3>실측 출발 제형 일부</h3><div class="scroll">{table(measured_table)}</div>
<h3>미측정 예측 후보 일부</h3><div class="scroll">{table(candidate_table)}</div>
<p><a href="results/observed_starting_formulations.csv">실측 출발 제형 CSV</a> · <a href="results/candidate_predictions.csv">후보별 전체 투입량·예측·오차대 CSV</a></p>
<h2>skin·eye 연결</h2><p>skin: {audit['skin']['rows']}개, 성분 미기재 함량 1% 초과 {audit['skin']['undeclared_over_1pct']}개. eye: EPA 라벨이 있는 제형 {audit['eye']['mixture_labeled_rows']}개. 사용 가능한 개별 분자 구조와 샴푸 대표 구조의 Morgan Tanimoto 유사도를 기록했습니다. 분자 유사도는 검증된 제형 적용범위 판정이나 독립성 확인을 대신하지 않습니다.</p><div class="scroll">{table(audit_table)}</div>
<h2>한계와 다음 검증</h2><ul>{''.join('<li>'+html.escape(s)+'</li>' for s in limitations)}</ul>
<p>다음 연구에서는 실측 출발 제형의 물성곡선·상업 원료 등급을 우선 확인하고, 농도와 계면활성제 상호작용 표현을 비교해야 합니다. 자극성까지 연결하려면 같은 샴푸 제형의 별도 피부·안자극 결과와 입력 호환성이 필요합니다.</p>
<h2>검증과 재현</h2><p>검증 상태: <strong>{verification['status']}</strong>. 원자료·기존 skin/eye 입력 해시, 조성 보존, 그룹 분할, 모든 저장 지표, 모델 재로딩, outer fold 0 재학습을 확인했습니다.</p><ul>{''.join('<li>'+html.escape(c)+'</li>' for c in verification['checks'])}</ul>
<p><a href="README.md">재현 방법</a> · <a href="protocol.json">고정 분석계획</a> · <a href="results/verification.json">검증 기록</a></p>
<h2>출처</h2><p><a href="https://www.nature.com/articles/s41597-024-03573-w">Chitre et al., Scientific Data (2024)</a> · <a href="https://springernature.figshare.com/articles/software/Liquid_Formulations_Dataset/25451878">샴푸 실측자료</a> · <a href="https://springernature.figshare.com/articles/dataset/Surfactants_Information/25451926">상업 계면활성제 정보</a> · <a href="https://springernature.figshare.com/articles/figure/Structures_of_Formulations_Ingredients/25451929">원료 구조 자료</a> · <a href="https://www.ecfr.gov/current/title-40/chapter-I/subchapter-E/part-156/subpart-D/section-156.62">EPA 피부·안자극 등급 정의</a></p>
</html>'''
    (ROOT / "report.html").write_text(html_text, encoding="utf-8")
    readme = f'''# 샴푸 물성 설계 연구 — 첫 기준선

샴푸 실측자료로 물성 모델을 만들고, 기존 skin·eye 자료는 성분 구조와 적용범위 검토에 연결한 연구입니다. **skin·eye 라벨을 물성 라벨로 합치거나 샴푸 자극성 전이학습을 수행한 결과가 아닙니다.**

## 현재 결과

- 실측 812개, 안정 294개 / 불안정 518개, 상업 원료 18개.
- 원료 조합 {qc['combination_groups']}개를 묶은 3회 5-fold 평가. 별도로 고분자·증점제 조합 {qc['subsystem_groups']}개를 차례로 제외하는 평가.
- Extra Trees 상안정성 ROC-AUC **{stability['mean']:.3f} ± {stability.repeat_sd:.3f}**. ±는 분할 반복 SD이며 신뢰구간이 아닙니다.
- 점도 log10 R² **{viscosity['mean']:.3f}**, 탁도 log10(1+NTU) R² **{turbidity['mean']:.3f}**. 정밀 처방 확정용으로 검증된 성능은 아닙니다.
- 선형 모델도 비슷하거나 일부 지표에서 더 높습니다. Extra Trees의 우월성이나 skin·eye 전이학습의 개선 효과를 주장하지 않습니다. 탁도 MAE는 상수 기준선보다 개선 폭이 작습니다.
- 예시 물성 목표를 실측상 충족하는 출발 제형 {candidate['observed_starting_formulations']}개. 미측정 보간 후보 {candidate['candidate_count']}개 중 점추정 충족 {candidate['point_target_matches']}개, 보수적 오차대까지 충족 **{candidate['conservative_error_band_target_matches']}개**.
- 기존 skin·eye 제형과 정확히 같은 샴푸/시험을 확인한 것이 아니며, 독립 외부검증은 미수행입니다.

브라우저에서 [report.html](report.html)을 열면 표와 그래프를 볼 수 있습니다.

## 실행

이 폴더를 기존 COS 저장소 최상위에 두고 실행합니다. 물성 예측/학습에는 이 폴더의 자료만 필요합니다. 원자료 준비와 skin·eye 검증에는 기존 `skin irritation/`, `eye irritation/` 파일도 필요합니다. Python 3.14 환경에서 확인했으며 정확한 라이브러리 버전은 `requirements.txt`에 있습니다.

```powershell
python -m pip install -r requirements.txt
python src/prepare_data.py
python -m unittest discover -s tests -v
python src/train_evaluate.py
python src/propose_candidates.py --min-viscosity 100 --max-turbidity 100 --min-stability-score 0.8
python src/verify.py --refit
python src/make_report.py
```

이미 저장된 결과를 확인하려면 `python src/verify.py`만 실행합니다. 모델 파일은 이 연구에서 생성한 신뢰할 수 있는 파일만 로드합니다. 준비 단계는 최초 실행 때 Figshare v1 파일을 받아 공개 MD5와 SHA-256을 확인하며, 이후에는 로컬 캐시를 사용합니다. 기존 학습 잠금과 입력/학습 코드가 달라지면 재학습을 거부합니다. 다른 설정 실험은 별도 버전 폴더에서 수행합니다.

예시 목표는 η(100 s⁻¹)≥100 mPa·s, NTU≤100입니다. 점도를 높이는 목표가 제품 적합성이나 저자극성을 의미하지 않습니다. 다른 목표로 후보를 다시 만들면 검증과 보고서 생성도 다시 실행해야 합니다.

## 파일

| 파일 | 역할 |
|---|---|
| `protocol.json` | 학습 전에 고정한 입력·타깃·모델·분할·QC 기준 |
| `sources/` | CC0 원본 JSON, 계면활성제 CSV, 구조 PPTX, 출처·해시 |
| `data/formulations.csv` | 원료 투입량·그룹·물성 타깃·시험조건 |
| `data/flow_curves.csv` | 원본의 전단속도별 점도와 표준편차 |
| `data/ingredient_registry.csv` | 상업 원료 등급·INCI·대표 구조·내부 함량 범위·검토사항 |
| `data/skin_eye_ingredient_audit.csv` | 피부/안자극 자료의 개별 성분과 구조 비교 |
| `data/endpoint_links.csv` | 각 샴푸 제형의 물성/skin/eye 자료 연결 상태 |
| `results/outer_folds.csv` | 모든 모델이 공유한 그룹 평가 분할 |
| `results/oof_predictions.csv` | 평가 시 해당 그룹을 학습에서 제외한 예측 |
| `results/metrics_summary.csv` | Dummy·선형·Extra Trees 결과를 모두 보존 |
| `results/observed_starting_formulations.csv` | 실제 측정으로 예시 목표를 충족한 제형 |
| `results/candidate_predictions.csv` | 같은 원료 조합 안에서 보간한 미측정 후보 |
| `models/` | 전체 적격 자료로 학습한 3개 물성 모델과 카드 |
| `results/verification.json` | 입력·분할·지표·재학습·후보 검증 기록 |

## 입력과 단위

- 모델 입력은 `feature_names.json`의 **18개 상업 원료 투입 질량%**입니다. 성분 합으로 다시 나누지 않습니다. 원료 등급을 바꾸면 같은 INCI여도 같은 입력으로 취급할 수 없습니다.
- `added_water_balance_pct_estimate = 100 - stock_sum_pct`: pH 조정 전 추정 잔량. 상업 원료 안의 물과 적정액을 완전히 복원한 최종 물 함량이 아닙니다. 원본 JSON에 물/적정액 별도 측정 열은 없습니다.
- `viscosity_100_mPa_s`: 25℃, 100 s⁻¹에서의 점도. 측정 범위 안에서 log–log 보간합니다. 전체 곡선에 비양수/비유한값 등이 있으면 이번 기준선에서는 점도 타깃을 제외합니다(S151 1개). S151의 안정성·탁도 결과는 유지합니다.
- `turbidity_ntu`: 420 nm 흡광도에서 보정한 탁도 proxy. 회귀는 log10(1+NTU)입니다.
- `stable`: 혼합 후 및 pH 조정 36시간 후 모두 안정한 경우 1. 장기 안정성 지표가 아닙니다.
- 안정성 점수는 별도로 보정한 확률이 아닙니다. 물성 회귀는 실측 안정 제형에 조건부입니다.
- S366은 투입량이 양수인 원료가 3개입니다. 누락으로 임의 보충하지 않고 공개 수치를 그대로 보존했습니다.

## skin·eye의 실제 사용 범위

현재 GitHub 커밋 `{audit['source_commit']}`의 skin 544개, eye EPA 라벨 제형 606개를 참조했습니다. 구조 캐시/원자료에서 실제로 확보되는 분자만 비교하고, 비교에 쓴 파일 해시를 남겼습니다. 완전한 분자 조성이 없는 상업 원료는 원료 ID로 물성 모델에 입력하며, 억지로 단일 SMILES로 치환하지 않습니다.

13개 샴푸 원료의 대표 SMILES와 사용 가능한 skin·eye 참조 분자 사이에서 정확한 전체 SMILES 일치는 각각 0개입니다. 나머지 5개 원료는 이 방식의 대표 구조 비교가 불가능합니다. 이는 대표 구조 수준의 제한된 조사이며 검증된 AD 판정이 아닙니다. 알려지지 않은 피부/안자극 결과는 `not_validated_for_shampoo`로 보존했습니다. 피부·눈 라벨을 서로 대신 쓰거나 물성 결과로 자극성을 추론하지 않습니다.

## 한계

{chr(10).join('- '+x for x in limitations)}

## 다음 우선순위

1. 실측 출발 제형 {candidate['observed_starting_formulations']}개의 원료 등급·상안정성·전체 유변곡선을 검토해 목표 제품의 측정 조건을 정합니다.
2. 18개 원료 투입량 기준선과 계면활성제 종류·총 활성분 범위·고분자 등급·원료 간 상호작용 입력을 같은 분할에서 비교합니다. 상업 원료 함량 범위를 단일 확정값으로 숨기지 않습니다.
3. 개선 실험은 별도 분석계획으로 버전 관리합니다. 현재 OOF로 계속 설정을 고르면 선택 편향이 생기므로 내부 선택과 외부 평가를 분리합니다.
4. 자극성 연결은 같은 샴푸 제형의 실측 피부/안자극 라벨과 입력 호환성이 확보된 뒤 별도로 검증합니다. 현재 후보를 저자극·안전 처방으로 표시하지 않습니다.

## 출처

- Chitre et al. (2024), [논문](https://www.nature.com/articles/s41597-024-03573-w).
- [샴푸 원자료](https://springernature.figshare.com/articles/software/Liquid_Formulations_Dataset/25451878), [계면활성제 정보](https://springernature.figshare.com/articles/dataset/Surfactants_Information/25451926), [원료 구조](https://springernature.figshare.com/articles/figure/Structures_of_Formulations_Ingredients/25451929): Figshare v1, CC0. 원본 파일과 출처를 보존했습니다.
- [EPA 분류 정의](https://www.ecfr.gov/current/title-40/chapter-I/subchapter-E/part-156/subpart-D/section-156.62).

이 연구 폴더만 새로 추가했습니다. 기존 데이터·모델·진행상황 문서는 수정하지 않았습니다.
'''
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "rdkit", "matplotlib", "requests"]
    (ROOT / "requirements.txt").write_text("\n".join(f"{p}=={importlib.metadata.version(p)}" for p in packages)+"\n", encoding="utf-8")
    manifest = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path != ROOT / "manifest.json":
            manifest[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
    write_json(ROOT / "manifest.json", {"files": manifest, "excludes": ["manifest.json", "__pycache__"]})
    print("Wrote report.html, README.md, requirements.txt and file manifest")


if __name__ == "__main__":
    run()
