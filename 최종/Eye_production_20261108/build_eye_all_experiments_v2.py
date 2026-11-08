"""Add executed production reporting cells without fitting or altering old artifacts."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "eye_all_experiments.ipynb"
DESTINATION = ROOT / "eye_all_experiments_v2.ipynb"
CLAUDE = ROOT / "claude"
IDENTITY = "f91f3c88787bec6be1bef4bbbfcb98b797e0d69c6fda8c9bd03d508661488f4b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


LOADER = r"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

PROD_ID = "f91f3c88787bec6be1bef4bbbfcb98b797e0d69c6fda8c9bd03d508661488f4b"

def prod_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prod_hash_json(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

def prod_record(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    assert record["identity"] == PROD_ID, f"다른 실험 identity: {path}"
    assert record["payload_sha256"] == prod_hash_json(record["payload"]), f"checksum 오류: {path}"
    return record["payload"]

# notebook 폴더 또는 repository의 최종/ 폴더에서 실행 가능. 개인 절대경로 없음.
prod_candidates = []
for parent in (Path.cwd(), *Path.cwd().parents):
    prod_candidates.extend([parent / "claude", parent,
        parent / "모델 실험/eye/claude", parent / "최종/모델 실험/eye/claude"])
PROD_CLAUDE = next((p for p in prod_candidates
                   if (p / "results/concentration_production" / PROD_ID[:16] / "config.json").exists()), None)
assert PROD_CLAUDE is not None, "claude/results/concentration_production/f91f3c88787bec6b 자료가 필요합니다."
PROD_RUN = PROD_CLAUDE / "results/concentration_production" / PROD_ID[:16]
prod_config = prod_record(PROD_RUN / "config.json")
assert prod_hash_json(prod_config) == PROD_ID
prod_metrics = pd.DataFrame(prod_record(PROD_RUN / "metrics.json"))
prod_paired = pd.DataFrame(prod_record(PROD_RUN / "paired_comparison.json"))
prod_export = prod_record(PROD_RUN / "export_manifest.json")
prod_delivery_path = PROD_CLAUDE / "results/continuation_20261108T111518/delivery_verification.json"
prod_delivery = json.loads(prod_delivery_path.read_text(encoding="utf-8"))
assert prod_delivery["identity"] == PROD_ID
assert prod_delivery["status"] == "delivery_verified"
for name, expected in prod_delivery["final_artifacts_sha256"].items():
    assert prod_sha(PROD_RUN / name) == expected, f"완료 후 파일 변경: {name}"
assert prod_config["input_sha256"] == prod_export["source_sha256"] == prod_delivery["input_eye_sha256"]
assert prod_export["output_sha256"] == prod_sha(PROD_RUN / "eye_vectorized_concentration.csv")
assert prod_export["rows"] == 673 and prod_export["feature_count"] == 2269
assert prod_config["seeds"] == [42, 202, 777]
assert set(prod_config["grids"]) == {"lr", "rf", "et", "svm", "xgb", "lgbm"}
assert all(len(grid) == 20 for grid in prod_config["grids"].values())
assert len(prod_config["features"]["baseline"]) == 2265
assert len(prod_config["features"]["concentration"]) == 2269
print("검증된 run identity:", PROD_ID)
print("Eye 입력 SHA256:", prod_config["input_sha256"])
print("최종 결과 4개 파일의 SHA256 일치. 파생 CSV는 checksum만 확인하며 CV 입력으로 읽지 않습니다.")
"""

OUTER_CHECK = r"""
from sklearn.metrics import matthews_corrcoef, roc_auc_score, confusion_matrix
from itertools import product

prod_groups = np.asarray(prod_config["groups"])
assert len(prod_groups) == 673
prod_evaluation = sorted(i for f in prod_config["splits"]["42"] for i in f["test"])
assert len(prod_evaluation) == len(set(prod_evaluation)) == 606
prod_evaluation_set = set(prod_evaluation)
prod_augmentation = set(range(673)) - prod_evaluation_set
assert len(prod_augmentation) == 67
prod_outer = []
for arm, label, model, seed, fold_number in product(
        ["baseline", "concentration"], ["A", "B"],
        sorted(prod_config["grids"]), prod_config["seeds"], range(5)):
    path = PROD_RUN / arm / label / model / str(seed) / f"fold-{fold_number}" / "outer.json"
    r = prod_record(path)
    assert (r["arm"], r["label"], r["model"], r["seed"], r["fold"]) == (arm, label, model, seed, fold_number)
    f = prod_config["splits"][str(seed)][fold_number]
    assert r["test"] == f["test"] and r["train_rows"] == f["train"]
    assert r["calibration_rows"] == f["calibration"]
    assert r["candidates_evaluated"] == 20 and r["selected_candidate"] in prod_config["grids"][model]
    test, train, calibration = map(set, (f["test"], f["train"], f["calibration"]))
    assert not (test | calibration) & prod_augmentation
    assert not set(prod_groups[f["train"]]) & set(prod_groups[f["test"]])
    assert train & prod_augmentation == {i for i in prod_augmentation if prod_groups[i] not in prod_groups[f["test"]]}
    assert len(f["inner"]) == 3
    inner_coverage = []
    for part in f["inner"]:
        a, b = set(part["train"]), set(part["validation"])
        inner_coverage.extend(part["validation"])
        assert not b & prod_augmentation and a <= train
        blocked_groups = set(prod_groups[sorted(test | b)])
        assert not set(prod_groups[part["train"]]) & blocked_groups
        assert a & prod_augmentation == {i for i in prod_augmentation if prod_groups[i] not in blocked_groups}
    assert sorted(inner_coverage) == sorted(calibration)
    assert np.isfinite(r["threshold"]) and np.isfinite(r["scores"]).all()
    np.testing.assert_array_equal(np.asarray(r["scores"]) >= r["threshold"], r["predictions"])
    assert len(r["test"]) == len(r["y_true"]) == len(r["scores"]) == len(r["predictions"])
    prod_outer.append(r)
assert len(prod_outer) == 360

# 각 seed의 5개 outer fold를 합친 606행 OOF에서 지표를 다시 계산한다.
prod_recomputed = []
prod_outer_frame = pd.DataFrame(prod_outer)
for keys, parts in prod_outer_frame.groupby(["arm", "label", "model", "seed"], sort=True):
    parts = parts.sort_values("fold")
    idx = np.concatenate(parts.test.to_list())
    assert sorted(idx.tolist()) == prod_evaluation
    y = np.concatenate(parts.y_true.to_list())
    scores = np.concatenate(parts.scores.to_list())
    predictions = np.concatenate(parts.predictions.to_list())
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    prod_recomputed.append(dict(zip(["arm", "label", "model", "seed"], keys),
        MCC=matthews_corrcoef(y, predictions), sensitivity=tp/(tp+fn),
        specificity=tn/(tn+fp), **{"ROC-AUC": roc_auc_score(y, scores)}))
prod_recomputed = pd.DataFrame(prod_recomputed)
prod_keys = ["arm", "label", "model", "seed"]
prod_metric_names = ["MCC", "sensitivity", "specificity", "ROC-AUC"]
assert len(prod_metrics) == len(prod_recomputed) == 72
assert not prod_metrics.duplicated(prod_keys).any()
np.testing.assert_allclose(prod_recomputed.set_index(prod_keys).sort_index()[prod_metric_names],
    prod_metrics.set_index(prod_keys).sort_index()[prod_metric_names], rtol=1e-12, atol=1e-12)
prod_check_pairs = prod_metrics.pivot(index=["label", "model", "seed"], columns="arm", values="MCC")
prod_check_pairs["delta_MCC"] = prod_check_pairs.concentration - prod_check_pairs.baseline
assert len(prod_paired) == 36 and not prod_paired.duplicated(["label", "model", "seed"]).any()
np.testing.assert_allclose(prod_check_pairs.sort_index()[["baseline", "concentration", "delta_MCC"]],
    prod_paired.set_index(["label", "model", "seed"]).sort_index()[["baseline", "concentration", "delta_MCC"]],
    rtol=1e-12, atol=1e-12)
print("360개 outer checksum/분할 확인, 72개 OOF × 606행 coverage, 4개 성능지표 재집계 일치")
print("증강행의 outer 평가·inner validation·threshold calibration 포함: 0행")
print("validation/test 연결 그룹의 증강행이 해당 학습에서 제외됨을 확인했습니다.")
print("후보 7,200개 전체 완료는 기존 delivery_verification.json의 검증 기록입니다.")
print("이 절에서는 후보/모델을 학습하지 않았습니다.")
"""

TABLES = r"""
prod_names = {"lr": "LogisticRegression", "rf": "RandomForest", "et": "ExtraTrees",
              "svm": "SVM", "xgb": "XGBoost", "lgbm": "LightGBM"}
prod_order = ["lr", "rf", "et", "svm", "xgb", "lgbm"]
prod_stats = prod_metrics.groupby(["label", "model", "arm"])[prod_metric_names].agg(["mean", "std"])
prod_delta_stats = prod_paired.groupby(["label", "model"]).delta_MCC.agg(["mean", "std"])
assert prod_metrics.groupby(["label", "model", "arm"]).size().eq(3).all()
prod_mcc_rows = []
for label in ["A", "B"]:
    for model in prod_order:
        b = prod_stats.loc[(label, model, "baseline")]
        c = prod_stats.loc[(label, model, "concentration")]
        delta = prod_delta_stats.loc[(label, model)]
        prod_mcc_rows.append({"라벨": label, "모델": prod_names[model],
            "기준 MCC": f"{b[('MCC','mean')]:.4f} ± {b[('MCC','std')]:.4f}",
            "농도 MCC": f"{c[('MCC','mean')]:.4f} ± {c[('MCC','std')]:.4f}",
            "짝지은 ΔMCC": f"{delta['mean']:+.4f} ± {delta['std']:.4f}"})
display(pd.DataFrame(prod_mcc_rows).style.hide(axis="index"))
print("평균 ± 표본 표준편차(ddof=1), seed 42/202/777의 3개 값. Δ = 농도 − 기준.")
print("fold 평균/표준편차가 아니며, Δ 표준편차는 동일 seed의 차이 3개로 계산합니다.")

for label in ["A", "B"]:
    display(Markdown(f"**라벨 {label}: 4개 성능지표 (평균 ± 표준편차)**"))
    rows = []
    for model in prod_order:
        for arm in ["baseline", "concentration"]:
            stats = prod_stats.loc[(label, model, arm)]
            row = {"모델": prod_names[model], "입력": "기준" if arm == "baseline" else "농도 추가"}
            for key, title in [("MCC", "MCC"), ("sensitivity", "Sensitivity"),
                               ("specificity", "Specificity"), ("ROC-AUC", "ROC-AUC")]:
                row[title] = f"{stats[(key,'mean')]:.4f} ± {stats[(key,'std')]:.4f}"
            rows.append(row)
    display(pd.DataFrame(rows).style.hide(axis="index"))
"""

PLOT = r"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
for ax, label in zip(axes, ["A", "B"]):
    frame = prod_delta_stats.loc[label].reindex(prod_order)
    values, deviations = frame["mean"].to_numpy(), frame["std"].to_numpy()
    ax.barh(range(6), values, xerr=deviations, capsize=3,
            color=["#2478a5" if value >= 0 else "#8a6666" for value in values], alpha=0.85)
    ax.set_yticks(range(6), [prod_names[m] for m in prod_order])
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_title(f"Label {label}")
    ax.set_xlabel("MCC change: concentration minus baseline")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Paired change across 3 seeds: mean ± sample SD (not a confidence interval)", fontsize=10)
fig.tight_layout()
plt.show()
plt.close(fig)
"""

GROUPING_AND_PIPELINE = r"""
prod_grouping_path = PROD_CLAUDE / "results/03_grouping_fix_edition_comparison.csv"
prod_grouping = pd.read_csv(prod_grouping_path)
prod_verdict = json.loads((PROD_CLAUDE / "results/03_grouping_fix_verdict.json").read_text(encoding="utf-8"))
assert np.isclose(prod_grouping.query("edition == 'v2' and label == 'ALL'").delta_MCC.iloc[0], prod_verdict["delta_MCC"])
display(prod_grouping.pivot(index="label", columns="edition", values="delta_MCC").rename(
    columns={"v1": "1판 ΔMCC", "v2": "2판 ΔMCC"}).style.format("{:+.6f}"))
print("그룹화 2판 판정:", prod_verdict["decision"])
print("이 표는 증강까지 평가에 포함한 673행 실험입니다. production 606행 결과와 섞지 않습니다.")

prod_pipeline_path = PROD_CLAUDE / "results/pipeline_v2_20261108T144904/delivery_manifest.json"
prod_pipeline = json.loads(prod_pipeline_path.read_text(encoding="utf-8"))
assert prod_pipeline["original_evaluation_identity"] == PROD_ID
assert prod_pipeline["regression_tests_passed"] == 28
assert prod_sha(prod_pipeline_path.parent / "regression_tests.log") == prod_pipeline["regression_log_sha256"]
assert prod_sha(prod_pipeline_path.parent / "integration/verification.json") == prod_pipeline["integration_report_sha256"]
for name, expected in prod_pipeline["v2_code_sha256"].items():
    assert prod_sha(PROD_CLAUDE / name) == expected, f"pipeline 검증 이후 변경된 파일: {name}"
assert not prod_pipeline["production_model_deployed_or_replaced"]
display(pd.DataFrame([
    {"구분": "전체 성능 검증", "확인 내용": "360 outer / 72 OOF 세트 / 기존 606행; 기존 완료 결과 재집계"},
    {"구분": "코드 기능 검증", "확인 내용": "28개 테스트; LR 두 arm 각 121행의 기존 결과·저장·별도 CLI 예측 일치"},
    {"구분": "최종 모델 학습·배포", "확인 내용": "전체 673행 최종 후보 학습 및 배포·교체 미실행"},
]).style.hide(axis="index"))
prod_notebook_validation = {
    "identity": PROD_ID, "outer_records_verified": len(prod_outer),
    "oof_sets_verified": len(prod_recomputed), "rows_per_oof_set": len(prod_evaluation),
    "augmentation_evaluated": 0, "heldout_group_leakage": False,
    "recomputed_metrics_match": True, "paired_deltas_match": True,
    "pipeline_tests_previously_passed": 28,
    "new_model_fits": 0, "new_full_cv_performed": False,
}
print("NOTEBOOK_VALIDATION_JSON=" + json.dumps(prod_notebook_validation, ensure_ascii=False, sort_keys=True))
"""


def build():
    if DESTINATION.exists():
        raise FileExistsError(f"Existing notebook preserved: {DESTINATION}")
    original = nbformat.read(SOURCE, as_version=4)
    nbformat.validate(original)
    audit = (
        CLAUDE
        / "results"
        / ("notebook_production_v2_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    )
    audit.mkdir(exist_ok=False)
    run = CLAUDE / "results/concentration_production" / IDENTITY[:16]
    protected = [
        SOURCE,
        *[p for p in run.rglob("*") if p.is_file()],
        *[p for p in CLAUDE.glob("*") if p.is_file()],
        CLAUDE / "results/03_grouping_fix_edition_comparison.csv",
        CLAUDE / "results/03_grouping_fix_verdict.json",
    ]
    source_hashes = {str(p): sha(p) for p in protected}
    save_new(
        audit / "preservation_snapshot.json",
        json.dumps(source_hashes, ensure_ascii=False, indent=2),
    )
    notebook = deepcopy(original)
    notebook.cells[0].source = """# Eye 엔드포인트 — 전체 실험 통합본 v2

**2026-11-08 갱신: 현재 성능 판단은 5부 production 검증과 6부 종합 결론을 기준으로 합니다.**

| 부 | 내용 | 해석 범위 |
|---|---|---|
| 1부 | EPA / EPA+GHS / GJC 통합 집계 | 기존 실행별 프로토콜 유지 |
| 2부 | Claude 00~06 | 과거 673행 평가·축소 격자 탐색 |
| 3부 | Claude 07~10 | 과거 탐색 결과 |
| 4부 | Claude 11 농도 복원 | 과거 5모델·2seed·축소 격자 결과 |
| **5부** | **농도 production 검증** | **606행 평가 + 67행 train-only, 6모델·3seed·20후보 전체 격자** |
| **6부** | **갱신된 종합 결론 및 학습·추론 상태** | **현재 판단과 한계** |

이번 갱신은 완료된 결과를 읽고 검증·집계한 작업입니다. 모델을 새로 학습하지 않았습니다.
**1~4부의 코드·저장 출력은 과거 기록으로 보존했습니다.** 이 출력들을 최신 production 결과로
해석하거나 서로 다른 모집단/격자의 최고점을 한 순위로 합치면 안 됩니다.
특히 4부의 '채택'과 최고 MCC 0.264는 당시 탐색 판단이며 현재 결론은 6부에서 갱신합니다.

5부는 앞부분을 실행하지 않아도, **5부 첫 코드 셀부터 순서대로 독립 실행**할 수 있습니다.
필요 자료는 이 노트북 옆 `claude/`에 둡니다. 신규 절에는 개인 절대경로가 없습니다.
1~4부의 기존 경로와 실행 환경까지 이 갱신에서 이식한 것은 아닙니다.
"""
    warnings = {
        3: "**과거 실행별 비교 규약입니다. 갱신된 production 규약은 5부를 따릅니다.**\n\n",
        19: "**과거 00~06 탐색 기록 — 아래 673행·6후보는 현재 production 프로토콜이 아닙니다.**\n\n",
        31: "**보존된 표·그래프 출력은 그룹화 1판입니다. 2판의 확인된 차이와 판정은 5부에 별도로 제시합니다.**\n\n",
        43: "**과거 2부 종합입니다. 전체 최신 종합 결론은 6부에 있습니다.**\n\n",
        63: "**과거 3부 종합입니다. 다른 프로토콜의 production 순위로 사용하지 않습니다.**\n\n",
        66: "**과거 실험 11 기록 — 5모델·2seed·축소 격자이며, 5부의 production 결과와 구분합니다.**\n\n",
        72: "**아래 '채택'은 당시 탐색 판단입니다. 최신 결과에 근거한 채택·배포 확정이 아닙니다.**\n\n",
        78: "**과거 4부 결론을 보존한 절입니다. 현재 결론은 6부가 우선합니다.**\n\n",
        80: "**아래 해석·최고점·다음 작업은 과거 기록입니다. 최신 검증 완료 상태와 결론은 5~6부를 확인하세요.**\n\n",
    }
    for index, warning in warnings.items():
        notebook.cells[index].source = warning + notebook.cells[index].source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.metadata["historical_output_preserved"] = True

    appended = []

    def md(text):
        appended.append(nbformat.v4.new_markdown_cell(text.strip()))

    def code(text):
        cell = nbformat.v4.new_code_cell(text.strip())
        cell.metadata["tags"] = ["production-results-only", "no-training"]
        appended.append(cell)

    md("""---
# 5부 — 농도 production 검증 (완료)

기존 결과 identity: `f91f3c88787bec6be1bef4bbbfcb98b797e0d69c6fda8c9bd03d508661488f4b`.
학습은 2026-11-08에 완료됐으며 아래 표는 그 결과의 재집계입니다.

| 항목 | 이번 production 규약 |
|---|---|
| 입력/평가 | 673행 중 기존 **606행만 평가**, 증강 67행은 train-only |
| 그룹 | comp/split 연결 및 원본 fp/desc의 소수 6자리 반올림 중복을 병합 |
| 증강 제외 | outer test 또는 inner validation 그룹과 연결된 증강은 해당 학습에서 제외 |
| 비교 | baseline 2265개 vs 농도 피처 4개 추가 2269개; 원본 벡터 재스케일 없음 |
| 모델/seed | LR, RF, ExtraTrees, SVM, XGBoost, LightGBM × 42/202/777 |
| 검증 | outer 5-fold × inner 3-fold, 각 모델 **20후보 전체 격자**; inner 평균 AUC로 선택 |
| threshold | 기존행 inner OOF에서 민감도–특이도 차 최소화, MCC·작은 threshold 순으로 동률 해소 |
| 작업 수 | 2 arm × 2 label × 6 model × 3 seed × 5 fold = **360 outer** |
| 지표 집계 | seed별 5개 outer 예측을 합친 606행 OOF에서 계산 → 3seed 평균 ± 표본 표준편차 |
| 라벨 | A: EPA I/II 양성, B: EPA I/II/III 양성 |

농도는 `known_pct → pct_sum_used → 해당 학습 fold 중앙값` 순으로 보완합니다.
피처는 `log10(pct+0.001)`, `pct/100`, `pct>=90`, `undeclared_pct/100`이며
`undeclared_pct` 결측은 0입니다. validation/test로 중앙값을 다시 계산하지 않습니다.

## 5-1. 결과 checksum 및 실제 outer 예측 확인

아래 셀은 저장된 JSON을 검증하고 OOF 지표를 재계산합니다. 학습 실행기를 호출하지 않습니다.
""")
    code(LOADER)
    code(OUTER_CHECK)
    md("""## 5-2. 라벨·모델별 MCC 차이 및 전체 성능표

`ΔMCC = 농도 추가 − baseline`. 각 칸의 ±는 **3seed 표본 표준편차(ddof=1)**입니다.
짝지은 Δ의 표준편차는 같은 seed의 두 arm 차이로 계산하며 두 arm 표준편차를 더하지 않습니다.
3개 seed는 같은 데이터의 재분할이므로 독립 표본 기반 신뢰구간·유의성 검정으로 해석하지 않습니다.
""")
    code(TABLES)
    md("""### 농도 추가의 변화량

오차 막대는 짝지은 ΔMCC의 seed 표준편차입니다. 신뢰구간이나 통계적 동등성 판정이 아닙니다.
기존 탐색에서 사용한 공통 ±0.024 기준을 이번 결과의 채택 기준으로 옮기지 않습니다.
""")
    code(PLOT)
    md("""## 5-3. 그룹화 결과와 파이프라인 검증 상태

그룹화 실험03은 **673행 CV**로 별도 연구입니다. 00~02 및 04~11의 과거 캐시를
2판 그룹화로 재계산한 것처럼 해석하지 않으며, historical 비교 기준은 `03_grouping_fix_v1.json`입니다.

전체 성능 검증 완료와 코드 테스트 통과는 서로 다른 근거입니다. 아래에서는 두 상태를 구분합니다.
""")
    code(GROUPING_AND_PIPELINE)

    metrics = pd.DataFrame(
        json.loads((run / "metrics.json").read_text(encoding="utf-8"))["payload"]
    )
    pairs = pd.DataFrame(
        json.loads((run / "paired_comparison.json").read_text(encoding="utf-8"))[
            "payload"
        ]
    )
    mean = metrics.groupby(["label", "model", "arm"]).MCC.mean()
    delta = pairs.groupby(["label", "model"]).delta_MCC.agg(["mean", "std"])
    a_delta, bx_delta = delta.loc[("A", "lr")], delta.loc[("B", "xgb")]
    md(f"""---
# 6부 — 갱신된 종합 결론

**현재 성능 판단은 5부의 606행 production 검증을 기준으로 합니다.**
1~4부는 모델·표현을 탐색했던 과거 기록이며 같은 프로토콜로 모두 재검증된 결과가 아닙니다.

1. **라벨 A:** 평균 MCC가 가장 높은 모델은 두 arm 모두 LR입니다.
   baseline **{mean.loc[("A", "lr", "baseline")]:.6f} → 농도 추가 {mean.loc[("A", "lr", "concentration")]:.6f}**,
   짝지은 변화는 **{a_delta["mean"]:+.6f} ± {a_delta["std"]:.6f}**입니다.
   개선 폭이 작으므로 농도 피처가 A 성능을 크게 올렸다고 결론내리지 않습니다.
2. **라벨 B:** 농도 추가 XGBoost의 평균 MCC는 **{mean.loc[("B", "xgb", "concentration")]:.6f}**로
   비교한 조합 중 가장 높습니다. 같은 모델 baseline 대비 변화는
   **{bx_delta["mean"]:+.6f} ± {bx_delta["std"]:.6f}**로 seed 변동이 큽니다.
   농도 RF **{mean.loc[("B", "rf", "concentration")]:.6f}**, 농도 LightGBM
   **{mean.loc[("B", "lgbm", "concentration")]:.6f}**도 비교 후보로 남습니다.
   평균 순위만으로 XGBoost의 우월성이나 최종 채택을 확정하지 않습니다.
3. **농도 피처의 해석:** 효과는 라벨·모델에 따라 다릅니다. 과거 실험11의
   'B에서 seed 변동 초과, A에 무해 → 채택' 문구를 이번 검증의 종합 결론으로 유지하지 않습니다.
   원래 fp/desc를 농도로 재스케일하는 희석 실험은 이번 production의 비교 대상이 아닙니다.
4. **그룹화:** 2판 전체 ΔMCC는 **−0.014205**입니다. 지정 기준에서 중간 구간이므로
   '걱정은 기우' 또는 '영향 없음'으로 확정하지 않습니다. 이 실험은 증강까지 평가한
   673행 CV이며 production 결과와 분리해 해석합니다.
5. **적용 상태:** 평가·학습·추론의 공통 진입점과 모델 저장 기능은 구현됐고 28개 테스트 및
   실제 LR fold의 예측 일치 검증이 완료됐습니다. **전체 673행 최종 후보 모델 학습,
   실제 배포 또는 기존 모델 교체는 아직 수행하지 않았습니다.**

검증은 동일 원자료 안의 내부 grouped CV입니다. 화장품 등 다른 도메인의 외부 성능이나
통계적 유의성·모델 동등성을 주장하지 않습니다. 과거 중복 라벨의 다수결 결과 역시
증명된 최대 MCC 상한으로 해석하지 않습니다.

## 앞으로 사용할 진입점

`claude/`에서 `python run_production_v2.py evaluate`를 평가 진입점으로 사용합니다.
`train`은 label/model/arm을 명시해 최종 후보를 만들고, `predict`는 모델에 저장된
피처 순서·전처리·학습 중앙값·threshold를 그대로 적용합니다.
사용법은 [PRODUCTION_V2.txt](claude/PRODUCTION_V2.txt)를 참고하세요.

`eye_vectorized_concentration.csv`는 **전체 학습용 파생 산출물**이며 CV에 사용하지 않았습니다.
이 노트북 역시 해당 CSV의 feature를 평가에 넣지 않고 checksum만 확인합니다.

GitHub 업로드 위치는 `최종/모델 실험/eye/`를 기준으로 합니다. 5부 재실행에는
`claude/results/concentration_production/f91f3c88787bec6b/`의 결과·360개 outer 파일,
관련 완료 검증 기록과 pipeline v2 코드·검증 기록이 필요합니다.
원본 노트북·학습 코드·기존 결과의 checksum은 유지합니다.
""")

    # Only these new cells run. The original 40 code cells remain historical.
    section = nbformat.v4.new_notebook(
        cells=appended, metadata=deepcopy(original.metadata)
    )
    client = NotebookClient(
        section,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.validate(section)
    errors = [
        out
        for c in section.cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors
    all_output = "\n".join(
        str(o.get("text", "")) for c in section.cells for o in c.get("outputs", [])
    )
    marker = "NOTEBOOK_VALIDATION_JSON="
    validation_line = next(
        line for line in all_output.splitlines() if line.startswith(marker)
    )
    validation = json.loads(validation_line[len(marker) :])
    assert (
        validation["outer_records_verified"] == 360
        and validation["oof_sets_verified"] == 72
    )
    notebook.cells.extend(section.cells)
    notebook.metadata["production_update"] = {
        "at": datetime.now().astimezone().isoformat(),
        "source_notebook_sha256": source_hashes[str(SOURCE)],
        "identity": IDENTITY,
        "historical_code_cells_reexecuted": 0,
        "new_code_cells_executed": sum(c.cell_type == "code" for c in section.cells),
        "new_training_performed": False,
    }
    nbformat.validate(notebook)
    for old, new in zip(original.cells, notebook.cells[: len(original.cells)]):
        if old.cell_type == "code":
            assert old.source == new.source and old.outputs == new.outputs
    assert all(sha(p) == digest for p, digest in source_hashes.items())
    rendered = nbformat.writes(notebook)
    save_new(DESTINATION, rendered)
    restored = nbformat.read(DESTINATION, as_version=4)
    nbformat.validate(restored)
    assert len(restored.cells) == len(original.cells) + len(section.cells)
    assert sha(SOURCE) == source_hashes[str(SOURCE)]
    save_new(audit / "new_section_execution.txt", all_output)
    # Export the plot for an independent visual check; notebook embeds the same PNG.
    import base64

    pngs = [
        o["data"]["image/png"]
        for c in section.cells
        for o in c.get("outputs", [])
        if "image/png" in o.get("data", {})
    ]
    assert len(pngs) == 1
    with (audit / "delta_mcc_plot.png").open("xb") as stream:
        stream.write(base64.b64decode(pngs[0]))
    manifest = {
        "at": datetime.now().astimezone().isoformat(),
        "status": "completed",
        "source_notebook": str(SOURCE),
        "source_sha256": source_hashes[str(SOURCE)],
        "output_notebook": str(DESTINATION),
        "output_sha256": sha(DESTINATION),
        "builder_sha256": sha(__file__),
        "original_cells": len(original.cells),
        "output_cells": len(restored.cells),
        "new_code_cells_executed": notebook.metadata.production_update.new_code_cells_executed,
        "historical_code_and_saved_outputs_preserved": True,
        "historical_code_cells_reexecuted": 0,
        "source_files_preserved": len(source_hashes),
        "nbformat_validation": "passed",
        "notebook_verification": validation,
        "plot_sha256": sha(audit / "delta_mcc_plot.png"),
        "new_section_execution_sha256": sha(audit / "new_section_execution.txt"),
        "new_full_cv_performed": False,
        "final_model_deployed_or_replaced": False,
        "github_target": "최종/모델 실험/eye/eye_all_experiments_v2.ipynb",
    }
    save_new(
        audit / "completion_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    for output in [sys.stdout, sys.stderr]:
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    build()
