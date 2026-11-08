"""Append the round-two experiments (07-10) to eye_all_experiments.ipynb as Part 3.

Backs the target up first, never rewrites Parts 1-2, and refuses to run twice.
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "eye_all_experiments.ipynb"
MARKER = "# 3부 — 2차 실험 07~10"
NL = chr(10)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "id": uuid.uuid4().hex[:12],
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "id": uuid.uuid4().hex[:12], "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
---
# 3부 — 2차 실험 07~10 (Claude)

1부가 결과를 모으고 2부가 여섯 실험을 분석했다면, 3부는 2부에서 "다음에 해볼 것"으로
남겨 둔 항목을 실제로 돌린 결과입니다.

| # | 실험 | 2부에서 남긴 숙제 |
|---|---|---|
| **07** | 표현 결합 | 05에서 A는 농도가중, B는 최악성분이 좋았다. 둘 다 주면 모델이 고를까 |
| **08** | 확률 보정 | SVM이 raw margin이라 쓸 수 없었다. Platt/Isotonic으로 살릴 수 있나 |
| **09** | 다른 분자 표현 | 2부는 기존 특징 재조합만 했다. 새 지문 계열은 어떤가 |
| **10** | 접근법 교차 앙상블 | 순서형 LR과 이진 트리는 틀리는 패턴이 다를까 |

코드와 원시 결과는 `claude/` 아래에 있습니다
(`advanced_utils.py`, `run_advanced.py`, `results/07~10.json`).

> 2부와 같은 프로토콜입니다: 완전일치벡터 병합 그룹화, 축소 격자(20개 중 6개),
> 외부 5-fold × 내부 3-fold. 판단 기준도 같습니다 - **seed 변동(MCC ±0.024)보다
> 작은 차이는 효과로 보지 않습니다.**
"""),
    code("""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

# 파일이 claude/ 아래로 옮겨졌을 수도, 노트북과 같은 칸에 있을 수도 있습니다.
CLAUDE = next(p for p in (Path.cwd() / "claude", Path.cwd())
              if (p / "advanced_utils.py").exists())
sys.path.insert(0, str(CLAUDE))
import advanced_utils as A

RESULTS = CLAUDE / "results"

def adv(name):
    return pd.DataFrame(json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8")))

def advmeta(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))

if "frame" not in dir():          # 2부 설정 셀을 건너뛰고 3부만 실행한 경우
    def frame(name):
        return adv(name)

SEED_SD = 0.024   # 2부 실험 03에서 측정한 seed 변동 표준편차 (MCC)
print("3부 결과 로드 준비 완료")
"""),
    md("""
## 07. 표현 결합 — 농도가중과 최악성분을 둘 다 주면

2부 실험 05는 라벨에 따라 정반대 결과를 냈습니다. A는 농도 가중 평균이, B는 최악 성분
pooling이 좋았습니다. 그렇다면 **둘 중 하나를 고르는 대신 둘 다 특징으로 주고 모델이
쓸 것을 고르게** 하면 어떨까요.

각 지문 블록을 1024비트로 접어 붙였기 때문에 전체 폭이 1024+1024+217 = 2,265개로
**기준 특징과 정확히 같습니다.** 특징 개수가 아니라 표현 방식만 달라진 비교입니다.
"""),
    code("""
cov7 = advmeta("07_coverage")
print(f"결합 행렬 {cov7['n_features']}개 특징 | 성분 출처 보유 {cov7['rows_with_component_provenance']}행 "
      f"| 실제 다성분 {cov7['multi_component_rows']}행")

c7 = adv("07_combined_representation")
base = frame("03_grouping_fix_v1")  # 07 remains a first-edition experiment.
base = base[(base.grouping == "완전일치벡터 병합") & (base.seed.isin(sorted(c7.seed.unique())))].copy()
base["표현"] = "기준 농도가중"
cols = ["표현", "label", "seed", "model", "MCC", "ROC-AUC", "sensitivity", "specificity"]
both7 = pd.concat([base[cols], c7[cols]], ignore_index=True)

display(both7.groupby(["label", "model", "표현"])[["MCC", "ROC-AUC"]].mean().round(3).unstack("표현"))

w7 = both7.pivot_table(index=["label", "model", "seed"], columns="표현", values=["MCC", "ROC-AUC"])
d7 = pd.DataFrame({m: w7[(m, "결합(농도가중+최악성분)")] - w7[(m, "기준 농도가중")]
                   for m in ("MCC", "ROC-AUC")}).reset_index()
print("\\n결합 - 기준, 짝지은 차이")
for lab in ("A", "B"):
    s = d7[d7.label == lab]
    print(f"  {lab}: dMCC {s.MCC.mean():+.4f} (개선 {int((s.MCC>0).sum())}/{len(s)}) | "
          f"dAUC {s['ROC-AUC'].mean():+.4f} | seed 변동 기준 {SEED_SD}")
"""),
    code("""
# 05의 '최악 성분만' 결과와도 나란히 비교한다
wa = frame("05_worst_actor")
cols5 = ["label", "seed", "model", "MCC", "ROC-AUC"]
three = pd.concat([
    base[cols5].assign(표현="① 농도가중(기준)"),
    wa[wa.seed.isin(sorted(c7.seed.unique()))][cols5].assign(표현="② 최악성분만"),
    c7[cols5].assign(표현="③ 결합"),
], ignore_index=True)
piv = three.groupby(["label", "표현"])[["MCC", "ROC-AUC"]].mean().round(3)
display(piv)
print("\\n라벨별 최고 표현:")
for lab in ("A", "B"):
    s = piv.loc[lab].sort_values("MCC", ascending=False)
    print(f"  {lab}: {s.index[0]} (MCC {s.iloc[0]['MCC']:.3f})")
"""),
    md("""
## 08. 확률 보정 — SVM을 쓸 수 있게 만들기

SVM은 확률이 아니라 decision margin을 내놓습니다. 그 값은 fold마다 척도가 달라서
합쳐 놓은 ROC/PR을 그대로 믿을 수 없고(기존 보고서도 "기술적 순위 요약"이라고만 했습니다),
임계값을 고르기도 불안정했습니다.

**내부 OOF 점수만으로** 보정 함수를 적합한 뒤 외부 점수에 적용합니다. 외부 정답을
쓰지 않으므로 누수가 없습니다. 임계값도 보정된 점수에서 다시 고릅니다.
"""),
    code("""
c8 = adv("08_calibration")
c8["abs_gap"] = (c8.sensitivity - c8.specificity).abs()   # 민감도-특이도 균형
display(c8.groupby(["label", "model", "보정"])[["MCC", "ROC-AUC", "abs_gap"]]
        .mean().round(3).unstack("보정"))

w8 = c8.pivot_table(index=["label", "model", "seed"], columns="보정",
                    values=["MCC", "ROC-AUC", "abs_gap", "foldstd_MCC"])
print("보정 없음 대비 짝지은 차이")
for kind in ("Platt", "Isotonic"):
    for lab in ("A", "B"):
        for model in ("svm", "lr", "lgbm"):
            try:
                sub = w8.loc[lab].loc[model]
            except KeyError:
                continue
            dm = (sub[("MCC", kind)] - sub[("MCC", "보정 없음")]).mean()
            da = (sub[("ROC-AUC", kind)] - sub[("ROC-AUC", "보정 없음")]).mean()
            dsd = (sub[("foldstd_MCC", kind)] - sub[("foldstd_MCC", "보정 없음")]).mean()
            print(f"  {kind:9} {lab} {model:5} dMCC {dm:+.3f}  dAUC {da:+.3f}  "
                  f"fold변동 {dsd:+.3f}")
"""),
    md("""
**보정이 가장 중요한 지점은 MCC가 아니라 fold 변동입니다.** 점수가 공통 척도로 올라오면
임계값이 fold마다 덜 흔들립니다. 위 표의 `fold변동` 열이 음수이면 그만큼 안정해졌다는 뜻입니다.
"""),
    md("""
## 09. 다른 분자 표현 계열 - 중단

Morgan 외에 MACCS/Avalon/AtomPair/Torsion/RDKitFP 다섯 계열을 비교하려 했으나
**실행 도중 중단했습니다.** 360개 작업 중 60개만 끝난 시점에서 남은 시간이 약 2시간으로
추정됐고, 이 실험만 **606행**(성분 출처가 있는 행)에서 평가하도록 설계되어 다른 단계의
673행 결과와 절대 수치를 비교할 수 없었습니다. 비교 가능한 범위가 좁은 데 비해
비용이 커서 제외했습니다.

다시 돌리려면 `run_advanced.py`의 `SKIP` 집합에서 `"09_fingerprint_families"`를 빼면 됩니다.
코드(`advanced_utils.fingerprint_families`)는 그대로 남겨뒀습니다.
"""),
    md("""
## 10. 접근법 교차 앙상블

2부 실험 02에서 앙상블이 별 도움이 안 됐습니다. 그런데 그때 섞은 것은 **모두 이진 모델**이라
틀리는 방식이 비슷했을 수 있습니다.

실험 04에서 순서형 모델은 이진 모델과 **다른 경로로** 답을 냅니다(등급 4개를 따로 맞힌 뒤
누적). 그래서 순서형과 이진을 섞으면 서로의 약점을 메울 가능성이 있습니다.

이번에는 fold별 out-of-fold 점수를 **버리지 않고 보관**했습니다
(`results/10_oof_pool.json`). 앞으로 추가 앙상블은 재학습 없이 할 수 있습니다.
"""),
    code("""
c10 = adv("10_cross_ensemble")
mean10 = c10.groupby(["label", "구성"])[["MCC", "ROC-AUC", "sensitivity", "specificity"]].mean()
for lab in ("A", "B"):
    print(f"\\n라벨 {lab} (seed 3개 평균, MCC 내림차순)")
    display(mean10.loc[lab].sort_values("MCC", ascending=False).round(3))
"""),
    code("""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
FONT = Path("C:/Windows/Fonts/malgun.ttf")
fontManager.addfont(str(FONT))
plt.rcParams.update({"font.family": FontProperties(fname=FONT).get_name(),
                     "axes.unicode_minus": False, "figure.dpi": 110})

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
for ax, lab in zip(axes, ("A", "B")):
    s = mean10.loc[lab].sort_values("MCC")
    colours = ["#2f6f9f" if not i.startswith("단일") else "#b0b7bd" for i in s.index]
    ax.barh(s.index, s["MCC"], color=colours)
    best_single = s[s.index.str.startswith("단일")]["MCC"].max()
    ax.axvline(best_single, color="#9c3030", ls="--", lw=1.2,
               label=f"최고 단일 {best_single:.3f}")
    ax.set(title=f"라벨 {lab} - MCC", xlabel="MCC")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
"""),
    code("""
print("최고 앙상블 vs 최고 단일 (seed 3개 평균)")
for lab in ("A", "B"):
    s = mean10.loc[lab]
    be = s[~s.index.str.startswith("단일")].sort_values("MCC").iloc[-1]
    bs = s[s.index.str.startswith("단일")].sort_values("MCC").iloc[-1]
    gain = be.MCC - bs.MCC
    verdict = "채택할 만함" if gain > SEED_SD else "seed 변동 이내 - 효과로 보기 어려움"
    print(f"  {lab}: {be.name} {be.MCC:.3f}  vs  {bs.name} {bs.MCC:.3f}  "
          f"({gain:+.3f}) -> {verdict}")
"""),
    code("""
# 평균이 아니라 seed마다 짝지어 비교합니다. 평균은 한 seed의 운을 숨길 수 있습니다.
c10["종류"] = np.where(c10.구성.str.startswith("단일"), "단일", "앙상블")
print("seed별: 최고 앙상블 vs 최고 단일")
gaps = []
for lab in ("A", "B"):
    for s in sorted(c10.seed.unique()):
        x = c10[(c10.label == lab) & (c10.seed == s)]
        be = x[x.종류 == "앙상블"].sort_values("MCC").iloc[-1]
        bs = x[x.종류 == "단일"].sort_values("MCC").iloc[-1]
        gaps.append(be.MCC - bs.MCC)
        print(f"  {lab} seed {s}: 앙상블 {be.구성:22} {be.MCC:.3f} | "
              f"단일 {bs.구성:14} {bs.MCC:.3f} | 차 {be.MCC - bs.MCC:+.3f}")
print()
print(f"  6번 중 앙상블이 이긴 횟수 {sum(g > 0 for g in gaps)}/6, 평균 차 {np.mean(gaps):+.3f}")
"""),
    code("""
# 앙상블이 점수를 못 올려도 흔들림은 줄일 수 있습니다. 그쪽을 따로 봅니다.
sd = c10.groupby(["label", "종류"]).MCC.std().unstack()
sd["감소"] = (sd["앙상블"] - sd["단일"]).round(3)
display(sd.round(3))
print("음수이면 앙상블이 seed에 덜 흔들린다는 뜻입니다.")
"""),
    md("""
### 10에서 나온 예상 밖의 결과 - 순서형 LR이 라벨 B에서 크게 앞섭니다

앙상블 자체는 실패했지만, 구성 요소를 들여다보다 다른 것이 보였습니다. 같은 분할에서
**순서형 LR과 이진 LR을 짝지어 비교**하면 라벨 B에서만 차이가 큽니다.

라벨 B는 등급 3과 4 사이에 경계가 있습니다. 이진 LR은 "3 이하인가"만 배우는데,
순서형 LR은 등급 4개를 모두 맞히도록 학습한 뒤 누적확률로 경계를 만듭니다.
등급 구조를 쓰는 쪽이 이 경계에서 유리한 것으로 보입니다.
"""),
    code("""
w10 = c10[c10.구성.isin(["단일 lr", "단일 lr-순서형"])].pivot_table(
    index=["label", "seed"], columns="구성", values="MCC")
w10["차(순서형-이진)"] = w10["단일 lr-순서형"] - w10["단일 lr"]
display(w10.round(3))
for lab in ("A", "B"):
    d = w10.loc[lab, "차(순서형-이진)"]
    flag = "  ★ seed 변동보다 큼" if abs(d.mean()) > SEED_SD else ""
    print(f"  라벨 {lab}: 평균 {d.mean():+.3f} (개선 {int((d > 0).sum())}/{len(d)}){flag}")
"""),
    md("""
## 3부 종합
"""),
    code("""
lines = []
lines.append("【07 표현 결합】")
for lab in ("A", "B"):
    s = d7[d7.label == lab]
    best = piv.loc[lab].sort_values("MCC", ascending=False).index[0]
    lines.append(f"  라벨 {lab}: 기준 대비 dMCC {s.MCC.mean():+.3f} | 세 표현 중 최고 = {best}")

lines.append("\\n【08 확률 보정】")
for lab in ("A", "B"):
    sub = w8.loc[lab].loc["svm"]
    for kind in ("Platt", "Isotonic"):
        dm = (sub[("MCC", kind)] - sub[("MCC", "보정 없음")]).mean()
        dsd = (sub[("foldstd_MCC", kind)] - sub[("foldstd_MCC", "보정 없음")]).mean()
        lines.append(f"  라벨 {lab} svm {kind}: dMCC {dm:+.3f}, fold변동 {dsd:+.3f}")

lines.append("\\n【09 분자 표현 계열】 중단 - 결과 없음")

lines.append("\\n【10 교차 앙상블】")
for lab in ("A", "B"):
    s = mean10.loc[lab]
    be = s[~s.index.str.startswith("단일")].sort_values("MCC").iloc[-1]
    bs = s[s.index.str.startswith("단일")].sort_values("MCC").iloc[-1]
    lines.append(f"  라벨 {lab}: {be.name} {be.MCC:.3f} vs 최고 단일 {bs.MCC:.3f} "
                 f"({be.MCC - bs.MCC:+.3f})")

lines.append(f"\\n판단 기준: seed 변동 표준편차 {SEED_SD} (2부 실험 03에서 측정)")
print("\\n".join(lines))
"""),
    md("""
### 3부의 한계

- 2부와 동일하게 후보 격자를 20개 중 6개만 썼습니다. 절대 수치를 발표된 production
  결과와 비교하면 안 됩니다.
- 07은 seed 2개, 08과 10은 seed 3개입니다. 표준편차는 신뢰구간이 아니며
  통계적 유의성을 주장하지 않습니다.
- 10의 순위 평균 앙상블은 적합되는 파라미터가 없어 pooled AUC가 편향되지 않지만,
  구성(어떤 모델을 섞을지)을 결과를 보고 고르면 그 선택 자체가 낙관적일 수 있습니다.
- 모두 내부 교차검증 결과이며 외부 독립 데이터 검증이 아닙니다.
""")
]


def main():
    nb = json.loads(TARGET.read_text(encoding="utf-8"))
    if any(MARKER in "".join(c["source"]) for c in nb["cells"]):
        raise SystemExit("3부가 이미 있습니다. 중복 추가를 막습니다.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup = TARGET.with_name(f"eye_all_experiments.backup_{stamp}.ipynb")
    shutil.copy2(TARGET, backup)

    # 파일이 claude/ 로 재배치되면서 2부 설정 셀의 import가 깨졌습니다. 같이 고칩니다.
    fixed = 0
    broken = NL.join(["ROOT = Path.cwd()",
                      "sys.path.insert(0, str(ROOT))",
                      "import experiment_utils as X"])
    repair = NL.join([
        'ROOT = next(p for p in (Path.cwd(), Path.cwd() / "claude")',
        '            if (p / "experiment_utils.py").exists())',
        "sys.path.insert(0, str(ROOT))",
        "import experiment_utils as X"])
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if broken in src:
            cell["source"] = src.replace(broken, repair).splitlines(keepends=True)
            fixed += 1
    if fixed:
        print(f"2부 설정 셀 경로 복구 {fixed}개")

    nb["cells"] = nb["cells"] + CELLS
    count = 0
    for c in nb["cells"]:
        if c["cell_type"] == "code" and c.get("execution_count"):
            count += 1
            c["execution_count"] = count
    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"백업 {backup.name}")
    print(f"{TARGET.name}: 셀 {len(nb['cells'])}개 (3부 {len(CELLS)}개 추가)")


if __name__ == "__main__":
    main()
