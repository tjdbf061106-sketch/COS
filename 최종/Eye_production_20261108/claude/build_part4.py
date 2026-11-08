"""eye_all_experiments.ipynb 뒤에 4부(실험 11 농도 복원)를 덧붙인다.

3부와 같은 원칙: 기존 셀은 읽지도 고치지도 않고 뒤에 append 한다.
중복 추가를 막고, 쓰기 전에 백업부터 뜬다.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "eye_all_experiments.ipynb"
MARKER = "# 4부 — 실험 11: 농도 복원"


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
# 4부 — 실험 11: 농도 복원 (Claude)

3부까지의 결론은 "무엇을 해도 MCC가 안 움직인다"였습니다. 표현을 바꿔도, 확률을 보정해도,
순서형과 이진을 섞어도 전부 seed 변동(±0.024) 안이었습니다.

4부는 **모델이 아니라 입력을 고칩니다.**

## 문제

production 데이터를 해시로 검사하다 찾았습니다. 평가 606행 중 **217행이 "완전히 같은
2265차원 입력인데 정답이 다른" 상태**입니다. 원인은 농도 가중 평균의 약분입니다.

단일성분 제형에서 가중 평균을 내면 농도가 수학적으로 사라집니다.

$$\\frac{p \\cdot \\mathbf{v}}{p} = \\mathbf{v}$$

그래서 같은 물질을 0.05%로 쓴 제형과 98%로 쓴 제형이 **모델에게 똑같은 입력**이 됩니다.
실제로 CASRN 82657-04-3은 19개 제형이 0.05%~98%에 걸쳐 있는데 전부 동일한 벡터이고,
정답은 등급 3과 4로 갈립니다. 어떤 모델도 이 행들을 구분할 수 없습니다.

충돌 217행 중 97.7%가 단일성분입니다. 그리고 **농도 값은 CSV에 이미 있습니다**
(`known_pct` 컬럼). feature 열(`fp_`, `desc_`)이 아니라서 모델에 전달되지 않았을 뿐입니다.

## 검증한 두 가지 방안

| 팔 | 방법 | 피처 수 |
|---|---|---|
| ① 기준 | 현재 그대로 | 2,265 |
| ② 농도 피처 추가 | 벡터는 두고 농도 4개를 덧붙임 | 2,269 |
| ③ 희석 스케일링 | 정규화를 빼서 희석을 벡터에 반영 | 2,265 |
| ④ 둘 다 | ② + ③ | 2,269 |

**③의 구현 요령**: 현재 벡터가 $\\sum p_i v_i / \\sum p_i$ 이므로

$$\\sum \\frac{p_i}{100} v_i = \\frac{\\sum p_i}{100} \\times \\text{현재 벡터}$$

즉 행마다 `pct_sum_used/100`을 곱하기만 하면 됩니다. featurization을 다시 돌릴 필요가
없어서 673행 전부에 적용되고, 기존 결과와도 비교 가능합니다.

> 프로토콜은 2·3부와 같습니다. 완전일치벡터 병합 그룹화, 축소 격자(20개 중 6개),
> 외부 5-fold × 내부 3-fold, seed 42/202. **et는 뺐습니다** - 비용이 가장 큰데
> 지금까지 한 번도 이긴 적이 없습니다. 기준 팔은 2부 실험 03의 결과를 재사용합니다
> (같은 분할, 같은 격자이므로 다시 돌릴 이유가 없습니다).
"""),
    code("""
# 3부 설정 셀을 먼저 실행했다면 adv/advmeta/SEED_SD가 이미 있습니다.
if "adv" not in dir():
    import json, sys
    from pathlib import Path
    import numpy as np, pandas as pd
    CLAUDE = next(p for p in (Path.cwd() / "claude", Path.cwd())
                  if (p / "advanced_utils.py").exists())
    RESULTS = CLAUDE / "results"
    def adv(name):
        return pd.DataFrame(json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8")))
    def advmeta(name):
        return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
    def frame(name):
        return adv(name)
    SEED_SD = 0.024

SEEDS11 = [42, 202]
MODELS11 = ["lr", "svm", "rf", "lgbm", "xgb"]
ARMS = ["① 기준", "② 농도 피처 추가", "③ 희석 스케일링", "④ 농도 피처 + 희석"]

c11 = adv("11_concentration")
base11 = frame("03_grouping_fix_v1")  # 11 remains a first-edition experiment.
base11 = base11[(base11.grouping == "완전일치벡터 병합")
                & (base11.seed.isin(SEEDS11))
                & (base11.model.isin(MODELS11))].copy()
base11["표현"] = "① 기준"
keep = ["label", "seed", "model", "표현", "MCC", "ROC-AUC",
        "sensitivity", "specificity", "foldstd_MCC"]
t11 = pd.concat([base11[keep], c11[keep]], ignore_index=True)
print(f"기준 {len(base11)}칸 + 신규 {len(c11)}칸 = {len(t11)}칸")
"""),
    md("""
## 11-1. 먼저 - 중복이 실제로 깨지는가

학습을 돌리기 전에 확인할 것이 있습니다. 두 방안이 **정말로 같은 입력을 갈라놓는지**
행렬을 직접 만들어 해시로 세어 봤습니다. 안 깨지면 돌릴 이유가 없습니다.
"""),
    code("""
meta11 = advmeta("11_meta")
chk = pd.DataFrame(meta11["duplicate_check"], index=["완전중복 행", "라벨충돌 행"]).T
chk["해소율"] = (1 - chk["라벨충돌 행"] / chk.loc["① 기준", "라벨충돌 행"]).map("{:.0%}".format)
display(chk)
print(f"피처 수: {meta11['arms']}")
print(f"농도 범위 {meta11['pct_range'][0]}% ~ {meta11['pct_range'][1]}%")
print(f"희석 계수 범위 {meta11['scale_range'][0]} ~ {meta11['scale_range'][1]}"
      "   <- 다섯 자릿수. 선형 모델에 무슨 일이 생길지 여기서 짐작됩니다.")
"""),
    md("""
**221행 → 18행. 92%가 해소됩니다.** 그리고 세 방안이 전부 18행으로 똑같습니다.

남은 18행은 농도까지 같은데 정답이 다른 경우라, 어떤 방법으로도 못 고칩니다.
비공개 부형제나 시험 조건의 차이일 것입니다.

여기서 이미 하나가 판정됩니다 - **방안 2는 중복 해소 측면에서 방안 1보다 나을 것이
없습니다.** 그렇다면 남은 질문은 "그래도 학습에는 더 좋은가"입니다.
""" ),
    code("""
piv11 = t11.pivot_table(index=["label", "model"], columns="표현", values="MCC")[ARMS]
display(piv11.round(3))
print("라벨 평균")
for metric in ("MCC", "ROC-AUC", "foldstd_MCC"):
    print(f"  --- {metric} ---")
    display(t11.groupby(["label", "표현"])[metric].mean().unstack()[ARMS].round(3))
"""),
    md("""
## 11-2. 방안 1 (농도 피처) - 채택

평균만 보면 한두 칸의 운에 끌려갈 수 있으므로 **seed와 모델마다 짝지어** 비교합니다.
라벨당 5개 모델 × 2개 seed = 10칸입니다.
"""),
    code("""
w11 = t11.pivot_table(index=["label", "model", "seed"], columns="표현", values="MCC")

def paired(a, b, title):
    print(title)
    for lab in ("A", "B"):
        d = (w11.loc[lab][a] - w11.loc[lab][b]).dropna()
        flag = "   ★ seed 변동 초과" if abs(d.mean()) > SEED_SD else ""
        print(f"    라벨 {lab}: 평균 {d.mean():+.3f}  개선 {int((d > 0).sum())}/{len(d)}"
              f"  최악 {d.min():+.3f}  최고 {d.max():+.3f}{flag}")

paired("② 농도 피처 추가", "① 기준", "방안 1: 농도 피처 추가")
print()
paired("③ 희석 스케일링", "① 기준", "방안 2: 희석 스케일링")
print()
paired("③ 희석 스케일링", "② 농도 피처 추가", "방안 2 vs 방안 1 (직접 비교)")
print()
paired("④ 농도 피처 + 희석", "② 농도 피처 추가", "방안 1+2 vs 방안 1")
"""),
    code("""
# 라벨 B에서 어느 모델이 농도를 쓰는가
d = (w11["② 농도 피처 추가"] - w11["① 기준"]).groupby(level=["label", "model"]).mean()
display(d.unstack("label").round(3).rename(columns=lambda c: f"라벨 {c} dMCC")
        .sort_values("라벨 B dMCC", ascending=False))
print("트리 계열(lgbm/rf/xgb)이 농도를 씁니다. 농도와 등급이 단조롭지 않아")
print("선형 모델은 거의 쓰지 못하고, 분기로 구간을 나눌 수 있는 트리가 가져갑니다.")
"""),
    md("""
라벨 A에서 효과가 없는 것도 설명이 됩니다. A의 경계는 등급 2와 3 사이인데,
충돌 217행의 등급은 대부분 3과 4입니다. **충돌이 B의 경계에 몰려 있어서**
A는 애초에 이 문제의 영향을 덜 받고 있었습니다.

## 11-3. 방안 2 (희석 스케일링) - 기각

라벨 A에서 무너집니다. 범인은 선형 모델입니다.
"""),
    code("""
lin = piv11.loc["A"].loc[["lr", "svm", "rf"], ["① 기준", "③ 희석 스케일링"]].copy()
lin["차"] = (lin["③ 희석 스케일링"] - lin["① 기준"]).round(3)
display(lin.round(3))
lo, hi = meta11["scale_range"]
print(f"희석 계수가 {lo} ~ {hi}, 다섯 자릿수에 걸쳐 있습니다.")
print("행마다 벡터 전체를 이만큼 다른 배율로 곱하면 열 단위 표준화로는 복구되지 않습니다.")
print("희석 0.001%짜리 행은 벡터가 거의 0이 되어, LR 입장에서는 원점 근처에 뭉친")
print("점 덩어리가 됩니다. 저희 최고 모델이 거기서 주저앉습니다.")
"""),
    code("""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
FONT = Path("C:/Windows/Fonts/malgun.ttf")
fontManager.addfont(str(FONT))
plt.rcParams.update({"font.family": FontProperties(fname=FONT).get_name(),
                     "axes.unicode_minus": False, "figure.dpi": 110})

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
for ax, lab in zip(axes, ("A", "B")):
    sub = piv11.loc[lab]
    x = np.arange(len(sub))
    width = 0.2
    for i, arm in enumerate(ARMS):
        colour = ["#9aa6b1", "#1f6f43", "#9c3030", "#a06a12"][i]
        ax.bar(x + (i - 1.5) * width, sub[arm], width, label=arm, color=colour)
    ax.set_xticks(x); ax.set_xticklabels(sub.index, fontsize=9)
    ax.set(title=f"라벨 {lab} - MCC", ylabel="MCC")
    ax.grid(axis="y", alpha=0.25)
    if lab == "A":
        ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
"""),
    md("""
## 4부 결론
"""),
    code("""
lines = []
lines.append("【중복 해소】")
for arm in ARMS:
    n = meta11["duplicate_check"][arm][1]
    lines.append(f"  {arm:16} 라벨충돌 {n:3}행")
lines.append("  -> 세 방안 모두 221행에서 18행으로. 방안 2에 추가 이득 없음.")

lines.append("")
lines.append("【학습 결과, 기준 대비 짝지은 평균 dMCC】")
for arm in ARMS[1:]:
    row = []
    for lab in ("A", "B"):
        d = (w11.loc[lab][arm] - w11.loc[lab]["① 기준"]).dropna()
        mark = "★" if abs(d.mean()) > SEED_SD else " "
        row.append(f"{lab} {d.mean():+.3f}{mark} ({int((d > 0).sum())}/{len(d)})")
    lines.append(f"  {arm:16} " + "   ".join(row))

lines.append("")
lines.append("【라벨별 최고】")
best = t11.groupby(["label", "표현", "model"]).MCC.mean().reset_index()
for lab in ("A", "B"):
    b0 = best[(best.label == lab) & (best.표현 == "① 기준")].MCC.max()
    top = best[best.label == lab].sort_values("MCC").iloc[-1]
    lines.append(f"  라벨 {lab}: 기준 최고 {b0:.3f}  ->  "
                 f"{top.표현} {top.model} {top.MCC:.3f} ({top.MCC - b0:+.3f})")

lines.append("")
lines.append("【판정】")
lines.append("  방안 1 (농도 피처)   채택  - B에서 seed 변동 초과, A에 무해, fold 변동 감소")
lines.append("  방안 2 (희석 스케일) 기각  - A에서 선형 모델 붕괴, B에서도 방안 1보다 못함")
lines.append("  방안 1+2             기각  - 방안 1 대비 이득 없음")
lines.append("")
lines.append(f"판단 기준: seed 변동 표준편차 {SEED_SD}")
print("\\n".join(lines))
"""),
    md("""
### 무엇이 달라졌나

지금까지 돌린 모든 실험 - 앙상블, 순서형, 표현 결합, 확률 보정, 최악성분 pooling -
가운데 **처음으로 seed 변동을 넘긴 개선**입니다. 그리고 그것은 모델을 바꿔서가 아니라
**입력에 없던 정보를 넣어서** 얻었습니다.

2부와 3부가 전부 제자리였던 이유가 이것으로 설명됩니다. 평가 606행 중 217행의 입력이
동일했으니, 어떤 모델도 그 행들을 구분할 수 없었습니다. 모델링 기법으로는 애초에
닿을 수 없는 문제였습니다.

### 솔직한 평가 - 기대만큼은 아닙니다

입력 그룹을 다수결로 맞힌다고 가정한 이론적 천장은 라벨 A 0.840 → 0.984,
B 0.787 → 0.982로 올라갑니다. 그런데 실측은 B 0.237 → 0.264입니다.

**천장이 열렸다고 모델이 거기까지 가는 것은 아닙니다.** 농도와 등급의 관계가
단조롭지 않습니다 - 같은 물질이 25.1%에서 등급 4인데 98%에서 등급 3입니다.
673행으로 그런 함수를 배우기는 어렵습니다.

### 4부의 한계

- seed가 2개뿐입니다(2·3부의 08·10은 3개). 짝지은 비교로 보완했지만
  신뢰구간이 아니며 통계적 유의성을 주장하지 않습니다.
- et를 뺐으므로 2·3부의 6모델 표와 절대 수치를 나란히 두면 안 됩니다.
- 라벨 B 최고 0.264는 ④(둘 다)에서 나왔지만, ④는 방안 1 대비 평균 +0.008로
  잡음 수준이고 10칸 중 한 칸입니다. **여러 조합 중 최고를 사후에 고르면 낙관적으로
  나옵니다.** 안전한 선택은 ② 단독입니다.
- 축소 격자(20개 중 6개)이므로 발표된 production 수치와 비교하면 안 됩니다.
- 내부 교차검증 결과이며 외부 독립 데이터 검증이 아닙니다.

### 다음

같은 방향으로 더 갈 수 있습니다.

| 할 일 | 근거 |
|---|---|
| `test_concentration`을 원본에서 복구 | 현재 673행 전부 결측. 시험 농도는 자극성의 직접 결정 인자 |
| 비공개 부형제(`undeclared_pct`) 세분 | 지금은 비율 하나뿐. 어떤 부형제인지가 남은 18행의 열쇠일 수 있음 |
| 농도 x 물성 교호작용 항 | 비단조 관계를 트리에만 맡기지 말고 명시적으로 제공 |
""")
]


def main():
    nb = json.loads(TARGET.read_text(encoding="utf-8"))
    if any(MARKER in "".join(c["source"]) for c in nb["cells"]):
        raise SystemExit("4부가 이미 있습니다. 중복 추가를 막습니다.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup = TARGET.with_name(f"eye_all_experiments.backup_{stamp}.ipynb")
    shutil.copy2(TARGET, backup)

    nb["cells"] = nb["cells"] + CELLS
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            count += 1
            cell["execution_count"] = count
    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"백업 {backup.name}")
    print(f"{TARGET.name}: 셀 {len(nb['cells'])}개 (4부 {len(CELLS)}개 추가)")


if __name__ == "__main__":
    main()
