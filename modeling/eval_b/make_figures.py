"""figdata.json -> 발표/보고서용 PNG 4장."""
import os
import json
HERE2 = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- 공통 스타일 (surface / ink / grid / 계열색)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue / orange / aqua (검증 통과)


def _korean_font():
    """OS별로 있는 한글 폰트를 하나 고른다 (Windows=맑은 고딕, mac=애플고딕, Linux=Noto/나눔)."""
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR",
                 "NanumGothic", "Noto Sans KR", "Apple SD Gothic Neo"):
        if name in have:
            return name
    print("[경고] 한글 폰트를 찾지 못해 라벨이 깨질 수 있음")
    return plt.rcParams["font.family"][0]


plt.rcParams.update({
    "font.family": _korean_font(),
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.labelsize": 11,
    "legend.frameon": False, "legend.fontsize": 10,
    "figure.dpi": 200,
})


def tidy(ax, xgrid=True, ygrid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", visible=xgrid or ygrid)
    if xgrid:
        ax.grid(axis="y", visible=False)


def title(ax, main, sub=None):
    ax.set_title(main, loc="left", fontsize=13, color=INK, pad=16 if sub else 8, fontweight="semibold")
    if sub:
        ax.annotate(sub, xy=(0, 1.02), xycoords="axes fraction", fontsize=10, color=INK2, va="bottom")


D = json.load(open(os.path.join(HERE2, "figdata.json")))
OUT = os.path.join(HERE2, "figs")
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# Fig 1 — 단계별 MCC: 반복 CV 분포 vs A의 단일 분할 수치
# =====================================================================
stages = D["stages"]
fig, ax = plt.subplots(figsize=(9.2, 4.4))
ys = np.arange(len(stages))[::-1]
rng = np.random.default_rng(7)

for y, st in zip(ys, stages):
    v = np.array(st["mcc"])
    ax.scatter(v, y + rng.uniform(-0.13, 0.13, len(v)), s=26, color=S1, alpha=0.45,
               linewidths=0, zorder=3)
    m, sd = v.mean(), v.std()
    ax.plot([m - sd, m + sd], [y + 0.28] * 2, color=S1, lw=2, solid_capstyle="round", zorder=4)
    ax.scatter([m], [y + 0.28], s=70, color=S1, zorder=5,
               edgecolors=SURFACE, linewidths=2)
    ax.scatter([st["a_reported"]], [y - 0.26], s=95, color=S2, marker="D", zorder=5,
               edgecolors=SURFACE, linewidths=2)
    ax.annotate(f"{m:.3f}±{sd:.3f}", xy=(m, y + 0.28), xytext=(0, 11), textcoords="offset points",
                ha="center", fontsize=9, color=INK2)
    ax.annotate(f"{st['a_reported']:.3f}", xy=(st["a_reported"], y - 0.26), xytext=(0, -17),
                textcoords="offset points", ha="center", fontsize=9, color=INK2)

ax.set_yticks(ys)
ax.set_yticklabels([s["label"] for s in stages], fontsize=10, color=INK)
ax.set_ylim(-0.75, len(stages) - 0.35)
ax.set_xlabel("MCC (높을수록 좋음)")
tidy(ax)
title(ax, "A의 단계별 개선은 측정 변동폭 안에 있다",
      "반복 group 5-fold 10회(분할·초기화 시드 동시 변경) vs A가 보고한 단일 분할 1회 · skin, 양성 30건")
ax.legend(handles=[
    Line2D([], [], marker="o", color=S1, alpha=0.5, lw=0, markersize=7, label="반복 CV 각 회차"),
    Line2D([], [], marker="o", color=S1, lw=2, markersize=9, markeredgecolor=SURFACE,
           markeredgewidth=2, label="반복 CV 평균 ±1 표준편차"),
    Line2D([], [], marker="D", color=S2, lw=0, markersize=9, markeredgecolor=SURFACE,
           markeredgewidth=2, label="A 보고 (단일 분할 1회)"),
], loc="lower right", bbox_to_anchor=(1.0, -0.33), ncol=3)
fig.savefig(f"{OUT}/fig1_stage_mcc_variance.png", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# Fig 2 — threshold trade-off
# =====================================================================
sw = D["sweep"]
g = np.array(sw["grid"])
fig, ax = plt.subplots(figsize=(9.2, 4.8))
series = [("sensitivity", "Sensitivity", S1), ("specificity", "Specificity", S2), ("mcc", "MCC", S3)]
for key, name, c in series:
    m, sd = np.array(sw[key]["mean"]), np.array(sw[key]["sd"])
    ax.fill_between(g, m - sd, m + sd, color=c, alpha=0.13, linewidth=0)
    ax.plot(g, m, color=c, lw=2, solid_capstyle="round", zorder=3)
    ax.annotate(name, xy=(g[-1], m[-1]), xytext=(8, 0), textcoords="offset points",
                color=c, fontsize=10, va="center", fontweight="semibold")

ax.axvline(0.5, color=AXIS, lw=1, zorder=1)
ax.annotate("현재 0.5", xy=(0.5, 0.985), xytext=(5, 0), textcoords="offset points",
            fontsize=9, color=INK2, va="top")
ax.set_xlabel("분류 threshold")
ax.set_ylabel("지표값")
ax.set_xlim(0.02, 1.08)
ax.set_ylim(0.05, 1.0)
ax.set_xticks(np.arange(0.1, 0.96, 0.1))
ax.grid(axis="y", visible=True)
ax.grid(axis="x", visible=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.set_axisbelow(True)
title(ax, "threshold를 내려도 recall은 공짜로 생기지 않는다",
      "반복 group 5-fold 10회 OOF 평균, 음영 = ±1 표준편차 · MCC 최대점은 오히려 threshold 0.95")
fig.savefig(f"{OUT}/fig2_threshold_tradeoff.png", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# Fig 3 — B4 짝지은 차이
# =====================================================================
P = D["paired"]
cands = [("nz20", f"nonzero<20 제거 ({P['dims']['nz20']}차원)", S1),
         ("nz5", f"nonzero<5 제거 ({P['dims']['nz5']}차원)", S2),
         ("pca16", f"PCA 16성분 ({P['dims']['pca16']}차원)", S3)]
metrics = [("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
           ("gap", "Sens-Spec 격차\n(작을수록 좋음)"), ("mcc", "MCC"), ("roc_auc", "ROC-AUC")]

fig, ax = plt.subplots(figsize=(9.2, 5.0))
base = np.arange(len(metrics))[::-1]
offs = [0.24, 0.0, -0.24]
for (k, label, c), off in zip(cands, offs):
    xs, ys_, es = [], [], []
    for i, (key, _) in enumerate(metrics):
        d = P[k][key]
        xs.append(d["mean"]); ys_.append(base[i] + off); es.append(d["sd"] / np.sqrt(d["n"]))
    ax.errorbar(xs, ys_, xerr=es, fmt="o", color=c, markersize=8, lw=0, elinewidth=2,
                capsize=0, markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=4)

ax.axvline(0, color=AXIS, lw=1, zorder=1)
ax.annotate("변화 없음", xy=(0, base[-1] - 0.45), xytext=(6, 0), textcoords="offset points",
            fontsize=9, color=INK2, va="center")
ax.set_yticks(base)
ax.set_yticklabels([m[1] for m in metrics], fontsize=10, color=INK)
ax.set_ylim(base[-1] - 0.6, base[0] + 0.78)
ax.set_xlabel("현재(전체 2,266차원) 대비 차이 — 같은 분할·같은 시드 20회 짝지은 비교")
tidy(ax)
title(ax, "차원축소는 MCC를 못 올리지만 격차와 Sensitivity를 개선한다",
      "점 = 회차별 차이의 평균, 막대 = 표준오차 · MCC는 세 후보 모두 0과 구분되지 않음")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.30), ncol=3)
fig.savefig(f"{OUT}/fig3_feature_reduction_paired.png", bbox_inches="tight")
plt.close(fig)

# =====================================================================
# Fig 4 — 조기종료 진단 (두 패널, x축 공유 — 이중축 금지)
# =====================================================================
C = D["curve"]
ep = np.array(C["epoch"])
best = int(ep[int(np.argmin(C["val_loss"]))])
fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), sharex=True,
                         gridspec_kw={"height_ratios": [1.25, 1]})

a0 = axes[0]
a0.plot(ep, C["train_loss"], color=S1, lw=2, solid_capstyle="round", label="train loss")
a0.plot(ep, C["val_loss"], color=S2, lw=2, solid_capstyle="round", label="validation loss")
a0.annotate("train loss", xy=(ep[-1], C["train_loss"][-1]), xytext=(8, 0), textcoords="offset points",
            color=S1, fontsize=10, va="center", fontweight="semibold")
a0.annotate("validation loss", xy=(ep[-1], C["val_loss"][-1]), xytext=(8, 0), textcoords="offset points",
            color=S2, fontsize=10, va="center", fontweight="semibold")
a0.set_ylabel("loss")
a0.legend().remove()

a1 = axes[1]
a1.plot(ep, C["test_auc"], color=S3, lw=2, solid_capstyle="round")
a1.annotate("test ROC-AUC", xy=(ep[-1], C["test_auc"][-1]), xytext=(8, 0), textcoords="offset points",
            color=S3, fontsize=10, va="center", fontweight="semibold")
a1.set_ylabel("test ROC-AUC")
a1.set_xlabel("epoch")

for a in axes:
    a.axvline(best, color=AXIS, lw=1, zorder=1)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    a.set_axisbelow(True)
    a.grid(axis="y", visible=True)
    a.grid(axis="x", visible=False)
a0.annotate(f"조기종료가 채택하는 지점 (epoch {best})", xy=(best, max(C["val_loss"]) * 0.42),
            xytext=(10, 0), textcoords="offset points", fontsize=9, color=INK2,
            va="center", ha="left")
a1.set_xlim(0.5, 44)
title(a0, "조기종료는 '거의 학습되지 않은 모델'을 채택하고 있다",
      "skin fold 1개 · train loss는 5 epoch 안에 0에 수렴, validation loss는 epoch 2부터 단조 증가")
fig.savefig(f"{OUT}/fig4_earlystopping_diagnosis.png", bbox_inches="tight")
plt.close(fig)

print("saved:")
for f in sorted(os.listdir(OUT)):
    print(" ", f)
