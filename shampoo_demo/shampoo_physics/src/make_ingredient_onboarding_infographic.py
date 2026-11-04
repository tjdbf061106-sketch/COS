"""Generate a Korean infographic for safely onboarding new shampoo ingredients."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "new_ingredient_onboarding_protocol.png"
regular = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
bold = font_manager.FontProperties(fname="C:/Windows/Fonts/malgunbd.ttf")

fig, ax = plt.subplots(figsize=(18, 10), dpi=160)
fig.patch.set_facecolor("#F4F8FA")
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")


def round_box(x, y, w, h, edge, face="white", radius=0.16, lw=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    return patch


def step_box(x, y, w, h, number, title, body, color):
    round_box(x, y, w, h, "#C9D8DF", "white")
    ax.add_patch(plt.Circle((x + 0.43, y + h - 0.43), 0.28, color=color))
    ax.text(x + 0.43, y + h - 0.43, str(number), ha="center", va="center",
            fontproperties=bold, fontsize=16, color="white")
    ax.text(x + 0.82, y + h - 0.43, title, ha="left", va="center",
            fontproperties=bold, fontsize=16, color="#173C51")
    ax.text(x + w / 2, y + (h - 0.78) / 2, body, ha="center", va="center",
            fontproperties=regular, fontsize=13.5, color="#2D424E", linespacing=1.45)


def arrow(x1, y1, x2, y2, color="#2486A6", width=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=width, color=color))


# Title
ax.text(9, 9.62, "신규 샴푸 원료를 모델에 안전하게 추가하는 방법", ha="center", va="center",
        fontproperties=bold, fontsize=28, color="#123D54")
ax.text(9, 9.18, "SMILES만으로 즉시 예측하지 않고, 최소 실험과 독립 검증을 거쳐 지원 범위를 확장",
        ha="center", va="center", fontproperties=regular, fontsize=16, color="#58717E")

# Current limitation panel
round_box(0.45, 1.32, 4.05, 7.35, "#1F4C70", "#F9FCFD", radius=0.20, lw=2)
ax.add_patch(FancyBboxPatch((0.45, 7.85), 4.05, 0.82,
                            boxstyle="round,pad=0.04,rounding_size=0.20",
                            linewidth=0, facecolor="#1F4C70"))
ax.add_patch(plt.Rectangle((0.45, 7.85), 4.05, 0.38, color="#1F4C70", linewidth=0))
ax.text(2.48, 8.25, "현재 방식의 한계", ha="center", va="center",
        fontproperties=bold, fontsize=20, color="white")

round_box(0.85, 6.15, 3.25, 1.18, "#9BB8C8", "#EAF4F8")
ax.text(2.48, 6.89, "검증 원료 18종", ha="center", va="center",
        fontproperties=bold, fontsize=18, color="#1F5C7E")
ax.text(2.48, 6.45, "이름별 고정 입력 열", ha="center", va="center",
        fontproperties=regular, fontsize=14, color="#365665")
arrow(2.48, 6.15, 2.48, 5.65, color="#557B8F")
round_box(0.85, 4.42, 3.25, 1.18, "#9BB8C8", "white")
ax.text(2.48, 5.15, "지원 범위 안", ha="center", va="center",
        fontproperties=bold, fontsize=17, color="#207467")
ax.text(2.48, 4.72, "조합·투입량 물성 예측", ha="center", va="center",
        fontproperties=regular, fontsize=13.5, color="#365665")
arrow(2.48, 4.42, 2.48, 3.92, color="#C6553D")
round_box(0.78, 2.35, 3.40, 1.50, "#E06A4E", "#FFF3EE", lw=2)
ax.text(2.48, 3.34, "새 원료 · 새 공급사", ha="center", va="center",
        fontproperties=bold, fontsize=17, color="#B54028")
ax.text(2.48, 2.78, "물성 예측 근거 부족\nSMILES 유사도만으로 대체 불가", ha="center", va="center",
        fontproperties=regular, fontsize=13.5, color="#734334", linespacing=1.45)
ax.text(2.48, 1.78, "미지원 원료를 0%로 숨기면 안 됨", ha="center", va="center",
        fontproperties=bold, fontsize=13.5, color="#B54028")

# Main protocol panel
round_box(4.85, 1.32, 12.70, 7.35, "#17809B", "#F9FCFD", radius=0.20, lw=2)
ax.add_patch(FancyBboxPatch((4.85, 7.85), 12.70, 0.82,
                            boxstyle="round,pad=0.04,rounding_size=0.20",
                            linewidth=0, facecolor="#147B98"))
ax.add_patch(plt.Rectangle((4.85, 7.85), 12.70, 0.38, color="#147B98", linewidth=0))
ax.text(11.20, 8.25, "대안: 신규 원료 단계적 편입 프로토콜", ha="center", va="center",
        fontproperties=bold, fontsize=21, color="white")

xs = [5.20, 9.25, 13.30]
step_box(xs[0], 5.32, 3.65, 2.03, 1, "상업 원료 정보 등록",
         "공급사·제품명·INCI\n활성분·물·염·분자량\nSMILES 또는 혼합물 정보", "#1F5C82")
step_box(xs[1], 5.32, 3.65, 2.03, 2, "기존 원료와 비교",
         "구조·기능·이온성\n활성분·고형분·농도 비교\n유사성은 실험 설계에만 사용", "#1F5C82")
step_box(xs[2], 5.32, 3.65, 2.03, 3, "브리지 실험 설계",
         "기존 원료 1:1 대체\n저·중·고 농도 및 주요 조합\nActive Learning으로 실험 선택", "#1F5C82")
arrow(8.88, 6.34, 9.18, 6.34)
arrow(12.93, 6.34, 13.23, 6.34)

step_box(xs[0], 2.52, 3.65, 2.03, 4, "실제 물성 측정",
         "단기 안정성·탁도·점도\n동일 제조·측정 조건 유지\n반복 측정과 불확실성 기록", "#218A78")
step_box(xs[1], 2.52, 3.65, 2.03, 5, "재학습 · 독립 검증",
         "일부 신규 원료 조합은\n학습에서 완전히 제외\n기존·신규 성능 모두 확인", "#218A78")
step_box(xs[2], 2.52, 3.65, 2.03, 6, "적용범위 등록",
         "원료 + 공급사 + 등급\n검증 농도·조합·공정 명시\n모델 버전과 함께 배포", "#218A78")
arrow(15.12, 5.26, 15.12, 4.62)
arrow(13.27, 3.54, 12.98, 3.54)
arrow(9.22, 3.54, 8.92, 3.54)

# Decision and loop
cx, cy = 11.20, 1.67
ax.add_patch(Polygon([[cx, cy + 0.47], [cx + 0.92, cy], [cx, cy - 0.47], [cx - 0.92, cy]],
                     closed=True, edgecolor="#23526D", facecolor="#E8F4F7", linewidth=1.8))
ax.text(cx, cy, "검증 기준\n통과?", ha="center", va="center",
        fontproperties=bold, fontsize=13.5, color="#19455C", linespacing=1.2)
arrow(12.14, cy, 13.20, cy, color="#25916D")
ax.text(12.62, cy + 0.22, "예", ha="center", va="center", fontproperties=bold, fontsize=12, color="#208460")
round_box(13.25, 1.31, 3.90, 0.72, "#25916D", "#ECF8F2", lw=1.8)
ax.text(15.20, 1.67, "검증 범위에서 예측 지원", ha="center", va="center",
        fontproperties=bold, fontsize=14, color="#176D50")
arrow(10.27, cy, 9.18, cy, color="#D55A3D")
ax.text(9.70, cy + 0.22, "아니오", ha="center", va="center", fontproperties=bold, fontsize=12, color="#C04A31")
round_box(5.20, 1.31, 3.90, 0.72, "#D55A3D", "#FFF2ED", lw=1.8)
ax.text(7.15, 1.67, "추가 실험 후 다시 검증", ha="center", va="center",
        fontproperties=bold, fontsize=14, color="#B5432C")

# Bottom key message
round_box(0.45, 0.20, 17.10, 0.78, "#173F62", "#173F62", radius=0.14, lw=0)
ax.text(9, 0.59,
        "핵심: AI가 새 원료를 바로 승인하지 않고, 최소 실험으로 검증한 농도·조합·공정만 지원 범위에 편입",
        ha="center", va="center", fontproperties=bold, fontsize=16.5, color="white")

plt.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(OUTPUT)
