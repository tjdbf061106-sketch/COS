"""Generate a Korean infographic explaining the shampoo property app."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "shampoo_app_principle.png"

regular = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
bold = font_manager.FontProperties(fname="C:/Windows/Fonts/malgunbd.ttf")

fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
fig.patch.set_facecolor("#F5F8FA")
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")


def box(x, y, w, h, title, body, color, title_color="white", body_size=14):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=1.5, edgecolor=color, facecolor="white",
    ))
    ax.add_patch(FancyBboxPatch(
        (x, y + h - 0.62), w, 0.62,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=0, facecolor=color,
    ))
    ax.add_patch(plt.Rectangle((x, y + h - 0.62), w, 0.30, color=color, linewidth=0))
    ax.text(x + w / 2, y + h - 0.31, title, ha="center", va="center",
            fontproperties=bold, fontsize=16, color=title_color)
    ax.text(x + w / 2, y + (h - 0.62) / 2, body, ha="center", va="center",
            fontproperties=regular, fontsize=body_size, color="#21313C", linespacing=1.55)


def arrow(x1, y1, x2, y2, color="#5B7180", width=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=width, color=color,
                                 connectionstyle="arc3,rad=0"))


ax.text(8, 11.55, "샴푸 물성 앱은 어떻게 작동할까?", ha="center", va="center",
        fontproperties=bold, fontsize=30, color="#123B4A")
ax.text(8, 11.10, "예측은 물성에만 · Skin/Eye는 참고자료로 분리", ha="center", va="center",
        fontproperties=regular, fontsize=17, color="#536A76")

box(4.6, 9.25, 6.8, 1.35, "① 처방 입력",
    "18개 상업 원료의 투입량(wt%)\n예: 계면활성제 10% + 폴리머 0.5% + 물 잔량",
    "#246B8E", body_size=15)

arrow(8, 9.25, 4.1, 8.45)
arrow(8, 9.25, 11.9, 8.45)

box(0.7, 6.25, 6.8, 2.15, "② 물성 예측 경로",
    "샴푸 실험 처방 812개에서 패턴 학습\n\n단기 상안정성 · 점도 · 탁도\n→ 새 처방의 대략적인 물성 추정",
    "#167A74", body_size=15)

box(8.5, 6.25, 6.8, 2.15, "③ Skin · Eye 참고 경로",
    "Skin 3,079행 · Eye 2,971행\n단일화학물질 구조 자료와 비교\n\n정확한 일치 · 구조 유사도 · 관련 라벨 조회",
    "#8A5A9B", body_size=15)

ax.text(4.1, 5.88, "모델 예측", ha="center", va="center", fontproperties=bold,
        fontsize=14, color="#167A74")
ax.text(11.9, 5.88, "예측이 아니라 참고", ha="center", va="center", fontproperties=bold,
        fontsize=14, color="#8A5A9B")

arrow(4.1, 6.25, 5.6, 5.05, color="#167A74")
arrow(11.9, 6.25, 10.4, 5.05, color="#8A5A9B")

box(3.1, 3.05, 9.8, 1.95, "④ 한 화면에서 함께 확인",
    "안정성 점수  |  예상 점도  |  예상 탁도  |  유사 실측 처방\nSkin/Eye: 원료별 구조 참고정보 + ‘샴푸 제형 미검증’ 표시",
    "#D4832F", body_size=16)

arrow(8, 3.05, 8, 2.45, color="#D4832F")

box(3.1, 0.55, 9.8, 1.85, "⑤ 최종 목적: 실험 후보를 좁히기",
    "유망한 처방을 먼저 실험하고 실패 가능성이 높은 후보는 뒤로 보냄\n앱 결과는 처방 확정이나 안전성 인증을 대신하지 않음",
    "#C54D4D", body_size=16)

ax.text(0.8, 0.18,
        "핵심: ‘샴푸를 자동 완성’하는 도구가 아니라, 시행착오를 줄이고 실험 순서를 정하는 연구 보조 도구",
        ha="left", va="center", fontproperties=bold, fontsize=14, color="#123B4A")

plt.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(OUTPUT)
