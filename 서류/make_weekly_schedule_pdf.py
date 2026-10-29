# -*- coding: utf-8 -*-
"""주간 연구일정표 PDF 생성 (2026.08.03 ~ 2026.11.06)
붙임2._2026학년도_미래인재대학_학술제_연구계획서.hwpx의 '6. 일정 계획' 대분류 일정을
8/3(월) 시작 기준 주 단위로 세분화. 공유용으로 가독성/페이지수를 우선."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold', 'C:/Windows/Fonts/malgunbd.ttf'))

NAVY = colors.HexColor('#1a3d6d')
GRID = colors.HexColor('#c9cfd8')

title_style = ParagraphStyle('Title', fontName='MalgunBold', fontSize=16, leading=20,
                              alignment=1, spaceAfter=3)
subtitle_style = ParagraphStyle('Subtitle', fontName='Malgun', fontSize=10, leading=14,
                                 alignment=1, textColor=colors.HexColor('#444444'), spaceAfter=1)
date_style = ParagraphStyle('Date', fontName='Malgun', fontSize=9, leading=12,
                             alignment=1, textColor=colors.HexColor('#666666'), spaceAfter=10)
phase_head_style = ParagraphStyle('PhaseHead', fontName='MalgunBold', fontSize=10.3, leading=13,
                                   textColor=colors.white, alignment=0, leftIndent=2)
week_cell = ParagraphStyle('WeekCell', fontName='MalgunBold', fontSize=8.6, leading=11.5, alignment=1)
date_cell = ParagraphStyle('DateCell', fontName='Malgun', fontSize=8.6, leading=11.5, alignment=1)
task_cell = ParagraphStyle('TaskCell', fontName='Malgun', fontSize=8.8, leading=12.3)
note_style = ParagraphStyle('Note', fontName='Malgun', fontSize=8.3, leading=12,
                             textColor=colors.HexColor('#666666'), spaceAfter=3)
legend_style = ParagraphStyle('Legend', fontName='Malgun', fontSize=8.6, leading=12)

# 6개 대분류 단계(연구계획서 6.일정 계획) -> 색상 매핑
PHASES = [
    ("P1", "① 주제 확정·EDA·문헌조사", colors.HexColor('#1a3d6d')),
    ("P2", "② ICE 전처리·화장품 데이터 수집", colors.HexColor('#2e6da4')),
    ("P3", "③ 라벨링·전이학습 파이프라인 구축", colors.HexColor('#3d8b6b')),
    ("P4", "④ 모델 개발·초기 실험", colors.HexColor('#b8860b')),
    ("P5", "⑤ 도메인 갭·AD 분석·모델 보완", colors.HexColor('#a5502a')),
    ("P6", "⑥ 보고서 작성·발표 준비", colors.HexColor('#8a3a3a')),
]

# (주차, 기간, 활동, phase_idx)
WEEKS = [
    (1, "8/3 (월) ~ 8/9 (일)",
     "연구 주제·역할 최종 확정, NTP ICE 다운로드 데이터(skin/eye_irritation.xlsx) 구조 확인 및 1차 EDA 착수", 0),
    (2, "8/10 (월) ~ 8/16 (일)",
     "EDA 심화(Mixture/EPA Classification 분포, 클래스 불균형 규모 파악), FIFRA/EPA 자극성 분류·도메인 적응/전이학습 선행연구 조사 마무리", 0),
    (3, "8/17 (월) ~ 8/23 (일)",
     "Formulation_ID 기준 성분 조성 파싱 로직 설계 및 구현 착수(성분명·CAS·활성성분 비율 추출)", 1),
    (4, "8/24 (월) ~ 8/30 (일)",
     "결측·단일성분 제형 처리 방침 수립(baseline vs 다성분 서브셋 비교 설계), 전처리 파이프라인 마무리", 1),
    (5, "8/31 (월) ~ 9/6 (일)",
     "화장품 전성분표·특허문헌 제형 데이터 수집 착수(타깃 도메인 데이터셋 구성)", 1),
    (6, "9/7 (월) ~ 9/13 (일)",
     "화장품 데이터 수집 마무리, 성분명-CAS 매핑, 화장품 소규모 실측 검증셋(patch test/in vitro) 후보 조사", 1),
    (7, "9/14 (월) ~ 9/20 (일)",
     "EPA Classification 4단계/이분류 동시 산출 라벨링 파이프라인 구축, 클래스 가중치·재샘플링(SMOTE 등) 전략 설계", 2),
    (8, "9/21 (월) ~ 9/27 (일)",
     "전이학습 파이프라인 구축: (a) 사전학습+미세조정, (b) zero-shot, (c) 화장품 전용 소규모 학습 3가지 경로 구현", 2),
    (9, "9/28 (월) ~ 10/4 (일)",
     "Random Forest/XGBoost 기반 4단계·이분류 baseline 모델 학습 및 초기 성능 확인(macro F1, ROC-AUC)", 3),
    (10, "10/5 (월) ~ 10/11 (일)",
     "3가지 전이학습 방식 본실험 및 비교, 화장품 소규모 검증셋 성능 비교", 3),
    (11, "10/12 (월) ~ 10/18 (일)",
     "중간 결과 점검·모델 튜닝, 지도교수 중간 피드백 반영", 3),
    (12, "10/19 (월) ~ 10/25 (일)",
     "ICE mixture-화장품 성분 조성 공간 거리 측정, 적용범위(AD) 정의", 4),
    (13, "10/26 (월) ~ 11/1 (일)",
     "AD 내부/외부 화장품 제형 예측 오차 비교 분석, 모델 최종 보완", 4),
    (14, "11/2 (월) ~ 11/6 (금)",
     "최종 보고서 작성, 학술제 발표자료(포스터/PPT) 제작 및 발표 리허설", 5),
]


def build_schedule_table():
    header = [Paragraph("주차", week_cell), Paragraph("기간", date_cell), Paragraph("주요 활동", task_cell)]
    data = [header]
    row_phase = [None]
    for wk, period, task, phase_idx in WEEKS:
        data.append([
            Paragraph(f"{wk}주차", week_cell),
            Paragraph(period, date_cell),
            Paragraph(task, task_cell),
        ])
        row_phase.append(phase_idx)

    col_widths = [16 * mm, 34 * mm, 122 * mm]
    t = Table(data, colWidths=col_widths, repeatRows=1)

    style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.4, GRID),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]
    # 단계별 옅은 배경색 + 좌측 컬러 바(주차 셀 텍스트 컬러로 대체)
    for i, phase_idx in enumerate(row_phase):
        if phase_idx is None:
            continue
        base = PHASES[phase_idx][2]
        light = colors.Color(base.red, base.green, base.blue, alpha=0.10)
        style.append(('BACKGROUND', (0, i), (-1, i), light))
        style.append(('TEXTCOLOR', (0, i), (0, i), base))
    t.setStyle(TableStyle(style))
    return t


def build_legend():
    cells = []
    for code, label, color in PHASES:
        cells.append(Paragraph(
            f'<font color="{color.hexval()[:7]}">■</font> {label}', legend_style))
    # 2열 x 3행 배치
    rows = [[cells[0], cells[1]], [cells[2], cells[3]], [cells[4], cells[5]]]
    t = Table(rows, colWidths=[86 * mm, 86 * mm])
    t.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    return t


def build_doc(out_path):
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=16 * mm, bottomMargin=14 * mm,
                             leftMargin=16 * mm, rightMargin=16 * mm)
    els = []
    els.append(Paragraph("2026학년도 미래인재대학 학술제 — 주간 연구일정표", title_style))
    els.append(Paragraph("NTP ICE 혼합물 EPA 자극성 분류 데이터 기반 화장품 제형의 피부·안 자극성 예측", subtitle_style))
    els.append(Paragraph("팀 COS · 김서율(팀장) · 장다예, 전수진, 한세음", subtitle_style))
    els.append(Paragraph("연구 기간: 2026.08.03(월) ~ 2026.11.06(금) · 총 14주", date_style))

    els.append(build_legend())
    els.append(Spacer(1, 8))
    els.append(build_schedule_table())
    els.append(Spacer(1, 8))

    els.append(Paragraph(
        "※ 본 일정은 연구계획서 '6. 일정 계획'의 대분류 일정(8/1~11/6)을 8/3(월) 시작 기준 주 단위로 세분화한 것으로, "
        "데이터 수집 난이도 및 중간 결과에 따라 조정될 수 있음.", note_style))
    els.append(Paragraph(
        "※ 특히 화장품 도메인과 ICE mixture 도메인 간 성분군 중복 정도(9주차~13주차 관련)가 전체 일정과 모델 성능에 "
        "가장 큰 영향을 줄 것으로 예상되어, 해당 구간 진행 상황에 따라 앞뒤 주차 활동이 유동적으로 조정될 수 있음.",
        note_style))

    doc.build(els)
    print("OK ->", out_path)


if __name__ == "__main__":
    build_doc(r"C:\Users\UserK\Desktop\학술제\주간연구일정표_20260803_20261106.pdf")
