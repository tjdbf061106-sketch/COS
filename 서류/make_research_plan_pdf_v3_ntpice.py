# -*- coding: utf-8 -*-
"""연구계획서 PDF 생성 (화장품 버전 v3 - NTP ICE 혼합물 데이터 기반 도메인 전이)
MAC(최대허용농도) 전이예측 버전에서, 지도교수님 피드백(MAC은 단일물질 단위로 주로 관리되며
혼합물 단위 MAC 데이터가 대규모로 존재하지 않음)을 반영하여 예측 타깃을
skin/eye irritation GHS 등급으로 전환하고, 학습 소스를 NTP ICE의 mixture/formulation
데이터로 변경한 버전."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold', 'C:/Windows/Fonts/malgunbd.ttf'))

title_style = ParagraphStyle('Title', fontName='MalgunBold', fontSize=17, leading=22,
                              alignment=1, spaceAfter=4)
subtitle_style = ParagraphStyle('Subtitle', fontName='Malgun', fontSize=11, leading=16,
                                 alignment=1, textColor=colors.HexColor('#444444'), spaceAfter=2)
date_style = ParagraphStyle('Date', fontName='Malgun', fontSize=9.5, leading=13,
                             alignment=1, textColor=colors.HexColor('#666666'), spaceAfter=14)
h1_style = ParagraphStyle('H1', fontName='MalgunBold', fontSize=13.5, leading=18,
                           spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#1a3d6d'))
h2_style = ParagraphStyle('H2', fontName='MalgunBold', fontSize=11, leading=15,
                           spaceBefore=8, spaceAfter=4, textColor=colors.HexColor('#2a2a2a'))
body_style = ParagraphStyle('Body', fontName='Malgun', fontSize=9.7, leading=15,
                             spaceAfter=6, alignment=4)
bullet_style = ParagraphStyle('Bullet', fontName='Malgun', fontSize=9.7, leading=15,
                               leftIndent=10, spaceAfter=5, alignment=4)
note_style = ParagraphStyle('Note', fontName='Malgun', fontSize=8.8, leading=13,
                             textColor=colors.HexColor('#666666'), spaceAfter=4)
topic_style = ParagraphStyle('Topic', fontName='MalgunBold', fontSize=11.5, leading=17,
                              spaceAfter=4, textColor=colors.HexColor('#1a3d6d'))
cell_style = ParagraphStyle('Cell', fontName='Malgun', fontSize=9, leading=13.5)
cell_head_style = ParagraphStyle('CellHead', fontName='MalgunBold', fontSize=9.3, leading=13.5,
                                  textColor=colors.white, alignment=1)
sched_head_cell = ParagraphStyle('SchedCell', fontName='MalgunBold', fontSize=9, leading=13, alignment=1)


def build_table(rows, col_widths):
    data = [[Paragraph(r[0], cell_head_style if i == 0 else sched_head_cell if col_widths[0] < 35 * mm else cell_style),
             Paragraph(r[1], cell_head_style if i == 0 else cell_style)]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3d6d')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bbbbbb')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4f6f9')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def build_doc(out_path, sub1, sub2, necessity_sections, purposes,
              topic_kr, topic_en, content_rows, method_sections, schedule_rows, extra_notes):
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=20 * mm, bottomMargin=18 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    els = []
    els.append(Paragraph("연구계획서", title_style))
    els.append(Paragraph(sub1, subtitle_style))
    els.append(Paragraph(sub2, subtitle_style))
    els.append(Paragraph("작성일: 2026년 7월 22일", date_style))

    els.append(Paragraph("1. 연구 필요성", h1_style))
    for heading, paras in necessity_sections:
        els.append(Paragraph(heading, h2_style))
        for p in paras:
            els.append(Paragraph(p, body_style))

    els.append(Paragraph("2. 연구 목적", h1_style))
    for i, p in enumerate(purposes, 1):
        els.append(Paragraph(f"{i}. {p}", bullet_style))

    els.append(Paragraph("3. 연구 주제", h1_style))
    els.append(Paragraph(topic_kr, topic_style))
    els.append(Paragraph(topic_en, note_style))

    els.append(Paragraph("4. 연구 내용", h1_style))
    els.append(build_table(content_rows, [42 * mm, 124 * mm]))

    els.append(Paragraph("5. 연구 방법", h1_style))
    for heading, paras in method_sections:
        els.append(Paragraph(heading, h2_style))
        for p in paras:
            els.append(Paragraph(p, body_style))

    els.append(Paragraph("6. 일정 계획 (2026.08.01 ~ 2026.11.06, 러프 계획)", h1_style))
    els.append(build_table(schedule_rows, [30 * mm, 136 * mm]))
    els.append(Spacer(1, 6))
    for n in extra_notes:
        els.append(Paragraph(n, note_style))

    doc.build(els)
    print("OK ->", out_path)


# =====================================================================
# 화장품 버전 v3 : NTP ICE 혼합물 자극성 데이터 기반 도메인 전이
# =====================================================================
necessity = [
    ("1.1 화장품 안전성 평가 방식의 전환 요구", [
        "화장품 원료 및 제형의 피부·안(眼) 자극성 평가는 전통적으로 동물실험(Draize test)에 의존해왔으나, "
        "EU를 비롯한 주요 시장에서 화장품 원료에 대한 동물실험이 금지되었고, 3R 원칙(Replacement, "
        "Reduction, Refinement)에 따라 in vitro·in silico 대체 평가법에 대한 수요가 커지고 있음.",
    ]),
    ("1.2 예측 타깃 및 데이터 소스 전환 배경", [
        "본 연구는 당초 화학구조 기반으로 성분의 제품범주별 <b>최대허용농도(MAC)</b>를 전이(transfer) "
        "예측하는 방향으로 검토되었으나, 지도교수님 검토 과정에서 다음 두 가지 문제가 확인됨.",
        "첫째, MAC은 IFRA Standards, EU Cosmetics Regulation Annex III, CIR 권고농도 등 대부분의 규제·문헌 "
        "체계에서 <b>단일 성분(single substance) 단위</b>로 산출·관리되며, 다수 성분이 혼합된 <b>제형(mixture) "
        "단위</b>의 MAC 데이터가 대규모로 축적된 공개 데이터베이스는 확인되지 않음.",
        "둘째, 혼합물/제형 단위 독성 데이터가 비교적 풍부하게 모여 있는 공개 데이터베이스인 NTP ICE "
        "(Integrated Chemical Environment)를 직접 조사한 결과, ICE Search 도구 내 \"Mixtures and Formulations "
        "in ICE\" 목록 및 substance type(chemical/mixture) 필드를 통해 혼합물 단위 조회는 가능하지만, "
        "MAC(최대허용농도) 또는 이에 준하는 필드(safe use concentration, margin of safety 등)는 존재하지 않음을 "
        "확인함. 즉 MAC을 혼합물 단위 예측 타깃으로 유지할 경우, 학술제 활동 기간 내내 라벨 데이터 확보에만 "
        "시간을 소모할 위험이 큼.",
        "이에 따라 예측 타깃을 <b>제형(mixture) 단위로 실제 데이터가 존재하는 skin/eye irritation GHS 분류 "
        "등급</b>으로 전환하고, 학습 데이터 소스를 NTP ICE의 mixture/formulation 데이터로 변경함.",
    ]),
    ("1.3 NTP ICE 혼합물 데이터의 특성과 화장품 도메인과의 간극", [
        "ICE Search에서 조회되는 mixture 데이터는 활성성분 비율(Percent AI), 성분별 CAS 및 GHS 분류, 혼합물 "
        "전체에 대한 GHS Classification(예: 특정 질의에서 76개 물질에 대해 150건의 GHS 분류 레코드가 반환된 "
        "사례) 등을 포함함. 다만 ICE에는 <b>화장품만 별도로 모아둔 서브셋은 없으며</b>, 실제 조회되는 "
        "mixture는 살균제·살충제(예: Lannate 계열 살충제)·산업용 방부제·세정제 등 <b>농약/산업용 화학제품이 "
        "주를 이룸</b>. 이는 화장품 제형(계면활성제·보습제·방부제·향료 조합)과 성분군 구성 및 농도 분포가 "
        "다를 수 있음을 의미하며, 학습 도메인(ICE mixture)과 적용 도메인(화장품 제형) 간의 <b>도메인 "
        "갭(domain gap)</b>이 본 연구가 정면으로 다뤄야 할 핵심 난제가 됨.",
    ]),
    ("1.4 완성 제형 단위 예측의 필요성", [
        "화장품·산업용 제형 모두 단일 성분이 아닌 다수 성분의 혼합물이며, 실제 자극성은 성분 간 상호작용"
        "(예: 계면활성제 블렌드의 비선형 효과)에 의해 단순 합산과 다르게 나타나는 경우가 많음. 완성 제형 "
        "단위에서 이를 사전에 스크리닝할 수 있는 예측 모델은 개별 성분 정보만으로 접근하기 어려운 문제를 "
        "보완할 수 있음.",
    ]),
    ("1.5 예측값과 함께 적용범위(applicability domain)를 제시해야 하는 이유", [
        "본 연구는 서로 다른 두 도메인(ICE의 농약/산업용 제형, 화장품 제형) 사이의 전이를 다루므로, 학습에 "
        "사용된 제형들과 화학적 조성이 유사한 화장품 제형에 대해서만 예측을 신뢰할 수 있음. 단일 예측값만 "
        "제시하는 대신 학습 도메인과의 조성 유사도 기반 적용범위(AD)를 함께 제시하면, 전이 예측을 신뢰할 수 "
        "있는 화장품 제형과 여전히 확인시험이 필요한 제형을 구분하는 근거로 활용할 수 있음.",
    ]),
]

purposes = [
    "NTP ICE의 mixture/formulation 단위 skin·eye irritation(Irritation/Corrosion 카테고리) 데이터를 이용하여, "
    "성분 조성(성분명·CAS, 함량 비율, 성분별 GHS 분류)으로부터 제형 전체의 GHS 자극성 등급(무자극/자극/심한 "
    "자극 등)을 예측하는 모델을 개발한다.",
    "ICE mixture 데이터(농약/산업용 화학제품 위주)로 학습한 모델을, 화학적 조성이 다른 화장품 제형에 적용할 "
    "때 발생하는 도메인 갭을 정량화하고, 소규모 화장품 실측·문헌 자극성 데이터를 이용한 미세조정(fine-"
    "tuning) 전이학습 방법론을 설계·비교한다.",
    "GHS/CLP 가산성(additivity) 규칙 기반 baseline과 데이터 기반 모델을 비교하여, 성분 간 비선형·상호작용 "
    "효과를 데이터 기반 모델이 어느 정도 포착하는지 규명한다.",
    "학습 도메인(ICE mixture)과 적용 도메인(화장품 제형) 간 성분 조성 공간의 유사도를 기준으로 적용범위"
    "(applicability domain)를 정의하여, 전이 예측을 신뢰할 수 있는 화장품 제형 영역과 확인시험이 필요한 "
    "영역을 구분하는 기준을 제시한다.",
]

topic_kr = ("“NTP ICE 혼합물 자극성 데이터 기반 화장품 제형의 피부·안 자극성 예측: 도메인 전이 및 "
            "적용범위 정량화”")
topic_en = ("(영문 부제 예시: Transferring Mixture-Level Skin/Eye Irritation Models from NTP ICE to Cosmetic "
            "Formulations: A Domain Adaptation Approach with Applicability Domain Quantification)")

content_rows = [
    ["구성요소", "내용"],
    ["① NTP ICE 혼합물 데이터 수집", "ICE Search에서 substance type=mixture 필터와 Irritation/Corrosion "
     "카테고리를 이용해 skin/eye irritation 관련 mixture 레코드(GHS Classification, Percent AI 등)를 수집. "
     "실제 확보 가능한 건수는 8월 초 예비조사를 통해 확정."],
    ["② 성분 조성 파싱 및 정규화", "각 mixture의 성분별 CAS·GHS 분류·활성성분 비율(Percent AI)을 추출, "
     "성분명 표기 이형 정규화 및 중복·오류 검증"],
    ["③ 화장품 제형 데이터 수집 (전이 적용 대상)", "공개 화장품 전성분표, 특허문헌 제형 예시 등에서 성분 "
     "조성(성분명+함량 추정치)을 수집. 학습용이 아닌 모델을 적용·검증할 타깃 도메인 데이터셋으로 구성"],
    ["④ 화장품 소규모 실측 검증셋 구축", "문헌에 보고된 화장품/계면활성제 블렌드의 실측 patch test·in "
     "vitro(EpiSkin, EpiOcular 등) 결과를 최소 규모로 수집하여 전이 성능 평가용 held-out 검증셋으로 구성"],
    ["⑤ GHS 가산성 규칙(baseline) 구현", "CLP/GHS Annex I의 skin/eye irritation 혼합물 분류 가산 공식을 "
     "코드로 구현하여 규칙 기반 baseline 라벨 및 예측치 생성"],
    ["⑥ 전이학습 모델 개발", "(a) ICE mixture 데이터로 사전학습 후 화장품 소규모 데이터로 미세조정, (b) "
     "미세조정 없는 zero-shot 전이, (c) 화장품 데이터만으로 학습한 소규모 모델을 비교"],
    ["⑦ 도메인 갭 및 적용범위(AD) 평가", "ICE mixture 도메인과 화장품 도메인 간 성분 조성 공간의 거리를 "
     "측정하고, AD 내부/외부 화장품 제형에 대한 전이 예측 성능 차이를 분석"],
    ["⑧ 결과 정리 및 발표 준비", "학술제 제출용 초록, 포스터/발표자료, 최종 보고서 작성"],
]

method_sections = [
    ("5.1 NTP ICE 데이터 수집 및 전처리", [
        "ICE Search(ice.ntp.niehs.nih.gov)에서 substance type 필터를 mixture로 설정하고 Irritation/Corrosion "
        "카테고리를 조회하여 skin irritation·eye irritation 관련 레코드와 GHS Classification, 성분별 Percent "
        "AI를 수집. 성분명은 CAS 기반으로 정규화하고, 화장품 도메인과 겹치는 성분(계면활성제, 방부제 등)이 "
        "있는지 별도로 확인.",
    ]),
    ("5.2 화장품 제형(타깃 도메인) 데이터 수집", [
        "공개 화장품 전성분표, 특허문헌 제형 예시에서 성분명+함량 추정치를 수집. 학습에 직접 사용하지 않고 "
        "모델의 적용 대상 및 미세조정용 소규모 데이터로만 활용.",
    ]),
    ("5.3 전이학습 방법론", [
        "사전학습-미세조정: ICE mixture 데이터로 성분 조성 → GHS 자극성 등급 관계를 사전학습한 후, 화장품 "
        "소규모 데이터로 미세조정. Zero-shot 전이: 미세조정 없이 ICE 사전학습 모델을 화장품 제형에 직접 "
        "적용. 세 방식(zero-shot, 미세조정, 화장품 전용 소규모 학습)의 성능을 비교하여 도메인 전이의 효과와 "
        "한계를 규명.",
    ]),
    ("5.4 GHS 가산성 규칙(baseline) 및 모델 비교", [
        "GHS/CLP Annex I 가산성 공식을 Python으로 구현하여 규칙 기반 baseline을 산출하고, Random "
        "Forest/XGBoost 등 트리 기반 모델 및 상호작용 항을 반영한 개선 모델과 비교. 평가지표: 분류 정확도, "
        "F1-score, baseline과의 성능 차이(Δ).",
    ]),
    ("5.5 도메인 갭 및 적용범위(Applicability Domain) 설정", [
        "ICE mixture 도메인과 화장품 도메인의 성분 조성 벡터 공간 내 거리(예: 성분군 분포, Tanimoto 유사도 "
        "등)를 측정하여 AD를 정의하고, AD 내부/외부 화장품 제형에 대한 예측 오차 차이를 비교 분석.",
    ]),
    ("5.6 연구 수행 체계 및 사용 도구", [
        "연구팀 내 역할을 ICE 데이터 수집·정제, 화장품 데이터 수집, 모델링, 도메인 갭/AD 검증 등으로 분담. "
        "Python(pandas, scikit-learn, XGBoost), 문헌 조사(RISS, PubMed 등) 활용.",
    ]),
]

schedule_rows = [
    ["기간", "주요 활동"],
    ["8/1 ~ 8/15", "주제 확정, ICE Search 예비조사(mixture·Irritation/Corrosion 카테고리 실제 데이터 규모 "
     "확인), 관련 문헌(GHS/CLP 가산성, 도메인 적응/전이학습 선행연구) 조사, 역할 분담"],
    ["8/16 ~ 9/15", "ICE mixture 데이터 수집·정제, 화장품 제형(전성분표) 데이터 수집, 성분명-CAS 매핑"],
    ["9/16 ~ 9/30", "GHS 가산성 규칙(baseline) 구현, 전이학습 파이프라인(사전학습+미세조정, zero-shot) 구축"],
    ["10/1 ~ 10/20", "모델 개발 및 초기 실험, 화장품 소규모 검증셋을 통한 성능 비교"],
    ["10/21 ~ 10/31", "도메인 갭 분석, AD 내/외 오류 분석, 모델 보완"],
    ["11/1 ~ 11/6", "학술제 발표자료(포스터/PPT) 및 최종 보고서 작성, 발표 리허설"],
]

extra_notes = [
    "※ 위 일정은 데이터 수집 난이도 및 중간 결과에 따라 조정 가능한 러프한 계획임 (특히 ICE mixture 데이터의 "
    "실제 규모 확인 및 화장품 도메인과의 성분군 중복 정도가 전체 일정과 모델 성능에 가장 큰 영향을 줄 것으로 "
    "예상됨).",
    "※ 본 연구계획서는 기존 \"화학구조 기반 MAC 전이예측\" 버전을 지도교수님 피드백에 따라 수정한 것임. "
    "MAC은 성분 단위로 관리되는 경우가 많아 혼합물 단위 대규모 라벨 데이터가 부재하다는 점, 그리고 혼합물 "
    "단위 독성 데이터가 모여 있는 NTP ICE에도 MAC에 해당하는 필드가 없다는 점을 확인하여, 예측 타깃을 "
    "실제 데이터가 존재하는 GHS 자극성 등급으로 전환함.",
]

build_doc(
    out_path=r"C:\Users\UserK\Desktop\formulation_audit_extracted\role_c_outputs\연구계획서_화장품버전_NTPICE전이_20260722.pdf",
    sub1="NTP ICE 혼합물 자극성 데이터 기반 화장품 제형의 피부·안 자극성 예측",
    sub2="— 도메인 전이(Domain Transfer) 및 적용범위(Applicability Domain) 정량화 —",
    necessity_sections=necessity,
    purposes=purposes,
    topic_kr=topic_kr,
    topic_en=topic_en,
    content_rows=content_rows,
    method_sections=method_sections,
    schedule_rows=schedule_rows,
    extra_notes=extra_notes,
)
