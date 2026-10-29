# -*- coding: utf-8 -*-
"""장다예(B) 보강데이터 조사 검증 리포트 PDF - 원본 파일 직접 재현 검증 + 실사용 가능성 검수"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 HRFlowable, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold', 'C:/Windows/Fonts/malgunbd.ttf'))

NAVY = colors.HexColor('#1a3d6d')
GREEN = colors.HexColor('#2a6f3d')
RED = colors.HexColor('#a02020')
GRAY = colors.HexColor('#777777')
LIGHT_GREEN_BG = colors.HexColor('#eaf5ec')
LIGHT_RED_BG = colors.HexColor('#fbeaea')
LIGHT_GRAY_BG = colors.HexColor('#f2f2f2')

title_style = ParagraphStyle('Title', fontName='MalgunBold', fontSize=17, leading=22,
                              textColor=NAVY, spaceAfter=6)
subtitle_style = ParagraphStyle('Subtitle', fontName='Malgun', fontSize=10, leading=15,
                                 textColor=colors.HexColor('#444444'), spaceAfter=14)
kicker_style = ParagraphStyle('Kicker', fontName='Malgun', fontSize=9.5,
                               textColor=GRAY, spaceAfter=4)
h1_style = ParagraphStyle('H1', fontName='MalgunBold', fontSize=13, leading=18,
                           spaceBefore=16, spaceAfter=8, textColor=NAVY)
h2_ok_style = ParagraphStyle('H2ok', fontName='MalgunBold', fontSize=11, leading=16,
                              spaceBefore=10, spaceAfter=5, textColor=GREEN)
h2_bad_style = ParagraphStyle('H2bad', fontName='MalgunBold', fontSize=11, leading=16,
                               spaceBefore=10, spaceAfter=5, textColor=RED)
body_style = ParagraphStyle('Body', fontName='Malgun', fontSize=9.7, leading=15,
                             spaceAfter=6, alignment=4)
body_bold_style = ParagraphStyle('BodyBold', fontName='MalgunBold', fontSize=9.7, leading=15,
                                  spaceAfter=6, alignment=4)
note_style = ParagraphStyle('Note', fontName='Malgun', fontSize=8.5, leading=12.5,
                             textColor=GRAY, spaceAfter=4)
cell_style = ParagraphStyle('Cell', fontName='Malgun', fontSize=8.5, leading=12)
cell_hdr_style = ParagraphStyle('CellHdr', fontName='MalgunBold', fontSize=8.7, leading=12,
                                 textColor=colors.white)
tag_style = ParagraphStyle('Tag', fontName='MalgunBold', fontSize=9, leading=13,
                            textColor=colors.white, alignment=1)


def tag_table(text, bg_color):
    t = Table([[Paragraph(text, tag_style)]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def verdict_box(label, bg_color, border_color, paras):
    els = [Table([[Paragraph(label, ParagraphStyle('lbl', fontName='MalgunBold', fontSize=9.5,
                                                     textColor=border_color))]],
                  colWidths=[170 * mm],
                  style=TableStyle([
                      ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                      ('TOPPADDING', (0, 0), (-1, -1), 5),
                      ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                      ('LEFTPADDING', (0, 0), (-1, -1), 8),
                  ]))]
    body_rows = [[Paragraph(p, body_style)] for p in paras]
    body_tbl = Table(body_rows, colWidths=[170 * mm])
    body_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    outer = Table([[els[0]], [body_tbl]], colWidths=[170 * mm])
    outer.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.8, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    return outer


def data_table(header, rows, col_widths):
    data = [[Paragraph(h, cell_hdr_style) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), cell_style) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(('BACKGROUND', (0, i), (-1, i), LIGHT_GRAY_BG))
    t.setStyle(TableStyle(style))
    return t


OUT_PATH = r"C:\Users\UserK\Desktop\학술제\세팅\장다예_보강데이터_검증리포트.pdf"

doc = SimpleDocTemplate(OUT_PATH, pagesize=A4,
                         topMargin=18 * mm, bottomMargin=16 * mm,
                         leftMargin=16 * mm, rightMargin=16 * mm)
els = []

# ── 헤더 ──
els.append(Paragraph("보강데이터 조사 검증 리포트", kicker_style))
els.append(Paragraph("장다예(B) 1주차 보강데이터 조사 결과 — 원본파일 재현 검증 + 실사용성 검수", title_style))
els.append(Paragraph(
    "검증 대상: CASRN 결측 12건 원인분석, ChemSkin DB 2.0 / XGB(Kang 2023) 보강데이터 조사 결과 "
    "· 검증 방법: 원본 xlsx/pdf/docx 파일을 직접 열어 보고된 수치·결론을 재현 · 검증일: 2026-08-13",
    subtitle_style))
els.append(tag_table("① 1주차 필터링/CASRN 결측 확인 단계 검증", GREEN))
els.append(Spacer(1, 10))

# ── 섹션 1: 검증 결과 총괄 ──
els.append(Paragraph("1. 검증 결과 총괄", h1_style))
els.append(data_table(
    header=["항목", "B의 보고 내용", "검증 결과"],
    rows=[
        ["CASRN 결측 건수/위치", "1,799건 중 12건, MIX82/380/86/78/514/382 등 6개 Formulation", "[일치] 원본 xlsx 직접 필터링 재현"],
        ["ChemSkin DB2.0 규모", "Table 3, 234개 참조 화학물질", "[일치] 논문 abstract·본문 4곳에서 234 확인"],
        ["ChemSkin DB2.0 겹침(mixture)", "skin_filtered CASRN 270종과 6종만 겹침", "[부분확인] PDF Table3 회전텍스트로 155/234종만 추출됨 — 전체 234종 기준 재검증 필요"],
        ["XGB(Kang2023) 규모/분포", "545개, CAS 500종, Cat1 160/Cat2 229/Cat3 95/NC 61", "[일치] 원본 xlsx 직접 로드 재현"],
        ["XGB ↔ skin_filtered 겹침", "CASRN 7종 겹침, 확인 필요 전제조건 2가지 제시", "[일치] 겹침 7종·등급차이 사례 3건 모두 재현됨"],
        ["GHS classification(내부)", "n=79, Chemical 단위, 보조피처로만 활용 가능", "[일치]"],
        ["CASRN 12건 내부 join 복구", "\"원본 파일 내부 join만으로 12건 전부 복구 가능\"", "[오류] 아래 2번 상세"],
        ["XGB 537건 순수 추가 시뮬레이션", "\"단위 플래그만 추가하면 즉시 병합 가능한 수준\"", "[오류] 아래 2번 상세"],
    ],
    col_widths=[35 * mm, 75 * mm, 60 * mm],
))
els.append(Spacer(1, 4))
els.append(Paragraph(
    "※ 위 표의 [일치]/[오류] 판정은 CASRN 필터링 재현, PDF 텍스트 추출, xlsx 직접 로드를 통해 원본 파일에서 다시 계산한 결과이며, "
    "B가 채팅으로 공유한 수치·결론과 대조한 것이다.", note_style))

# ── 섹션 2: 오류 상세 ──
els.append(Paragraph("2. 정정이 필요한 결론 (2건)", h1_style))

els.append(verdict_box(
    "오류 ① — \"CASRN 결측 12건, 원본 파일 내부 join만으로 전부 복구 가능\"",
    LIGHT_RED_BG, RED,
    [
        "실제 확인 결과: 결측 12건은 Record_ID가 여러 행으로 나뉘어 정보가 흩어진 문제가 아니라, "
        "<b>\"Captan-related derivatives\", \"Dichlorvos-related compounds\", \"Atrazine-related "
        "compounds\" 등 미확인 잔류/분해산물 카테고리</b>다. 데이터셋 전체(skin_irritation.xlsx, "
        "전체 2,058행)를 뒤져봐도 이 이름을 가진 성분이 CASRN을 가진 적이 단 한 번도 없다.",
        "즉 같은 Formulation_ID(예: MIX82)에 Captan(CASRN 133-06-2, 함량 47.5%)이라는 확인된 성분이 "
        "같이 존재하긴 하지만, 이는 \"주성분\"이고 \"related derivatives\"는 그 분해산물/불순물로 보이는 "
        "별개 행이라 서로 다른 CASRN을 가져야 하는 항목이다. 내부 join으로 메울 수 있는 결측이 아니다.",
        "→ <b>결론: 외부 보강 없이는 복구 불가.</b> 3주차 라벨링 단계에서 처리방침 결정 필요 "
        "(제외 / \"unidentified residue\" 플래그 부여 / 주성분 CASRN으로 대체 후 근사치 처리 중 택1).",
    ]
))
els.append(Spacer(1, 8))

els.append(verdict_box(
    "오류 ② — \"XGB 537건 순수 추가 시, 단위 플래그만 추가하면 즉시 병합 가능\"",
    LIGHT_RED_BG, RED,
    [
        "실제 확인 결과: XGB(Kang2023) 원본 파일(545행) 자체에 <b>CASRN 중복이 88행</b> 존재하고, "
        "그중 <b>43건은 서로 다른 Category로 상충</b>된 상태로 들어있다.",
        "예시: Geraniol(CASRN 106-24-1)이 한 행은 Cat 2, 다른 행은 Cat 3로 등재. 1,2-epoxybutane"
        "(106-88-7)은 Cat 1과 Cat 2로 상충. Methyl formate(107-31-3)은 Cat 2와 Cat 3로 상충 등.",
        "이 상태로 겹침 7종만 제외하고 나머지 537종(고유 CASRN 기준 493종)을 그대로 합치면, 상충된 "
        "라벨이 노이즈로 섞여 들어가 B가 제시한 \"Category 1: 71→227건, Category 2: 35→263건\" 같은 "
        "보강 시뮬레이션 수치가 실제보다 부풀려진다.",
        "부가로 CASRN 컬럼 자체에 엑셀 자동 날짜변환으로 깨진 값이 5건 있다 "
        "(예: Propionic acid 원래 CAS 79-09-4 → \"1979-09-04 00:00:00\"으로 저장됨). "
        "문자열 그대로 매칭하면 이 5건이 전부 누락된다.",
        "→ <b>결론: 병합 전 (1) CASRN 문자열 강제 캐스팅+5건 수동 복구, (2) CASRN 중복 88건 중 "
        "상충 43건 해소(다수결/최신문헌 우선 등 규칙 필요), (3) Physical state 대소문자 정규화(liquid/"
        "Liquid) 세 가지 전처리를 마친 뒤에야 실제 순증분을 다시 계산해야 한다.</b>",
    ]
))

# ── 섹션 3: 실사용 가능성 검수 ──
els.append(Spacer(1, 4))
els.append(tag_table("② 외부 보강데이터 3종 실사용 가능성 검수", NAVY))
els.append(Paragraph("3. 외부 데이터 소스별 실사용 가능성 검수", h1_style))
els.append(Paragraph(
    "B가 확보한 파일 3개(ChemSkin DB2.0 PDF, XGB 논문 supplementary xlsx/docx)를 직접 열어 구조·형식·"
    "결측·중복 관점에서 파이프라인 투입 가능 여부를 확인했다.", body_style))

els.append(data_table(
    header=["파일", "구조 확인 결과", "실사용 판정"],
    rows=[
        ["43188_2022_168_MOESM2_ESM.xlsx\n(XGB descriptor 데이터)",
         "545행×39열, 물리화학 descriptor 34개 전부 결측치 0건. 단 CASRN 컬럼에 datetime 오염 5건, "
         "CASRN 중복 88행(상충 43건), Physical state 대소문자 불일치.",
         "[부분가능] 전처리 후 사용 가능\n(캐스팅+dedupe+정규화 필요)"],
        ["43188_2022_168_MOESM1_ESM.docx\n(descriptor 설명문서)",
         "Table 1에 34개 descriptor 이름·정의 전부 문서화, 원 논문 개수와 1:1 일치. 결측/오류 없음.",
         "[가능] 즉시 사용 가능"],
        ["s00204-026-04406-y.pdf\n(ChemSkin DB2.0 논문)",
         "본문 텍스트는 정상 추출되나, 핵심 Table 3(234종 CAS 리스트)이 PDF 내부에서 문자 단위로 "
         "회전 렌더링(upright=False)되어 있어 표준 텍스트 추출 시 234종 중 155종(66%)만 인식되고 "
         "나머지는 레이아웃 파싱 실패.",
         "[불가] 구조적으로 즉시 사용 불가\n(raw data 재확보 필요)"],
    ],
    col_widths=[52 * mm, 88 * mm, 30 * mm],
))
els.append(Spacer(1, 6))
els.append(Paragraph(
    "※ ChemSkin DB2.0의 \"6종 겹침\" 결론은 155종(추출 성공분) 기준으로 계산됐을 가능성이 있어, "
    "전체 234종 기준 재검증이 필요하다. 234종 전체 CAS 리스트를 신뢰성 있게 확보하려면 PDF 뷰어로 "
    "수동 대조하거나 저자(Kyung-Min Lim, 이화여대)에게 raw data(csv/xlsx) 제공을 요청하는 방법을 "
    "권장한다.", note_style))

els.append(Spacer(1, 10))
els.append(HRFlowable(width="100%", color=colors.HexColor('#dddddd')))
els.append(Spacer(1, 6))

# ── 섹션 4: 다음 액션 ──
els.append(Paragraph("4. 다음 액션 제안", h1_style))
for txt in [
    "CASRN 12건: 처리방침(제외/플래그/근사치) 결정 — 3주차 라벨링 단계에서 MIX433 이상케이스와 함께 논의",
    "XGB xlsx: CASRN 문자열 캐스팅 + 중복 88건 dedupe/상충해소 + Physical state 정규화 스크립트 작성 → 순증분 재계산",
    "ChemSkin DB2.0: 234종 전체 CAS 리스트 확보(PDF 수동 대조 또는 저자 문의) 후 겹침 재검증",
]:
    els.append(Paragraph(f"□ {txt}", body_style))

els.append(Spacer(1, 8))
els.append(HRFlowable(width="100%", color=colors.HexColor('#dddddd')))
els.append(Spacer(1, 4))
els.append(Paragraph(
    "본 리포트는 skin_irritation.xlsx(Data_invivo), 43188_2022_168_MOESM1/2_ESM(docx/xlsx), "
    "s00204-026-04406-y.pdf 원본 파일을 pandas/pdfplumber/python-docx로 직접 재현하여 작성했다.",
    note_style))

doc.build(els)
print("OK ->", OUT_PATH)
