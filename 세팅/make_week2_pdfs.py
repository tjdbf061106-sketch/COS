# -*- coding: utf-8 -*-
"""Skin/Eye 팀 2주차 준비용 PDF(각 2페이지) 생성 스크립트."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('Malgun', r'C:\Windows\Fonts\malgun.ttf'))
pdfmetrics.registerFont(TTFont('Malgun-Bold', r'C:\Windows\Fonts\malgunbd.ttf'))

BLUE = colors.HexColor('#1F4E79')
GRAY = colors.HexColor('#595959')
LIGHT = colors.HexColor('#EAF1F8')
WARN = colors.HexColor('#C0392B')

styles = {
    'title': ParagraphStyle('title', fontName='Malgun-Bold', fontSize=16, textColor=BLUE,
                             spaceAfter=2*mm, leading=20),
    'subtitle': ParagraphStyle('subtitle', fontName='Malgun', fontSize=9, textColor=GRAY,
                                spaceAfter=5*mm, leading=12),
    'h2': ParagraphStyle('h2', fontName='Malgun-Bold', fontSize=12, textColor=BLUE,
                          spaceBefore=4*mm, spaceAfter=2*mm, leading=15),
    'h3': ParagraphStyle('h3', fontName='Malgun-Bold', fontSize=10, textColor=colors.black,
                          spaceBefore=2*mm, spaceAfter=1*mm, leading=13),
    'body': ParagraphStyle('body', fontName='Malgun', fontSize=9, leading=13,
                            spaceAfter=1.5*mm, textColor=colors.black),
    'bodysmall': ParagraphStyle('bodysmall', fontName='Malgun', fontSize=8.3, leading=11.5,
                                 spaceAfter=1*mm, textColor=colors.black),
    'warn': ParagraphStyle('warn', fontName='Malgun-Bold', fontSize=9, leading=13,
                            textColor=WARN, spaceAfter=1.5*mm),
    'cell': ParagraphStyle('cell', fontName='Malgun', fontSize=8, leading=11),
    'cellhead': ParagraphStyle('cellhead', fontName='Malgun-Bold', fontSize=8.3, leading=11,
                                textColor=colors.white),
}


def bullets(items, style='bodysmall'):
    return ListFlowable(
        [ListItem(Paragraph(t, styles[style]), leftIndent=4*mm, value='•') for t in items],
        bulletType='bullet', leftIndent=2*mm, spaceBefore=0, spaceAfter=2*mm,
    )


def make_table(data, col_widths, head_bg=BLUE):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), head_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BFBFBF')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def build_skin_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                             topMargin=15*mm, bottomMargin=13*mm,
                             leftMargin=16*mm, rightMargin=16*mm)
    F = []
    F.append(Paragraph('[Skin Irritation] 1주차 정리 &amp; 2주차 준비', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-08-18', styles['subtitle']))

    F.append(Paragraph('1. 1주차 한 일', styles['h2']))

    F.append(Paragraph('① 필터링 — skin_irritation.xlsx → skin_filtered.csv', styles['h3']))
    F.append(bullets([
        '기준: <b>Mixture=="Mixture"</b> &amp; <b>Endpoint=="EPA classification"</b> → <b>1,799건</b>, Formulation_ID 고유 544개',
        'Response는 "Category 1~4" 텍스트가 아니라 <b>숫자 1~4</b>(1=부식성/불가역손상 ~ 4=무자극 추정)',
        '클래스 분포: 4=1,434 / 3=259 / 1=71 / 2=35 — <b>4가 약 80%, 극심한 클래스 불균형</b> → 3주차 SMOTE/class_weight 방향 사전 결정 필요',
    ]))

    F.append(Paragraph('② 결측치 — CASRN 12건 (전체의 0.67%)', styles['h3']))
    F.append(bullets([
        '발생 Formulation 6개: MIX82, MIX380, MIX86, MIX78, MIX514, MIX382 (Dichlorvos/Captan/Atrazine 계열)',
        '(B 보고) 같은 Record_ID가 성분별로 여러 행으로 나뉘는 구조 때문 → 원본 내부 join으로 복구 가능하다고 판단',
    ]))

    F.append(Paragraph('③ skin 보강데이터(XGB.xlsx, Kang2023) 실사용성 확인', styles['h3']))
    F.append(bullets([
        '545행×39컬럼, descriptor 34개 결측 0건 — descriptor 자체는 바로 사용 가능',
        'CASRN 5건이 엑셀에서 날짜로 자동변환됨, 중복 88행 중 43건 라벨 상충(예: Geraniol이 Cat2/Cat3 둘 다), Physical state 대소문자 불일치',
        '→ 캐스팅+dedupe+정규화 스크립트 작성 전에는 사용 불가',
        'descriptor.pdf(34개 정의표)는 원문과 1:1 일치, 별도 정제 없이 바로 활용 가능',
    ]))

    F.append(Paragraph('2. 2주차 들어가기 전 확인 필요', styles['h2']))
    F.append(Paragraph('직접 데이터를 열어 재확인한 결과, 1주차 결론 중 2곳은 그대로 다음 단계로 넘기면 위험합니다.',
                        styles['bodysmall']))

    F.append(Paragraph('⚠ (1) CASRN 12건 = "결측 복구"가 아니라 "정책 결정" 문제', styles['warn']))
    F.append(bullets([
        '12건 전부 Chemical_Name이 "Captan-related derivatives", "Dichlorvos-related compounds" 형태 — 활성성분의 잔여/불특정 부산물이라 원래 CAS가 없는 성분',
        '같은 Record_ID의 다른 행(예: Captan 133-06-2)의 CAS를 그대로 붙이면 <b>서로 다른 화학물질에 동일 CAS를 부여하는 오류</b>가 됨 → "join으로 복구"는 틀린 접근',
        '다행히 제형당 비중은 0.03%~6%로 작음 → (a)제외 (b)"unknown residue" 더미성분 포함 중 하나로 정책만 정하면 됨',
    ]))

    F.append(Paragraph('⚠ (2) 조성 중복 11그룹(22건), 그중 1건 라벨 상충', styles['warn']))
    F.append(bullets([
        'CASRN+비율이 완전히 동일한데 Formulation_ID만 다른 그룹이 11개 존재 — 이대로 벡터화하면 train/test 분리 시 같은 조성이 양쪽에 걸쳐 성능이 부풀려질 위험(data leakage)',
        '11그룹 중 1그룹은 <b>동일 조성인데 Response(라벨)까지 다름</b> — 원인 규명 필요',
    ]))

    F.append(Paragraph('MIX433 처리방침 관련: Response 2/3/4 혼재 — 다수결/최고위험도/제외 중 선택, pivot에 라벨 join 전에 필요',
                        styles['bodysmall']))

    F.append(PageBreak())

    F.append(Paragraph('3. 팀 간(Skin↔Eye) 크로스체킹 항목', styles['h2']))
    F.append(bullets([
        '<b>잔여/미확인 성분 처리 원칙 통일</b> — eye팀도 동일 유형 문제(CASRN 결측 8건) 있음, 같은 규칙으로',
        '<b>중복조성 제형 처리 원칙 통일</b> — skin 11그룹(22건) vs eye 30그룹(62건), 같은 기준으로 결정',
        '<b>라벨 스케일 비교</b> — skin Response(1~4 정수) vs eye epa_class(1.0~4.0 실수), 이분류(1~2 vs 3~4) 매핑 함수를 공동으로 하나만 정의',
        '<b>스키마 스타일 차이</b> — skin은 원본 컬럼(PascalCase) 그대로, eye는 가공된 snake_case+플래그 컬럼 구조 → ③단계 합칠 때 통일 방식 논의',
    ], style='body'))

    F.append(Paragraph('4. 참고: 확인된 수치 요약', styles['h2']))
    ref_rows = [
        [Paragraph('항목', styles['cellhead']), Paragraph('값', styles['cellhead'])],
        ['필터링 결과', '1,799건 / Formulation_ID 544개'],
        ['클래스 분포', '4=1,434, 3=259, 1=71, 2=35'],
        ['CASRN 결측', '12건(0.67%) — 6개 Formulation'],
        ['단일성분 제형 비율', '334/544 (약 61%)'],
        ['조성중복 그룹(라벨상충 포함)', '11그룹 / 22건 (그중 1그룹 라벨 상충)'],
    ]
    ref_rows_p = [[Paragraph(str(c), styles['cell']) if not isinstance(c, Paragraph) else c for c in r] for r in ref_rows]
    F.append(make_table(ref_rows_p, [55*mm, 93*mm], head_bg=GRAY))

    F.append(Spacer(1, 4*mm))
    F.append(Paragraph('2주차 할 일 추천', styles['h2']))
    rows = [
        [Paragraph('순서', styles['cellhead']), Paragraph('작업', styles['cellhead']), Paragraph('비고', styles['cellhead'])],
        ['0.', 'CASRN 12건 정책 재확인 + 중복조성 11그룹 라벨상충 원인 확인',
         Paragraph('벡터화 전 <b>필수 선행</b>', styles['cell'])],
        ['1.', 'MIX433 처리방침 결정', '3주차로 미루지 않기'],
        ['2.', '"~-related" 12건 제외 vs unknown 더미 포함 결정', '팀 합의 필요'],
        ['3.', '11그룹 합치기 / 유지+split시 그룹단위 분리 결정', 'leakage 방지'],
        ['4.', '위 1~3 반영 후 CASRN 표기 통일(CAS매핑 확정)', ''],
        ['5.', 'Formulation_ID × CASRN pivot table(벡터화)', '원 계획대로'],
        ['6.', '단일성분 A/B 분리 — 단일성분 비율 334/544(약 61%)', '원 계획대로'],
        ['(여유시)', 'XGB.xlsx 전처리 스크립트(CAS캐스팅+dedupe) 착수', '급하지 않음'],
    ]
    rows_p = [[Paragraph(str(c), styles['cell']) if not isinstance(c, Paragraph) else c for c in r] for r in rows]
    F.append(make_table(rows_p, [18*mm, 90*mm, 40*mm]))

    doc.build(F)


def build_eye_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                             topMargin=15*mm, bottomMargin=13*mm,
                             leftMargin=16*mm, rightMargin=16*mm)
    F = []
    F.append(Paragraph('[Eye Irritation] 1주차 정리 &amp; 2주차 준비', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-08-18', styles['subtitle']))

    F.append(Paragraph('1. 1주차 한 일', styles['h2']))

    F.append(Paragraph('① 데이터 분류 — 원본 3,028행을 시험단위/평가기준별로 분리', styles['h3']))
    F.append(bullets([
        '혼합물(완성 제형) 613개 — 본 연구의 학습 대상',
        '순물질 EPA 192행 / GHS 179행(고유물질 173종) — 학습셋 미포함, 별도 보관',
        'in vitro(Vitrigel) 118종, Draize 원점수 118종 — 시험법이 달라 별도 보관',
        '혼합물은 GHS Response가 전부 결측 → <b>라벨은 EPA Classification만 사용</b>(skin과 동일 결정)',
    ]))

    F.append(Paragraph('② 데이터 보완 및 정제', styles['h3']))
    F.append(bullets([
        '(a) SMILES 결측 보충: 단일화합물 4종(Abamectin, Spinosad, Spinetoram, Emamectin benzoate)을 PubChem에서 확보 → 성분 행 24개 복구',
        '(b) 함량 커버리지: FIFRA는 활성성분만 공개 → 제외보다 변수로 활용하는 방향 검토 중(미결)',
        '(c) 동일조성 상충라벨 8쌍(16개) → 위험성 높은 쪽 8개만 유지, 8개 제외',
        '→ 보고된 최종 학습셋: <b>제형 579개</b>',
    ]))

    F.append(Paragraph('③ eye 보강데이터 후보 4개 확인', styles['h3']))
    F.append(bullets([
        'DRD_Barroso2017_extracted.xlsx(681행): 순물질 562종 GHS+동물반응개체수 — 발표자료 가치 높음, 학습셋 직접 사용 불가',
        'DRD_usable.csv(334행): 정제판, 우리 활성성분과 겹치는 건 20종뿐 — 성분 라벨 보강용',
        'ICE_Eye_Irritation_CAP.xlsx(1,996행): 제형 608개가 우리 613개의 부분집합 → 데이터 증가 없음, ID 매칭 금지(제형명 매칭)',
        'eye_invitro_vitrigel.csv(118행): 시험법 다름 → 학습셋에 섞지 말 것, lag_time 180초는 절단값(cutoff)',
    ], style='bodysmall'))

    F.append(Paragraph('2. 2주차 들어가기 전 확인 필요', styles['h2']))
    F.append(Paragraph('실제 eye_filtered.csv를 열어 재확인한 결과, 보고 수치와 파일 수치가 어긋납니다.',
                        styles['bodysmall']))

    F.append(Paragraph('⚠ (1) "613 → 579" 숫자가 파일과 안 맞음', styles['warn']))
    F.append(bullets([
        '실제 eye_filtered.csv의 mixture 고유 formulation_id는 <b>608개</b> — 보고된 613개와 5개 차이',
        '8쌍 상충 제거(8개 제외)를 재계산해도 정확히 맞지만, 608-8=600이라 최종 보고치 579와는 <b>21개 추가 차이</b>',
        '→ 613→608→579 각 단계에서 몇 건이 왜 빠졌는지 <b>로그로 재추적 필요</b>, 안 하면 이후 단계 전부 재작업 위험',
    ]))

    F.append(Paragraph('⚠ (2) 조성 완전동일 그룹 30개(62개 제형) — skin보다 큼', styles['warn']))
    F.append(bullets([
        'CASRN+비율까지 완전히 같은데 formulation_id만 다른 그룹이 30개 확인됨(skin은 11그룹/22건)',
        'skin과 동일 원칙(합치기/유지+split시 그룹단위 분리)으로 처리 필요, 규모가 더 커서 영향 큼',
    ]))

    F.append(Paragraph('⚠ (3) CASRN 결측 있는 mixture 제형 8개 — skin의 "~-related" 패턴과 같은지 미확인',
                        styles['warn']))

    F.append(PageBreak())

    F.append(Paragraph('3. 팀 간(Skin↔Eye) 크로스체킹 항목', styles['h2']))
    F.append(bullets([
        '<b>잔여/미확인 성분 처리 원칙 통일</b> — skin팀도 동일 유형 문제("~-related" 12건) 있음, 같은 규칙으로',
        '<b>중복조성 제형 처리 원칙 통일</b> — eye 30그룹(62건) vs skin 11그룹(22건), 같은 기준으로 결정',
        '<b>라벨 스케일 비교</b> — eye epa_class(1.0~4.0 실수) vs skin Response(1~4 정수), 이분류(1~2 vs 3~4) 매핑 함수를 공동으로 하나만 정의',
        '<b>스키마 스타일 차이</b> — eye는 가공된 snake_case+플래그 컬럼(has_casrn, needs_structure_lookup 등) 구조, skin은 원본 컬럼(PascalCase) 그대로 → ③단계 합칠 때 통일 방식 논의',
    ], style='body'))

    F.append(Paragraph('4. 참고: 확인된 수치 요약', styles['h2']))
    ref_rows = [
        [Paragraph('항목', styles['cellhead']), Paragraph('값', styles['cellhead'])],
        ['원본 분류', '혼합물 613개(보고) / 실제파일 mixture 608개'],
        ['최종 학습셋(보고)', '579개 — 파일 기준 재확인 필요'],
        ['상충라벨 제외', '8쌍(16개) 중 8개 제외'],
        ['CASRN 결측(mixture)', '8개 제형'],
        ['단일성분 제형 비율', '373/608 (약 61%)'],
        ['조성중복 그룹', '30그룹 / 62건'],
    ]
    ref_rows_p = [[Paragraph(str(c), styles['cell']) if not isinstance(c, Paragraph) else c for c in r] for r in ref_rows]
    F.append(make_table(ref_rows_p, [55*mm, 93*mm], head_bg=GRAY))

    F.append(Spacer(1, 4*mm))
    F.append(Paragraph('2주차 할 일 추천', styles['h2']))
    rows2 = [
        [Paragraph('순서', styles['cellhead']), Paragraph('작업', styles['cellhead']), Paragraph('비고', styles['cellhead'])],
        ['0.', '613→608→579 격차 재추적 + 조성동일 30그룹 확인',
         Paragraph('벡터화 전 <b>필수 선행</b>', styles['cell'])],
        ['1.', '0번 완료 후 "최종 N개" 재확정+근거 기록', ''],
        ['2.', 'CASRN결측 8건, skin과 같은 "~-related" 유형인지 파악', 'skin팀과 비교'],
        ['3.', 'skin과 같은 원칙으로 중복조성 처리 결정(스키마 통일)', 'leakage 방지'],
        ['4.', '위 1~3 반영 후 CASRN 표기 통일(CAS매핑 확정)', ''],
        ['5.', 'formulation_id × casrn pivot table(벡터화)', '원 계획대로'],
        ['6.', '단일성분 분리 — 단일성분 비율 373/608(약 61%)', '원 계획대로'],
        ['(여유시)', 'DRD_usable.csv 등 보강데이터 실사용 여부 결정', ''],
    ]
    rows2_p = [[Paragraph(str(c), styles['cell']) if not isinstance(c, Paragraph) else c for c in r] for r in rows2]
    F.append(make_table(rows2_p, [18*mm, 90*mm, 40*mm]))

    doc.build(F)


if __name__ == '__main__':
    build_skin_pdf(r'C:\Users\UserK\Desktop\학술제\세팅\skin_2주차_준비.pdf')
    build_eye_pdf(r'C:\Users\UserK\Desktop\학술제\세팅\eye_2주차_준비.pdf')
    print('done')
