# -*- coding: utf-8 -*-
"""Lechuga 2025 계면활성제 mixture 데이터 확보 보고 PDF 생성 스크립트.

용어 주의: 이 작업은 "전이학습"이 아니라 "외부검증/일반화 테스트"로 부른다
(세션_핸드오프_2026-11-07.md 5절 규칙 — 교수님 발표 때 오버클레임 방지).
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
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
                             spaceAfter=2 * mm, leading=20),
    'subtitle': ParagraphStyle('subtitle', fontName='Malgun', fontSize=9, textColor=GRAY,
                                spaceAfter=5 * mm, leading=12),
    'h2': ParagraphStyle('h2', fontName='Malgun-Bold', fontSize=12, textColor=BLUE,
                          spaceBefore=4 * mm, spaceAfter=2 * mm, leading=15),
    'h3': ParagraphStyle('h3', fontName='Malgun-Bold', fontSize=10, textColor=colors.black,
                          spaceBefore=2 * mm, spaceAfter=1 * mm, leading=13),
    'body': ParagraphStyle('body', fontName='Malgun', fontSize=9, leading=13,
                            spaceAfter=1.5 * mm, textColor=colors.black),
    'bodysmall': ParagraphStyle('bodysmall', fontName='Malgun', fontSize=8.3, leading=11.5,
                                 spaceAfter=1 * mm, textColor=colors.black),
    'warn': ParagraphStyle('warn', fontName='Malgun-Bold', fontSize=9, leading=13,
                            textColor=WARN, spaceAfter=1.5 * mm),
    'cell': ParagraphStyle('cell', fontName='Malgun', fontSize=8, leading=11),
    'cellhead': ParagraphStyle('cellhead', fontName='Malgun-Bold', fontSize=8.3, leading=11,
                                textColor=colors.white),
}


def bullets(items, style='bodysmall'):
    return ListFlowable(
        [ListItem(Paragraph(t, styles[style]), leftIndent=4 * mm, value='\u2022') for t in items],
        bulletType='bullet', leftIndent=2 * mm, spaceBefore=0, spaceAfter=2 * mm,
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


def p(text, style='cell'):
    return Paragraph(text, styles[style])


def build_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                             topMargin=15 * mm, bottomMargin=13 * mm,
                             leftMargin=16 * mm, rightMargin=16 * mm)
    F = []
    F.append(Paragraph('[외부검증/일반화 테스트] 계면활성제 Mixture 자극성 데이터 확보 보고', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-11-07 · 작성: 김서율 세션', styles['subtitle']))

    F.append(Paragraph('0. 용어 확인', styles['h2']))
    F.append(Paragraph(
        '세션_핸드오프_2026-11-07.md 5절 규칙에 따라 화장품 관련 적용은 '
        '<b>"전이학습"이 아니라 "외부검증/일반화 테스트"</b>로 표기함. 아래 내용은 모델 파라미터를 '
        '화장품 데이터로 재학습하는 작업이 아니라, production 모델(또는 그 표현)이 다른 도메인에서도 '
        '쓸 만한지 검증·보강하는 작업임.', styles['bodysmall']))

    F.append(Paragraph('1. 배경', styles['h2']))
    F.append(bullets([
        '기존 화장품 mixture 데이터 탐색은 EPA/GHS 라벨 기준으로 한정했다가 대부분 실패(COSMOS DB 폐기, SCCS/식약처/상업제품 논문 전수조사 세션 한도로 대부분 실패)',
        '기준을 완화함 — <b>EPA/GHS가 아니어도, 단일성분이 아니라 mixture 기준이어도, 실험데이터이기만 하면</b> 특정 기준으로 binary label 변환만 가능하면 채택',
        '이 기준으로 재탐색한 결과 <b>Lechuga et al. 2025(ACS Omega) 계면활성제 mixture Zein 자극성 데이터</b> 확보',
    ]))

    F.append(Paragraph('2. 확보한 데이터', styles['h2']))
    F.append(Paragraph('① 출처', styles['h3']))
    F.append(bullets([
        'Lechuga et al., <i>ACS Omega</i> 10(46):55868\u201355878 (2025)',
        'Zenodo DOI 10.5281/zenodo.15800001 (CC-BY-4.0, 무료 공개, 2025-07-03)',
        '계면활성제 20종 + 이들의 <b>2성분 mixture</b>, 농도 0.5/1/2%, 이온성 조합 A(Anionic)/NI(Nonionic)/AA/ANI/NINI, HLB1·HLB2·HLB_compost(기하평균) 포함',
    ]))
    F.append(Paragraph('② 측정값 → binary label 변환 근거', styles['h3']))
    F.append(bullets([
        '측정값: Zein Number(ZN) — 계면활성제가 zein 단백질을 변성/용해시키는 정도를 재는 in vitro 피부자극 대체시험(Draize 아님)',
        '<b>ECVAM DB-ALM Protocol n\u00b026</b>(The Zein Test, 1991, 공식 규제기관 문서)이 3단계 공식 cutoff 제공: Non-irritant(ZN&lt;200) / Moderate(200\u2013400) / Strong(&gt;400)',
        '→ <b>ZN\u2265200 = 1(자극), ZN&lt;200 = 0(비자극)</b>으로 변환(추측 아니라 공인 cutoff 사용)',
    ]))
    F.append(Paragraph('③ 처리 결과 파일', styles['h3']))
    ref_rows = [
        [p('파일', 'cellhead'), p('내용', 'cellhead')],
        [p('cosmetic_validation/lechuga2025_surfactant_mixture_zein.csv'),
         p('205행(ZN 결측 72행 제외) · label 0:149 / 1:56(양성비율 약 27%, 현 production보다 균형적) · mixture_type=binary_mixture 117행 / single 88행')],
        [p('cosmetic_validation/Lechuga2025_surfactant_zein_dataset.xlsx'),
         p('원본 보관(재가공 가능하도록)')],
    ]
    F.append(make_table(ref_rows, [75 * mm, 100 * mm], head_bg=GRAY))

    F.append(Paragraph('3. 한계 (발표/보고 시 반드시 같이 언급할 것)', styles['h2']))
    F.append(Paragraph('⚠ Zein test ≠ Draize', styles['warn']))
    F.append(bullets([
        'in vitro 대체시험이라 실제 Draize/EPA 등급 예측력 향상을 바로 주장할 수 없음 — "표현/보조정보로서 도움이 되는지"만 평가 가능',
    ]))
    F.append(Paragraph('⚠ 성분 구조(SMILES/CAS) 대부분 미해결', styles['warn']))
    F.append(bullets([
        '성분명이 INCI명(Laureth-11 Carboxylic Acid, Glycereth-7 Cocoate 등)으로 대부분 UVCB(비균질 다성분) 물질 — 지금 단계에서 구조 feature 적용 불가, HLB·이온성 등 비구조 feature만 사용 가능',
    ]))

    F.append(PageBreak())

    F.append(Paragraph('4. 적용(외부검증) 시 필요한 절차', styles['h2']))
    F.append(Paragraph(
        'pretraining/ 폴더에 이미 ChemBERTa 기반 mixture 인코딩 인프라(protocol.json 기준)가 있어 '
        '그 틀에 맞추는 것을 전제로 함.', styles['bodysmall']))
    steps_rows = [
        [p('단계', 'cellhead'), p('내용', 'cellhead'), p('비고', 'cellhead')],
        ['1', p('구조 해결(선행 블로커) — 20종 중 단일구조로 환원 가능한 성분만 SMILES 확보'),
         p('UVCB는 미해결(unresolved) 유지, 추측 금지')],
        ['2', p('mixture_predictor.py 입력 스키마 변환 — amount_pct(함량%)·structure_status 등'),
         p('2성분 비율(proportion)→절대% 환산 근거는 원문 재확인 필요')],
        ['3', p('라벨을 skin_epa와 분리된 별도 엔드포인트(예: skin_zein)로 유지'),
         p('Draize 예측력 주장과 혼동 방지')],
        ['4', p('적용 방식 택1 — (A)MLM 코퍼스 확장 (B)보조 head/멀티태스크 (C)warm-start'),
         p('아래 5절 비교표 참조')],
        ['5', p('검증 — group-aware split, scaffold split, train-only imputer, 결과 보기 전 설계 동결'),
         p('protocol.json 기존 원칙 그대로 적용')],
    ]
    F.append(make_table(steps_rows, [10 * mm, 115 * mm, 50 * mm]))

    F.append(Paragraph('5. 적용 방식 비교', styles['h2']))
    cmp_rows = [
        [p('방식', 'cellhead'), p('내용', 'cellhead'), p('장단점', 'cellhead')],
        [p('(A) MLM 코퍼스 확장'), p('Lechuga 성분 SMILES를 추가 MLM 코퍼스에 합쳐 pretraining/best 재학습'),
         p('가장 안전, skin_epa/eye_epa 안 건드림 / 효과는 작을 수 있음')],
        [p('(B) 보조 head(권장)'), p('같은 pooled 임베딩 위에 skin_zein 전용 head만 새로 학습'),
         p('구조상 가장 깔끔, 기존 head 영향 없음')],
        [p('(C) warm-start'), p('Zein 데이터로 먼저 학습 후 그 가중치를 skin_epa 분류기 초기값으로'),
         p('표본이 적어 과적합 위험 큼 / RandomForest(skin_rf)엔 적용 불가')],
    ]
    F.append(make_table(cmp_rows, [28 * mm, 85 * mm, 62 * mm]))

    F.append(Paragraph('6. 팀 결정이 필요한 사항', styles['h2']))
    F.append(bullets([
        '적용 방식 (A)/(B)/(C) 중 무엇으로 갈지',
        '20종 성분 SMILES 수작업 확인을 누가/언제 할지(대부분 UVCB라 시간이 걸림)',
        '2성분 mixture의 proportion("1-1" 등) → 절대 함량% 환산 규칙을 논문 원문에서 재확인할지',
        '발표자료에 이 작업을 "외부검증/일반화 테스트"로 표기하는 것에 이견 없는지',
    ], style='body'))

    F.append(Spacer(1, 3 * mm))
    F.append(Paragraph('참고 — 세팅/진행상황.md 14~15차 항목에 상세 기록됨', styles['bodysmall']))

    doc.build(F)


if __name__ == '__main__':
    build_pdf(r'C:\Users\UserK\학술제\세팅\Lechuga_계면활성제_mixture_외부검증_보고서.pdf')
    print('done')
