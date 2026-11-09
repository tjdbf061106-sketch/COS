# -*- coding: utf-8 -*-
"""eye팀 "Eye 최종 모델 외부 검증 결과"(샴푸 5종, 카카오톡 공유 결과보고서.html) 더블체크 보고서.

확인 항목: 모델/코드 무결성 재현, 라벨 극성 버그 여부, Eye A(라벨A) 선택 근거,
MCC=-1.000 헤드라인 수치의 통계적 의미.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('Malgun', r'C:\Windows\Fonts\malgun.ttf'))
pdfmetrics.registerFont(TTFont('Malgun-Bold', r'C:\Windows\Fonts\malgunbd.ttf'))

BLUE = colors.HexColor('#1F4E79')
GRAY = colors.HexColor('#595959')
LIGHT = colors.HexColor('#EAF1F8')
GOOD = colors.HexColor('#1E7B4D')
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
    'good': ParagraphStyle('good', fontName='Malgun-Bold', fontSize=9, leading=13,
                            textColor=GOOD, spaceAfter=1.5 * mm),
    'warn': ParagraphStyle('warn', fontName='Malgun-Bold', fontSize=9, leading=13,
                            textColor=WARN, spaceAfter=1.5 * mm),
    'cell': ParagraphStyle('cell', fontName='Malgun', fontSize=7.8, leading=10.5),
    'cellhead': ParagraphStyle('cellhead', fontName='Malgun-Bold', fontSize=7.9, leading=10.5,
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
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def cellify(rows):
    return [[Paragraph(str(c), styles['cellhead'] if r == 0 else styles['cell'])
             for c in row] for r, row in enumerate(rows)]


def build_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                             topMargin=14 * mm, bottomMargin=12 * mm,
                             leftMargin=15 * mm, rightMargin=15 * mm)
    F = []
    F.append(Paragraph('[Eye] "최종 모델 외부 검증 결과"(샴푸5종) skin팀 더블체크', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-11-09 · 원본: 카카오톡 공유 결과보고서.html(Eye A / LR / wmean)',
                        styles['subtitle']))

    F.append(Paragraph('1. 대상 보고서 요약', styles['h2']))
    F.append(Paragraph(
        'eye팀이 반려동물 벼룩/진드기 샴푸 5종의 공개 SDS 성분을 Eye A 최종모델(`최종/최종모델/eye_A_LR_wmean.joblib`, '
        'LogisticRegression+wmean, 라벨A: I/II=양성·III/IV=음성, threshold=0.481487)에 넣어 계산. '
        '<b>헤드라인 수치: 제품별 4개 계산 가능(1개는 구조 미해결로 계산불가) 중 0/4 라벨 일치, MCC -1.000.</b> '
        '등록번호 기준 3계열로 묶어도 0/3, MCC -1.000. 보고서 스스로 "탐색적 사례 비교이며 정식 외부검증 표본은 0건"이라고 '
        '명시함.', styles['body']))

    F.append(Paragraph('2. 더블체크 항목', styles['h2']))
    rows = [
        ['항목', '방법', '결과'],
        ['모델 파일 무결성', 'eye_A_LR_wmean.joblib SHA256 직접 계산', 'smoke_test_cases.json 기재값과 일치'],
        ['코드 재현성\n(다른 패키지버전)', 'smoke_test_cases.json 6개 가상입력을 predict.py로\n직접 재실행(numpy 2.5.0, 잠금버전 2.4.6과 다름)',
         '6/6 점수·판정 소수점 6자리까지 일치'],
        ['threshold 일치', 'selected_parameters.json의 threshold(0.4814867\u2026)와\n보고서 기재값(0.481486747469) 대조', '일치'],
        ['라벨 극성 버그 여부', 'metadata.json의 target/positive_class 정의와\n보고서 "라벨"(II\u21921, III\u21920) 대조', '일치, 반전 없음'],
        ['Eye A(라벨A) 선택 근거', 'delta_mcc_by_label_model.csv(eye 자체 공식비교)에서\n라벨A vs 라벨B baseline MCC 대조', '6개 모델 전부 라벨A가 라벨B보다 높음\n(LR 0.336 vs 0.156 등) \u2014 선택 타당'],
        ['원자료/커밋 추적', '보고서가 명시한 커밋 a693ae5를 레포 git log에서 대조', '실제 존재하는 커밋, 최종모델/ 전체가\n이 커밋에서 추가됨 \u2014 위조 아님'],
    ]
    F.append(make_table(cellify(rows), [28 * mm, 70 * mm, 68 * mm]))

    F.append(Paragraph('3. "MCC -1.000"을 어떻게 읽어야 하는가', styles['h3']))
    F.append(bullets([
        '<b>n=4, 이진분류에서 MCC=-1.000은 "4개 전부 반대로 맞춤"과 수학적으로 동일</b> \u2014 공정한 동전을 4번 던져 '
        '전부 틀릴 확률이 6.25%(0.5⁴)이므로, 극소표본에서는 특별히 놀라운 사건이 아님.',
        '<b>구조 해결 함량이 4개 제품 모두 0.78~3.93%뿐</b> \u2014 SDS에 농도가 공개된 성분(대개 유효성분)만 반영했고, '
        '비공개 "불활성 성분"이 제형의 96% 이상을 차지함. 사실상 제형의 극히 일부 정보만으로 계산한 값.',
        '모델 점수가 전부 0.415~0.488로 threshold(0.4815) 바로 근처에 몰려 있음 \u2014 보고서가 직접 밝힌 대로 '
        '4개 중 2개(Zodiac, Adams Plus)는 공개된 성분 함량 범위 안에서 점수가 threshold를 넘나듦(실측: 0.4809~0.4883). '
        '"0/4 불일치"는 특정 중간 함량 가정 하나에서 나온 스냅샷일 뿐, 모든 가능 조성에서 일관된 결과가 아님.',
        '더 크고 구조 커버리지가 높은 보조표본(CIR 성분단독 n=22, 100%원료 n=7)은 MCC 각각 0.089 / 0.167로 '
        '<b>방향은 양수</b>(모델이 뒤집혀 있지 않음)로 나옴 \u2014 샴푸 4개 결과가 "모델 자체의 구조적 결함"이 아니라 '
        '"극소표본 + 저정보 입력"에서 오는 변동임을 뒷받침.',
    ]))

    F.append(Paragraph('4. 결론', styles['h2']))
    F.append(Paragraph(
        'eye팀 보고서는 수치 조작·과장 없이 정직하게 작성됐고 스스로 적절한 수준으로 선을 긋고 있음(탐색적 비교로 명시, '
        '정식 외부검증 아님을 명시, threshold 불안정성 직접 공개, 등록번호 중복을 독립표본으로 해석 금지 명시).', styles['good']))
    F.append(bullets([
        '모델/코드/threshold/라벨정의 전부 독립 재현 확인 \u2014 버그나 조작 없음.',
        '"Eye A(라벨A, 농도無)" 최종 채택은 eye팀 자체 공식데이터(6개 모델 전부 라벨A 우세)로 뒷받침됨 \u2014 skin이 '
        '라벨B를 택한 것과 반대 결론이지만, 데이터가 다르면 결론이 다른 것은 정상.',
        '<b>"MCC -1.000" 헤드라인만 떼어서 "Eye 모델이 깨졌다"고 해석하면 안 됨</b> \u2014 n=4·저커버리지·threshold '
        '근접이라는 3중 불안정 조건이 겹친 결과이며, 더 큰 보조표본에서는 방향이 정상(양수)으로 나옴.',
        '추가 조치 불필요 \u2014 eye팀 보고서 그대로 "참고용 탐색적 사례"로 보관하면 됨. skin팀 측에서 별도로 고칠 코드나 '
        '지적할 결함은 없음.',
    ]))

    doc.build(F)


if __name__ == '__main__':
    build_pdf(r'C:\Users\UserK\학술제\세팅\Eye_외부검증(샴푸5종)_더블체크_20261109.pdf')
    print('done')
