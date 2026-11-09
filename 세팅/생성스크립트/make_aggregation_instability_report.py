# -*- coding: utf-8 -*-
"""skin 집계방식(wmean vs max 등) 채택 여부 논쟁 — 그동안의 전체 비교 타임라인을 정리해
"왜 순위가 계속 뒤집히는가"를 분석하고 권고안을 제시하는 보고서.

2026-11-09: "max로 바꾸지 말자(방법론 바뀔 때마다 순위가 뒤집혀서 신뢰도 낮음)"는 의견과
"바꾸자(가장 최근 공식 재검증에서 max가 이김)"는 의견이 갈려 사용자가 정리 요청.
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
WARN = colors.HexColor('#C0392B')
GOOD = colors.HexColor('#1E7B4D')

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
    'good': ParagraphStyle('good', fontName='Malgun-Bold', fontSize=9, leading=13,
                            textColor=GOOD, spaceAfter=1.5 * mm),
    'cell': ParagraphStyle('cell', fontName='Malgun', fontSize=7.6, leading=10.5),
    'cellhead': ParagraphStyle('cellhead', fontName='Malgun-Bold', fontSize=7.8, leading=10.5,
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
             if not isinstance(c, Paragraph) else c
             for c in row] for r, row in enumerate(rows)]


def build_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                             topMargin=14 * mm, bottomMargin=12 * mm,
                             leftMargin=15 * mm, rightMargin=15 * mm)
    F = []
    F.append(Paragraph('[Skin Irritation] 집계방식(wmean vs max) 채택 논쟁 — 근거 타임라인 분석', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-11-09 · "바꾸지 말자(신뢰도 낮음)" vs "바꾸자(최신 공식결과)" 의견 정리',
                        styles['subtitle']))

    F.append(Paragraph('1. 쟁점', styles['h2']))
    F.append(Paragraph(
        '현재 production은 wmean(투입비율 가중평균) 집계를 쓰고 있고, 2026-11-08 공식 16그리드에서 '
        'min/max까지 추가한 32그리드로 재검증한 결과 <b>라벨B(팀 실제 타겟)에서 max가 wmean보다 '
        'MCC 0.038 높게(0.302 vs 0.264) 나왔음</b>. 이를 두고 두 의견이 갈림:', styles['body']))
    F.append(bullets([
        '<b>채택 반대</b>: 지금까지 같은 비교(wmean/top/min/max 중 어느 게 제일 나은가)를 방법론을 '
        '바꿔가며 여러 번 돌렸는데, 그때마다 "1위"가 달라졌다 — 그만큼 이 비교 자체의 신뢰도가 낮으니 '
        '최신 결과 하나만 보고 바꾸면 안 된다.',
        '<b>채택 찬성</b>: 가장 최근에, 가장 엄밀한 공식 방법론(nz20 필터 + 그룹버그 수정 + 농도피처 + '
        '20회 반복, 15회→20회로 재현까지 확인)으로 나온 결과이니 이걸 신뢰해야 한다.',
    ]))
    F.append(Paragraph(
        '이 보고서는 실제로 "순위가 몇 번, 왜 바뀌었는지"를 전부 데이터로 추적해 두 의견 중 어느 쪽이 '
        '더 타당한지 판단하고 권고안을 제시함.', styles['body']))

    F.append(Paragraph('2. 전체 비교 타임라인 — 라벨B(팀 실제 타겟) 1위 집계방식', styles['h2']))
    rows = [
        ['#', '시점/스크립트', 'n', 'nz20', '그룹버그\n수정', 'threshold', '1위(라벨B)', '비고'],
        ['1', '10/6 구feature\n(544행)', 544, '✗', '해당없음\n(버그 발견 전)', '균형', 'min\n(PR-AUC 2배+)', ''],
        ['2', '11/7 17차\n(541행, 신feature)', 541, '✗', '수정 전', '균형', 'top\n(ET 0.260)', ''],
        ['3', '11/7 18차\nrange-policy', '646~649', '✗', '수정 전', '균형', 'max\n(ET 0.218)', 'top이 오히려\n최하위로 추락'],
        ['4', '11/7 19차(1)\n다성분만(n=296)', 296, '✗', '수정 전', '균형\n(재학습없음)', 'wmean≈max\n동률, top 최하위', '"3의 max 1위는\n단일성분 혼입 착시"'],
        ['5', '11/7 19차(3)\n그룹수정 재검증', 649, '✗', '수정 후', '균형', 'top\n(ET 0.255)', '차이(0.045)&lt;std(0.049)\n"확정 아님"'],
        ['6', '11/8 3차\n(637그룹+농도)', '649/659', '✗', '수정 후', '균형', 'top\n(RF 0.249)', ''],
        ['7', '11/8 4~5차\n공식(wmean,top만)', '649/659', '✓', '수정 후', '균형\n(OOF동일)', 'wmean\n(RF 0.264)', '15회→20회\n재현 확인'],
        ['8', '11/8 min/max확장\n공식 32그리드', '649/659', '✓', '수정 후', '균형\n(OOF동일)', 'max\n(RF 0.302)', '20회 반복,\nwmean보다 +0.038'],
    ]
    F.append(make_table(cellify(rows), [8 * mm, 26 * mm, 13 * mm, 10 * mm, 15 * mm, 14 * mm, 22 * mm, 23 * mm]))
    F.append(Spacer(1, 2 * mm))
    F.append(Paragraph(
        '8번의 비교에서 1위가 <b>min \u2192 top \u2192 max \u2192 (동률) \u2192 top \u2192 top \u2192 wmean \u2192 max</b>로 '
        '바뀜 — 완전히 같은 답이 두 번 연속 나온 적이 없음(5\u21926은 둘 다 "top"이지만 조건이 또 다름). '
        '"채택 반대" 쪽 주장은 숫자로 뒷받침됨.', styles['warn']))

    F.append(Paragraph('3. 왜 이렇게 뒤집히는가 — 원인 4가지', styles['h3']))
    F.append(bullets([
        '<b>① nz20 필터 유무</b>: fold 내부 학습데이터만으로 거의 0인 feature 컬럼을 제거하는 필터. '
        '6\u21927번 전환에서 이 필터 하나 넣었다고 "top 1위" \u2192 "wmean 1위"로 뒤집힘 \u2014 '
        '단일 변수 중 가장 영향이 큼.',
        '<b>② comp_group_id 그룹버그 수정 여부</b>: 수정 전(원본 544개의 97.2%가 조성해시가 아니라 '
        'formulation_id를 그대로 베낀 값)엔 그룹 기반 fold 분리가 사실상 작동 안 해서 누수 섞인 추정치였음. '
        '4\u21925번 전환에서 이 수정 하나로 "max 1위" \u2192 "top 1위"로 바뀜.',
        '<b>③ 표본 구성(단일성분 혼입)</b>: wmean=top=max=min이 수학적으로 똑같아지는 단일성분 제형을 '
        '포함한 전체표본과, 차이가 실제로 나는 다성분만 떼어낸 하위표본에서 순위가 다르게 나옴(4번 항목).',
        '<b>④ OOF 임계값 선택 편향</b>(eye팀 감사에서 2026-11-09 신규 발견): 7\u21928번을 포함해 "공식" 비교 '
        '전부가 평가용 OOF로 threshold까지 고르고 같은 OOF로 점수를 매김 \u2014 모든 조건에 동일하게 섞여있는 '
        '편향이라 상대순위를 반드시 뒤집히게 만들지는 않지만, 8번(max 1위)을 포함한 모든 절대 MCC 수치를 '
        '다소 낙관적으로 부풀릴 가능성이 있음. <b>이건 "채택 찬성" 쪽이 내세우는 8번 결과의 신뢰도를 깎는 '
        '새로운 변수</b>.',
    ]))

    F.append(Paragraph('4. 두 의견 평가', styles['h2']))
    rows2 = [
        ['의견', '근거의 타당성'],
        ['채택 반대\n(신뢰도 낮다)', '<b>타당함.</b> 8번의 비교에서 1위가 계속 바뀐 것은 사실이고, 그 원인(①~④)도 '
         '전부 특정 가능한 방법론 차이임 \u2014 "운이 나빠서 흔들린 것"이 아니라 "매번 다른 질문에 답한 것"에 가까움. '
         '특히 8번(최신, max 1위)도 ④번 편향에서 자유롭지 않음.'],
        ['채택 찬성\n(최신 공식결과 신뢰)', '<b>부분적으로 타당함.</b> 7\u21928번은 적어도 ①②를 전부 통제한 상태(같은 '
         'nz20, 같은 그룹버그수정)에서 나온 비교이고, 15회\u219220회 반복으로 재현성도 확인됨. 다만 ④번(OOF 편향)은 '
         '아직 통제 안 됐고, wmean은 n=659인데 top/min/max는 n=649로 <b>표본 자체도 완전히 동일하지 않음</b>'
         '(eye팀 감사가 이미 지적).'],
    ]
    F.append(make_table(cellify(rows2), [36 * mm, 130 * mm]))
    F.append(Spacer(1, 3 * mm))

    F.append(Paragraph('5. 권고', styles['h2']))
    F.append(Paragraph(
        '<b>지금 당장 max로 전환하지 않음을 권고함.</b> 그렇다고 "영원히 결론 못 낸다"는 뜻은 아님 \u2014 '
        '남은 불확실성(④번, 표본 불일치)을 해소하는 단 한 번의 확정 실험으로 끝낼 수 있음.', styles['good']))
    F.append(bullets([
        '<b>확정 실험 설계</b>: wmean vs max 두 가지만, 동일 649건(구조 해결 성분이 있어 네 방식 모두 계산 가능한 '
        '제형만) \u00d7 동일 comp_group_id 5-fold \u00d7 nested 평가(outer test 봉인, inner 3-fold에서만 '
        'threshold 선택) \u00d7 20회 반복.',
        '이렇게 하면 ①②는 이미 통제됨(둘 다 nz20/그룹수정 유지), ③은 해당 없음(둘 다 전체표본 비교), '
        '④(OOF 편향)와 표본 불일치가 둘 다 해소됨 \u2014 이번에야말로 "방법론 차이로 인한 흔들림"을 다 걷어낸 비교.',
        '이 실험에서도 max가 이기면 그때 전환(결정은 §5의 이전 안내대로 train.py/evaluate.py/predict.py '
        '집계로직 재작성 필요). wmean이 이기거나 차이가 변동폭 이내면 현행 유지로 확정하고 더 이상 재검증하지 않음.',
        '그 전까지는 <b>현재 production(wmean) 유지</b> \u2014 바꾸지 않는 쪽이 "이미 검증된 상태"를 지키는 '
        '것이므로 더 안전한 기본값.',
    ]))

    doc.build(F)


if __name__ == '__main__':
    build_pdf(r'C:\Users\UserK\학술제\세팅\집계방식_wmean_vs_max_채택논쟁_정리_20261109.pdf')
    print('done')
