# -*- coding: utf-8 -*-
"""wmean vs max 확정 비교(outer5xinner3 nested CV, 동일 649건, 사전등록 기준) 결과와
최종 권고를 담은 팀 공유용 보고서. 팀원 제안 프로토콜(2026-11-09)의 실제 실행 결과.
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
GOODBG = colors.HexColor('#E8F5EC')

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
    'good': ParagraphStyle('good', fontName='Malgun-Bold', fontSize=10, leading=14.5,
                            textColor=GOOD, spaceAfter=1.5 * mm),
    'warn': ParagraphStyle('warn', fontName='Malgun-Bold', fontSize=9, leading=13,
                            textColor=WARN, spaceAfter=1.5 * mm),
    'cell': ParagraphStyle('cell', fontName='Malgun', fontSize=7.9, leading=10.8),
    'cellhead': ParagraphStyle('cellhead', fontName='Malgun-Bold', fontSize=8, leading=10.8,
                                textColor=colors.white),
    'cellgood': ParagraphStyle('cellgood', fontName='Malgun-Bold', fontSize=7.9, leading=10.8,
                                textColor=GOOD),
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


def cellify(rows, good_cols=None):
    good_cols = good_cols or set()
    out = []
    for r, row in enumerate(rows):
        line = []
        for c, val in enumerate(row):
            if r == 0:
                style = styles['cellhead']
            elif c in good_cols:
                style = styles['cellgood']
            else:
                style = styles['cell']
            line.append(Paragraph(str(val), style))
        out.append(line)
    return out


def build_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                             topMargin=14 * mm, bottomMargin=12 * mm,
                             leftMargin=15 * mm, rightMargin=15 * mm)
    F = []
    F.append(Paragraph('[Skin] wmean vs max 확정 비교 결과 \u2014 nested CV', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-11-09 · 팀원 제안 프로토콜(outer5\u00d7inner3, 동일 649건, 사전등록 기준) 실제 실행 결과',
                        styles['subtitle']))

    F.append(Paragraph('1. 결론', styles['h2']))
    F.append(Paragraph(
        '<b>사전에 정해둔 4개 판정 기준을 전부 충족 \u2014 "max 전환 검토" 판정.</b> '
        '20회 반복 전부(20/20)에서 max가 wmean을 앞섰고, 통계적으로도 우연이라고 보기 어려운 수준(p&lt;0.000001). '
        '다만 이 보고서는 <b>"전환해도 되는지" 근거를 제공하는 것이지, 전환을 실행한 것은 아님</b> \u2014 '
        '실제 production 반영은 팀 확인 후 별도로 진행.', styles['good']))

    F.append(Paragraph('2. 왜 이 비교가 "확정"에 가까운가', styles['h2']))
    F.append(Paragraph(
        '이전까지의 비교들은 두 가지 문제가 있었음(팀원 지적, 2026-11-09 교정 반영 보고서에서 확인됨): '
        '① wmean은 659건, max는 649건으로 표본 자체가 달랐음 ② threshold를 고를 때 평가용 데이터를 그대로 '
        '재사용해(OOF 기반 동일 데이터로 기준선 선택+채점) 절대 수치가 낙관적으로 부풀었을 가능성. '
        '이번 비교는 두 문제를 모두 제거함:', styles['body']))
    rows0 = [
        ['항목', '이전 방식(32조건 공식 그리드)', '이번 방식(nested CV 확정비교)'],
        ['표본', 'wmean 659건 / max 649건(불일치)', '두 방식 다 계산 가능한 동일 649건'],
        ['threshold 선택', '평가용 OOF로 선택 + 같은 OOF로 채점', 'inner 3-fold(학습용 내부)에서만 선택,\nouter 봉인된 test는 한 번도 관여 안 함'],
        ['비교 단위', 'wmean·max를 서로 다른 분할로 각자 평가', '같은 rep=같은 outer/inner 분할 공유\n(쌍대응·paired 비교 가능)'],
        ['판정 기준', '결과를 본 뒤 "MCC가 더 높다"로 판단', '결과를 보기 전에 ΔMCC·p-value·gap·\n하위집단 일치 기준을 미리 고정'],
    ]
    F.append(make_table(cellify(rows0), [26 * mm, 62 * mm, 74 * mm]))

    F.append(Paragraph('3. 사전 등록 기준 (결과를 보기 전 고정) → 결과 대조', styles['h2']))
    rows1 = [
        ['기준', '전체(649건)', '다성분 하위집단(274건)', '충족'],
        ['ΔMCC(max\u2212wmean) > 0.02', '+0.0564', '+0.0958', '충족'],
        ['paired t-test p < 0.05', 'p < 0.000001 (t=9.92)', 'p < 0.000001 (t=11.06)', '충족'],
        ['gap(Sens-Spec 격차) 0.05 이상 악화 금지', '-0.0095(오히려 개선)', '-0.0164(오히려 개선)', '충족'],
        ['전체·다성분 방향 일치(모순 없음)', '+ (양수)', '+ (양수)', '충족'],
    ]
    F.append(make_table(cellify(rows1, good_cols={3}), [58 * mm, 40 * mm, 40 * mm, 20 * mm]))

    F.append(Paragraph('4. 상세 수치 (20회 반복, pooled outer 예측 기준)', styles['h3']))
    rows2 = [
        ['구분', 'wmean MCC', 'max MCC', 'wmean Sens/Spec', 'max Sens/Spec', 'wmean AUC', 'max AUC'],
        ['전체(649건)', '0.2457\u00b10.029', '0.3021\u00b10.028', '0.647 / 0.667', '0.692 / 0.691', '0.7151', '0.7368'],
        ['다성분(274건)', '0.2269\u00b10.035', '0.3227\u00b10.031', '0.732 / 0.538', '0.782 / 0.604', '0.7188', '0.7522'],
    ]
    F.append(make_table(cellify(rows2), [24 * mm, 24 * mm, 24 * mm, 28 * mm, 28 * mm, 18 * mm, 18 * mm]))
    F.append(Spacer(1, 2 * mm))
    F.append(Paragraph(
        '20회 반복 전부(20/20)에서 max의 MCC가 wmean보다 높았음(전체·다성분 둘 다) \u2014 ΔMCC 최솟값도 '
        '전체에서 +0.022, 다성분에서 +0.029로 전부 양수. Sens/Spec·AUC도 전부 max가 더 좋거나 동등 이상, '
        '한쪽을 희생해서 다른 쪽이 좋아지는 트레이드오프가 아님.', styles['bodysmall']))

    F.append(Paragraph('5. 남은 한계 (과장 금지)', styles['h3']))
    F.append(bullets([
        '20회 반복은 "같은 649건을 20번 다시 나눈 것"이지 새 표본 20개가 아님 \u2014 "영구 확정"으로 표현하지 않음.',
        'RandomForest·라벨B·+resolved_pct 조합 1가지만 확정 비교함. ExtraTrees 등 다른 모델에도 동일하게 '
        'max가 유리한지는 이번 비교 범위 밖(참고: 32조건 그리드에서는 ExtraTrees도 같은 패턴을 보였음).',
        'max는 descriptor 컬럼별 독립 선택이라 화학적으로 실재하지 않는 조합("프랑켄슈타인 벡터")이 합성될 '
        '수 있다는 해석가능성 문제는 이번 비교로 해소되지 않음 \u2014 성능과 별개의 고려사항.',
    ]))

    F.append(Paragraph('6. 승인 시 진행할 작업 (아직 실행 안 함)', styles['h2']))
    rows3 = [
        ['단계', '내용'],
        ['① feature 파이프라인', 'max 집계(descriptor 컬럼별 최댓값)를 평가용 임시계산이 아니라 skin_vectorized.csv에\n준하는 영구 CSV로 생성'],
        ['② 코드 재작성', 'modeling/skin_rf/train.py, evaluate.py, predict.py의 집계 로직을 wmean에서 max로 교체'],
        ['③ 재학습', '교체된 파이프라인으로 production 체크포인트 재학습'],
        ['④ 체크포인트 교체', '기존 skin_rf_checkpoint.joblib 백업 후 신규 체크포인트로 교체'],
        ['⑤ 검증', 'cosmetic_validation 노트북 등 기존 검증 스크립트 재실행해 정상 동작 확인, 커밋'],
    ]
    F.append(make_table(cellify(rows3), [34 * mm, 132 * mm]))
    F.append(Spacer(1, 2 * mm))
    F.append(Paragraph(
        '팀 확인 후 진행 여부 알려주면 바로 착수함. 산출물: `skin irritation/skin_wmean_vs_max_nested_cv.csv`'
        '(원시 결과), `skin irritation/rf_feature_compare/nested_cv_wmean_vs_max.py`(비교 스크립트, 사전등록 '
        '기준이 코드 docstring에 명시됨).', styles['bodysmall']))

    doc.build(F)


if __name__ == '__main__':
    build_pdf(r'C:\Users\UserK\학술제\세팅\wmean_vs_max_확정비교_nestedCV_결과_20261109.pdf')
    print('done')
