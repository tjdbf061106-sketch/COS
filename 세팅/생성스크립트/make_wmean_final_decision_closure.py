# -*- coding: utf-8 -*-
"""wmean vs max 논쟁 최종 종결 보고서. 사용자(팀) 최종 결정: wmean 유지.
2026-11-09 하루 동안의 전체 흐름(채택논쟁 정리 → 팀원 교정 반영 → nested CV 확정비교 →
해석가능성/견고성 재검토 → 최종 결정)을 한 장으로 요약.
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
    'good': ParagraphStyle('good', fontName='Malgun-Bold', fontSize=11, leading=15,
                            textColor=GOOD, spaceAfter=1.5 * mm),
    'cell': ParagraphStyle('cell', fontName='Malgun', fontSize=7.9, leading=10.8),
    'cellhead': ParagraphStyle('cellhead', fontName='Malgun-Bold', fontSize=8, leading=10.8,
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
    F.append(Paragraph('[Skin] wmean vs max 논쟁 최종 종결 \u2014 wmean 유지', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-11-09 · production 변경 없음, 코드/체크포인트 그대로',
                        styles['subtitle']))

    F.append(Paragraph('최종 결정: wmean 유지 (production 변경 없음)', styles['good']))
    F.append(Paragraph(
        'nested CV 확정비교(outer5\u00d7inner3, 동일 649건, 사전등록 기준)에서 max가 통계적으로는 '
        '명확히 앞섰음(20/20회, paired MCC +0.056~+0.096, p&lt;0.000001)에도 불구하고, 팀은 <b>wmean 유지</b>를 '
        '최종 선택함. 통계적 우위만으로 전환을 결정하지 않고, 아래 해석가능성·견고성 문제를 더 중요하게 '
        '본 판단.', styles['body']))

    F.append(Paragraph('결정에 이른 전체 흐름', styles['h2']))
    rows = [
        ['단계', '산출물', '핵심 내용'],
        ['① 타임라인 정리', '집계방식_wmean_vs_max_채택논쟁_정리_20261109.pdf',
         '8차례 비교에서 1위가 계속 바뀐 이력 정리, "채택반대/찬성" 양쪽 근거 평가'],
        ['② 팀원 교정 반영', 'wmean_vs_max_팀원교정반영_최종합의_20261109.pdf',
         '팀원이 지적한 4개 오류(표본불일치/threshold 서술오류/class_weight 혼입/15rep에 '
         'max 없음) 전부 레포로 재검증·인정, "확정 비교 프로토콜" 합의'],
        ['③ nested CV 확정비교', 'wmean_vs_max_확정비교_nestedCV_결과_20261109.pdf',
         '동일 649건·outer5\u00d7inner3·사전등록 4개 기준 전부 충족 \u2192 통계적으로는 "max 전환 검토" 판정'],
        ['④ 해석가능성·견고성 재검토', '대화 기록(미문서화 \u2192 본 문서에서 공식화)',
         'max는 성분 농도를 전혀 반영하지 않는 구조(코드 확인: desc_full.loc[...].max(axis=0), '
         'pct 미참조) \u2014 극미량 성분 하나가 특정 descriptor 컬럼의 값을 좌우할 수 있음. '
         '새로운(학습 안 해본) mixture에 적용할수록 이 민감성이 더 큰 리스크로 작용 가능. '
         'wmean은 EPA/GHS 농도가중 합산 원리와 대응되는 유일한 방식(top=governing-ingredient, '
         'max=대응 원칙 없음)'],
        ['⑤ 최종 결정', '본 문서', '통계적 우위(③)보다 해석가능성·견고성 리스크(④)를 우선 \u2014 wmean 유지'],
    ]
    F.append(make_table(cellify(rows), [26 * mm, 50 * mm, 90 * mm]))

    F.append(Paragraph('이 결정이 의미하는 것', styles['h3']))
    F.append(bullets([
        '<b>코드·체크포인트 변경 없음.</b> modeling/skin_rf/train.py·evaluate.py·predict.py, '
        'skin_rf_checkpoint.joblib(wmean·resolved_pct 포함·2267차원, 2026-11-08 교체분) 전부 그대로 유지.',
        'max 집계 관련 산출물(nested_cv_wmean_vs_max.py, skin_wmean_vs_max_nested_cv.csv, '
        '32조건 그리드의 min/max 열)은 "검토했으나 채택하지 않음"의 증거 자료로 레포에 보존.',
        '향후 같은 질문이 다시 나오면(예: 다른 라벨/모델 조합에서 max를 재검토) 이번 결정의 근거 '
        '(농도 무시 구조, 신규 mixture 민감성)부터 재확인할 것 \u2014 성능 지표만으로 재논쟁하지 않음.',
        '세팅/진행상황.md가 다른 팀원 작업으로 dirty하여 이 결정은 아직 그 파일에 병합되지 않음 '
        '(6차 min/max 결과 항목 뒤에 "최종: wmean 유지"로 이어붙일 내용, 병합 대기).',
    ]))

    doc.build(F)


if __name__ == '__main__':
    build_pdf(r'C:\Users\UserK\학술제\세팅\wmean_vs_max_최종결정_wmean유지_20261109.pdf')
    print('done')
