# -*- coding: utf-8 -*-
"""Skin RF production 공식 재검증(comp_group_id 수정 + 라벨 B 적용 후) 보고서 PDF 생성.

2026-11-07: comp_group_id 누수버그 수정(조성해시 UNION wmean벡터해시) + binary_label A->B
전환 후, 공식 반복 group 5-fold 스크립트(rf_feature_compare/rf_production_recheck.py)를
재실행해 production RandomForest(+wmean 집계)의 신뢰할 수 있는 성능을 확인한 결과를 정리.
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


def cellify(rows):
    return [[Paragraph(str(c), styles['cellhead'] if r == 0 else styles['cell'])
             if not isinstance(c, Paragraph) else c
             for c in row] for r, row in enumerate(rows)]


def build_pdf(path):
    doc = SimpleDocTemplate(path, pagesize=A4,
                             topMargin=15 * mm, bottomMargin=13 * mm,
                             leftMargin=16 * mm, rightMargin=16 * mm)
    F = []
    F.append(Paragraph('[Skin Irritation] RF production 공식 재검증 보고서', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-11-07 · 대상: modeling/skin_rf (RandomForest, wmean 집계, 라벨 B)',
                        styles['subtitle']))

    F.append(Paragraph('1. 배경 — 오늘 적용된 두 가지 수정', styles['h2']))
    F.append(bullets([
        '<b>comp_group_id 누수 수정</b>: 원본 544건의 comp_group_id 중 97.2%가 formulation_id를 그대로 복사한 값이라 '
        'train/test 그룹 분리가 사실상 작동하지 않고 있었음(83개 그룹·276행이 특징벡터는 동일한데 서로 다른 그룹으로 '
        '찢어져 있었음). 조성 시그니처 해시 \u222a wmean 특징벡터 해시(union-find)로 재계산 \u2192 635개 \u2192 463개 그룹.',
        '<b>binary_label A \u2192 B 전환</b>: 팀 결정에 따라 production 타겟 정의를 라벨 A(양성 37/662)에서 '
        '라벨 B(양성 122/662, 결측 제외 후 120/659)로 변경.',
        '이 두 수정은 <b>모델을 더 똑똑하게 만든 것이 아니라, 성능 비교 결과를 신뢰할 수 있게 만든 것</b>임 — '
        '아래 수치 변화는 "정확도가 올랐다/내렸다"가 아니라 "더 믿을 수 있는 숫자로 교체됐다"로 읽어야 함.',
    ]))

    F.append(Paragraph('2. 재검증 방법', styles['h3']))
    F.append(bullets([
        '스크립트: <b>skin irritation/rf_feature_compare/rf_production_recheck.py</b> (수정 없이 그대로 재실행 — '
        'binary_label/comp_group_id를 skin_vectorized.csv에서 그때그때 동적으로 읽으므로 오늘 수정이 자동 반영됨)',
        '모델: RandomForestClassifier(class_weight=balanced) — <b>현재 production과 동일 설정</b>(wmean 집계, nz20 특징필터)',
        '분할: comp_group_id 기준 group 5-fold, <b>20회 반복</b>(seed 0~19) \u2014 매 반복 fold별 검증셋에서 '
        '|Sensitivity-Specificity| 최소가 되는 threshold를 고르고(0.5 고정이 아님), out-of-fold 예측으로 집계',
        '사용 데이터: 659건(binary_label 분포 0=539 / 1=120), 2,266차원 특징',
    ]))

    F.append(Paragraph('3. 결과 — 수정 전(11-06, 라벨A) vs 수정 후(11-07, 라벨B)', styles['h2']))
    F.append(Paragraph(
        '두 줄은 "같은 반복 group 5-fold + 균형점 threshold" 방법으로 측정된 값이라 서로 비교 가능함. '
        '다만 라벨 정의(A\u2192B)와 그룹핑 수정이 동시에 들어간 비교라 두 변화를 분리해서 보여주지는 못함.',
        styles['bodysmall']))
    rows = [
        ['구분', 'Sensitivity', 'Specificity', '격차(|Sens-Spec|)', 'MCC', 'ROC-AUC', 'threshold'],
        ['11-06 (라벨A, comp_group_id 수정 전)', '0.703\u00b10.038', '0.700\u00b10.026', '0.027\u00b10.022',
         '0.199\u00b10.031', '0.741\u00b10.015', '0.044\u00b10.005'],
        ['11-07 (라벨B, comp_group_id 수정 후)', '0.636\u00b10.022', '0.642\u00b10.021', '0.014\u00b10.008',
         '0.218\u00b10.033', '0.701\u00b10.016', '0.200\u00b10.007'],
    ]
    F.append(make_table(cellify(rows), [48 * mm, 20 * mm, 20 * mm, 24 * mm, 18 * mm, 18 * mm, 18 * mm]))
    F.append(Spacer(1, 3 * mm))

    F.append(Paragraph('4. 해석', styles['h3']))
    F.append(Paragraph(
        '수정 후에도 production RandomForest(wmean 집계)는 <b>균형점 threshold를 쓰면 Sensitivity/Specificity가 '
        '거의 동률(0.636 vs 0.642, 격차 1.4%p)</b>로 상당히 안정적임.', styles['good']))
    F.append(bullets([
        'MCC는 오히려 소폭 상승(0.199\u21920.218), 격차는 더 줄어듦(2.7%p\u21921.4%p) \u2014 '
        '"수정했더니 성능이 나빠졌다"는 아님.',
        'Sensitivity/Specificity 절대값과 ROC-AUC는 소폭 하락(0.70\u21920.64대, 0.741\u21920.701) \u2014 '
        '양성 정의가 더 느슨한 쪽(A\u2192B)으로 바뀌면서 경계가 애매한 사례가 늘어난 영향 + fold 구성이 바뀐 영향이 '
        '섞여 있음. comp_group_id 수정만의 순수 효과는 아님.',
        '<b>핵심</b>: threshold를 0.5로 고정하지 않고 제대로 튜닝하면 라벨 B·수정된 그룹핑에서도 RF가 '
        '민감도·특이도 어느 한쪽에 치우치지 않는다는 것이 반복 5-fold로 확인됨.',
    ]))

    F.append(Paragraph('5. 이번 재검증이 다루지 않은 것 (중요)', styles['h3']))
    F.append(Paragraph(
        '이 보고서는 "현재 production 설정(RandomForest + wmean)"의 공식 재검증일 뿐, 별도 노트북'
        '(skin_final_combined_eval.ipynb)에서 나온 "라벨 B는 ExtraTrees+top 조합이 더 낫다"는 결론을 '
        '같은 반복 5-fold 방식으로 재확인한 것은 <b>아직 아님</b>.', styles['warn']))
    F.append(bullets([
        'rf_production_recheck.py는 모델을 RandomForest 한 가지로, 집계방식을 wmean 한 가지로 고정해서만 '
        '테스트하는 스크립트임 — ExtraTrees·top 조합을 같은 반복 group 5-fold 방식으로 넣어 비교하려면 '
        '별도 스크립트 작성이 필요함(단순 재실행이 아님).',
        '따라서 "ExtraTrees+top으로 실제 전환할지"는 이번 재검증과 별개로 아직 열려 있는 결정임. '
        '지금 결과는 "현재 production(RF+wmean)도 라벨B·수정된 데이터에서 꽤 균형 잡혀 있다"는 사실만 추가한 것.',
    ]))

    doc.build(F)


if __name__ == '__main__':
    build_pdf(r'C:\Users\UserK\학술제\세팅\RF_production_재검증_보고서_20261107.pdf')
    print('done')
