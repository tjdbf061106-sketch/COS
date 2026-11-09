# -*- coding: utf-8 -*-
"""확정된 skin irritation RandomForest production 모델 개요 (model card).
2026-11-09: 모델링 종료, eye팀과 함께 모델 확정 — 팀 공유/보고서용 최종 요약 문서.
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
    'title': ParagraphStyle('title', fontName='Malgun-Bold', fontSize=17, textColor=BLUE,
                             spaceAfter=2 * mm, leading=21),
    'subtitle': ParagraphStyle('subtitle', fontName='Malgun', fontSize=9, textColor=GRAY,
                                spaceAfter=5 * mm, leading=12),
    'h2': ParagraphStyle('h2', fontName='Malgun-Bold', fontSize=12.5, textColor=BLUE,
                          spaceBefore=5 * mm, spaceAfter=2 * mm, leading=16),
    'h3': ParagraphStyle('h3', fontName='Malgun-Bold', fontSize=10, textColor=colors.black,
                          spaceBefore=2.5 * mm, spaceAfter=1 * mm, leading=13),
    'body': ParagraphStyle('body', fontName='Malgun', fontSize=9, leading=13.5,
                            spaceAfter=1.5 * mm, textColor=colors.black),
    'bodysmall': ParagraphStyle('bodysmall', fontName='Malgun', fontSize=8.3, leading=11.8,
                                 spaceAfter=1 * mm, textColor=colors.black),
    'good': ParagraphStyle('good', fontName='Malgun-Bold', fontSize=10.5, leading=15,
                            textColor=GOOD, spaceAfter=1.5 * mm),
    'warn': ParagraphStyle('warn', fontName='Malgun-Bold', fontSize=9, leading=13,
                            textColor=WARN, spaceAfter=1.5 * mm),
    'cell': ParagraphStyle('cell', fontName='Malgun', fontSize=7.9, leading=10.8),
    'cellhead': ParagraphStyle('cellhead', fontName='Malgun-Bold', fontSize=8, leading=10.8,
                                textColor=colors.white),
    'mono': ParagraphStyle('mono', fontName='Malgun', fontSize=8, leading=11.5,
                            textColor=colors.HexColor('#333333'), spaceAfter=1 * mm),
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
    F.append(Paragraph('Skin Irritation 예측 모델 — 확정 개요', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 모델링 종료일 2026-11-09 · modeling/skin_rf/skin_rf_checkpoint.joblib',
                        styles['subtitle']))
    F.append(Paragraph('모델 상태: 확정(FINAL) — 이후 추가 모델링 없음', styles['good']))

    F.append(Paragraph('1. 한 줄 요약', styles['h2']))
    F.append(Paragraph(
        'NTP ICE skin 혼합물 자극성 자료(Draize in-vivo, EPA Classification)와 자체 보강 데이터를 합쳐 '
        '구축한 662개 제형을 성분 구조(RDKit descriptor)·농도가중평균(wmean)·농도비율(resolved_pct)로 '
        '벡터화하고, RandomForest(500 trees, nz20 feature 선택, group-aware 5-fold)로 EPA Category 1-3 '
        '(자극 있음) vs Category 4(무자극)를 이진분류하는 모델.', styles['body']))

    F.append(Paragraph('2. 데이터 파이프라인', styles['h2']))
    rows1 = [
        ['단계', '내용'],
        ['원천 데이터', 'NTP ICE skin_irritation.xlsx(Data_invivo 시트, Draize 동물실험) 중\nMixture=="Mixture" & Endpoint=="EPA classification" 필터 \u2192 1,799건 원시 레코드'],
        ['제형 단위 집계', 'Formulation_ID 기준 그룹화, 성분 CASRN·함량%로 조성 재구성.\n자체 보강 데이터(AUG_ 접두사, v2/v3/v4, 118건) 병합'],
        ['구조 해결', '성분 CASRN/SMILES를 RDKit descriptor(desc_*, 217종)·Morgan fingerprint(fp_w_*,\n2048비트)·AromaticProportion에 매칭(casrn_descriptors_full.csv)'],
        ['제형 벡터화', '성분별 descriptor를 <b>wmean(투입 함량 가중평균)</b>으로 집계 \u2192 2266차원.\n+resolved_pct(구조 해결된 성분이 차지하는 함량 비율, 1차원) 추가 \u2192 <b>총 2267차원</b>'],
        ['그룹 식별자', 'comp_group_id = 해결된 성분 CASRN:함량 조성 해시(제품명이 달라도 같은\n조성이면 같은 그룹) \u2014 637개 고유 그룹, train/test 분리 시 그룹 누수 방지'],
        ['최종 표본', '662개 제형 중 3건 제외(MIX306/MIX442/MIX506) \u2192 <b>659건 사용</b>'],
    ]
    F.append(make_table(cellify(rows1), [26 * mm, 140 * mm]))

    F.append(Paragraph('3. 라벨 정의', styles['h2']))
    F.append(Paragraph(
        'EPA Classification Category 1(부식/불가역)~4(무자극) 중, production이 쓰는 <b>라벨B</b> 기준: '
        'Category 1·2·3 \u2192 양성(자극 있음, 1) / Category 4만 \u2192 음성(무자극, 0). '
        '양성 122건 / 음성 540건(662건 기준). (참고: 비교실험용으로만 쓰인 라벨A는 Category 1·2만 양성 \u2014 '
        'production 체크포인트는 라벨B만 사용, `binary_label`\u2261`binary_label_B` 100% 일치 확인됨.)',
        styles['body']))

    F.append(Paragraph('4. 모델 사양', styles['h2']))
    rows2 = [
        ['항목', '값'],
        ['알고리즘', 'RandomForestClassifier(n_estimators=500, class_weight="balanced")'],
        ['집계방식', 'wmean(농도가중평균) \u2014 min/top/max 전부 비교 검토 후 최종 유지(§6 참고)'],
        ['feature 선택', 'nz20: train(or trainval) 샘플 기준 nonzero 개수 20 미만인 컬럼 제거(fold-internal, test 정보 미사용)'],
        ['CV 전략', 'StratifiedGroupKFold(groups=comp_group_id) \u2014 동일 조성 제형이 train/test에 동시에 들어가지 않음'],
        ['threshold 선택', '배포 체크포인트: train/val/test 3분할 중 val에서만 balance-point(|Sens\u2212Spec| 최소) 선택,\ntest는 참고용으로만 사용(공식 보고 수치 아님) \u2014 처음부터 누수 없는 구조'],
        ['체크포인트', 'modeling/skin_rf/skin_rf_checkpoint.joblib (2026-11-08 교체, 이후 재학습 불필요 확인)'],
    ]
    F.append(make_table(cellify(rows2), [28 * mm, 138 * mm]))

    F.append(Paragraph('5. 성능 (공식 수치 — 반복 group 5-fold, 20회)', styles['h2']))
    F.append(Paragraph(
        'production 조합(라벨B·RandomForest·wmean·+resolved_pct)의 공식 재검증 수치 '
        '(`skin irritation/skin_official_grid_recheck.csv`, 20회 반복 pooled):', styles['body']))
    rows3 = [
        ['지표', '값', '참고'],
        ['MCC', '0.264 \u00b1 0.020', 'Matthews 상관계수, 0에 가까울수록 무작위'],
        ['Sensitivity', '0.668', '실제 자극성 제형을 찾아내는 비율'],
        ['Specificity', '0.667', '실제 무자극 제형을 맞히는 비율'],
        ['Sens\u2212Spec 격차', '0.009', '균형잡힌 모델일수록 0에 가까움'],
        ['ROC-AUC', '0.718', ''],
        ['n', '659건 (comp_group_id 637그룹)', ''],
    ]
    F.append(make_table(cellify(rows3), [34 * mm, 50 * mm, 82 * mm]))
    F.append(Spacer(1, 1.5 * mm))
    F.append(Paragraph(
        '참고(배포 체크포인트 단일분할, 공식 수치 아님): n=134 테스트셋에서 Sens 0.594 / Spec 0.676 / '
        '격차 0.083 / MCC 0.236 / ROC-AUC 0.713 \u2014 위 공식 반복수치와 같은 범위, 재현성 확인됨.',
        styles['bodysmall']))

    F.append(Paragraph('6. 집계방식(wmean vs top/min/max) 결정 이력', styles['h2']))
    F.append(bullets([
        '8차례에 걸친 비교(날짜·필터·그룹버그수정·threshold방식이 바뀔 때마다)에서 1위가 min\u2192top\u2192max\u2192'
        'wmean\u2192max로 계속 바뀜 \u2014 방법론에 매우 민감한 비교였음.',
        '가장 엄밀한 확정비교(outer5\u00d7inner3 nested CV, 동일 649건, 사전등록 기준)에서는 <b>max가 통계적으로 '
        '명확히 앞섬</b>(20/20회, p&lt;0.000001).',
        '그럼에도 <b>wmean 유지로 최종 결정</b> \u2014 max는 성분 농도를 전혀 반영하지 않는 구조(descriptor 컬럼별 '
        '독립 최댓값)라 극미량 성분이 예측을 좌우할 수 있고, EPA/GHS의 어떤 공식 혼합물 분류 원칙과도 대응되지 '
        '않음. wmean은 유일하게 EPA/GHS 농도가중 합산 원리와 대응되는 방식.',
        '관련 산출물(비교 스크립트·결과 CSV)은 "검토했으나 채택 안 함"의 증거로 레포에 보존.',
    ]))

    F.append(Paragraph('7. 2026-11-09 수정된 결함 (eye팀 역-감사 계기)', styles['h3']))
    rows4 = [
        ['결함', '수정 내용'],
        ['predict.py 라벨 표시 극성 오류', '양성을 "EPA I/II"로만 표시(III도 양성인데) \u2192 "I/II/III 상당" / "IV 상당"으로 수정'],
        ['0% 농도 falsy 버그', 'amount_pct=0을 기본값 1%로 대체하던 버그 \u2192 None만 기본값, 범위밖 값은 오류 처리'],
        ['comp_group_id 정규식 10배 오류', '".5%"를 "5%"로 잘못 파싱 \u2014 원자료 전체 스캔해 실제 영향 0건 확인(재학습 불필요)'],
        ['검증 노트북 버그 2건', 'resolved_pct 미제외 KeyError + 구버전 라벨(A) 정답컬럼 사용 \u2014 수정 후 결과 재확인'],
    ]
    F.append(make_table(cellify(rows4), [48 * mm, 118 * mm]))

    F.append(Paragraph('8. 외부 사례 비교 (탐색적, production 성능 주장 아님)', styles['h3']))
    F.append(Paragraph(
        '반려동물 벼룩/진드기 샴푸 5종의 공개 SDS로 탐색적 비교(`cosmetic_validation/`). 구조 해결 가능한 4건 '
        '기준, §7의 버그 수정 후 재실행 시 라벨 일치 4/4. <b>단 구조 해결률이 성분 1~4%뿐이고 n=4라 일반화 '
        '근거로 쓸 수 없음</b> \u2014 README 원칙대로 "화장품 전이학습/외부검증 성능"이 아니라 적용가능성 탐색 '
        '사례로만 취급.', styles['body']))

    F.append(Paragraph('9. 알려진 한계 (문서화됨, 미해결 \u2014 팀 결정 필요 시 재검토)', styles['h3']))
    F.append(bullets([
        'baseline(농도 미포함) 비교군의 일부 그룹(81그룹/272행)이 fold를 넘나드는 현상 \u2014 production(+농도) '
        '조합에는 영향 없음, baseline 비교 수치만 실제보다 유리하게 나왔을 가능성.',
        '비교·그리드 스크립트(설정 선택용)의 threshold가 평가용 OOF를 재사용하는 구조 \u2014 상대순위는 '
        '비교적 안정적이나 절대수치는 낙관적일 수 있음. 배포 체크포인트 자체의 threshold는 이 문제와 무관(§4).',
        '재현성: 스크립트 일부가 개인 절대경로에 의존, 증강 원자료(skin_irritation_최종후보.xlsx) 일부가 '
        '레포가 아닌 개인 OneDrive에만 존재.',
    ]))

    F.append(Paragraph('10. 산출물 경로', styles['h2']))
    F.append(Paragraph(
        '모델: modeling/skin_rf/skin_rf_checkpoint.joblib, train.py, evaluate.py, predict.py<br/>'
        '데이터: skin irritation/skin_vectorized.csv (662\u00d72295)<br/>'
        '공식 성능: skin irritation/skin_official_grid_recheck.csv<br/>'
        '집계방식 결정 기록: 세팅/wmean_vs_max_*.pdf(4건, 2026-11-09)<br/>'
        '진행 전체 이력: 세팅/진행상황.md', styles['mono']))

    doc.build(F)


if __name__ == '__main__':
    build_pdf(r'C:\Users\UserK\학술제\세팅\Skin_Irritation_모델_확정개요_20261109.pdf')
    print('done')
