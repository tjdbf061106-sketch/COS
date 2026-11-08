# -*- coding: utf-8 -*-
"""Eye production(최종/Eye_production_20261108/) 더블체크 보고서 PDF 생성.

2026-11-08: eye팀 요청으로 GitHub(최종/Eye_production_20261108/, 커밋 4746187)에 올라간
eye 농도피처 검증 결과물을 skin 작업과 별개 세션에서 독립적으로 재검증. 자체감사 문서가
중간에 지적한 결함 3가지가 최종본에서 실제로 고쳐졌는지 config/manifest를 직접 읽어 확인하고,
팀이 만들어둔 공식 재검증 스크립트(verify_results.py)를 직접 실행해 재현성을 확인했다.
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
    F.append(Paragraph('[Eye Irritation] Production 농도피처 검증 더블체크 보고서', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-11-08 · 대상: 최종/Eye_production_20261108/ (커밋 4746187)',
                        styles['subtitle']))

    F.append(Paragraph('1. 배경', styles['h2']))
    F.append(bullets([
        'eye팀이 skin 트랙과 별개로 `최종/Eye_production_20261108/` 경로에 농도피처(conc_log10_pct/'
        'conc_pct/conc_is_neat/conc_undeclared) 도입 검증 결과물을 GitHub에 업로드함.',
        'eye팀 요청으로, 작업한 당사자가 아닌 입장에서 GitHub에 커밋된 결과물만 보고 독립적으로 '
        '재현성·방법론 결함 여부를 더블체크함(코드를 새로 짜지 않고 팀이 이미 만들어둔 공식 재검증 '
        '스크립트 <b>verify_results.py</b>를 그대로 실행).',
    ]))

    F.append(Paragraph('2. 자체 감사에서 지적됐던 결함 3가지 \u2192 최종본에서 해결 확인', styles['h2']))
    F.append(Paragraph(
        'eye팀이 중간 시점에 스스로 작성한 `claude/production_pipeline_audit.md`가 아래 3가지 결함을 '
        '솔직하게 지적해뒀음. 최종 커밋의 config.json / final_verification.json을 직접 열어 각각 '
        '해결됐는지 확인함.', styles['bodysmall']))
    rows = [
        ['지적된 결함', '최종본 확인 결과'],
        ['농도 컬럼 4개를 CSV에 추가해도 feature_columns()가 '
         '무시해서 모델 입력에 실제로 반영 안 됨(2265차원 그대로)',
         'config.json: baseline 2265차원 / concentration <b>2269차원</b> \u2014 '
         '농도 4개 실제로 feature에 포함됨'],
        ['결측 농도값을 전체 데이터 중앙값으로 채우면 '
         'fold간 전처리 누수 발생', 'protocol 문자열에 '
         '"median fitted on <b>training rows only</b>" 명시'],
        ['증강(train-only) 67행이 평가·임계값선택 fold에 '
         '섞여 들어가 production 수치를 왜곡', 'final_verification.json: '
         '<b>augmentation_in_evaluation_or_calibration = 0</b>'],
    ]
    F.append(make_table(cellify(rows), [78 * mm, 86 * mm]))
    F.append(Spacer(1, 3 * mm))

    F.append(Paragraph('3. 공식 재검증 스크립트(verify_results.py) 직접 실행 결과', styles['h3']))
    F.append(Paragraph(
        '학습을 다시 돌린 게 아니라, 저장된 360개 outer-fold 예측 JSON에서 지표를 독립적으로 재집계해 '
        '보고서 수치와 일치하는지, 체크섬이 변조되지 않았는지, 누수가 없는지를 검사하는 eye팀 자체 '
        '검증 도구임.', styles['bodysmall']))
    rows2 = [
        ['검증 항목', '결과'],
        ['status', 'passed'],
        ['files_sha256_verified', '428개 전부 일치'],
        ['heldout_group_leakage', 'false (그룹 누수 없음)'],
        ['augmentation_evaluated', '0 (증강행이 평가에 섞이지 않음)'],
        ['recomputed_metrics_match / paired_deltas_match', 'true / true (보고서 수치 재현됨)'],
        ['outer_records_verified / oof_sets_verified', '360 / 72 (계획된 전체 분량 완료)'],
    ]
    F.append(make_table(cellify(rows2), [78 * mm, 86 * mm], head_bg=GOOD))
    F.append(Spacer(1, 3 * mm))

    F.append(Paragraph('4. 핵심 결과 \u2014 농도피처 추가 효과 (라벨B, 팀 실제 타겟)', styles['h2']))
    rows3 = [
        ['모델', 'baseline MCC', '농도피처 MCC', '\u0394MCC(3seed 평균\u00b1표준편차)'],
        ['XGBoost', '0.183', '0.224', '+0.041 \u00b1 0.075'],
        ['LightGBM', '0.184', '0.212', '+0.028 \u00b1 0.029'],
        ['RandomForest', '0.200', '0.215', '+0.015 \u00b1 0.034'],
        ['ExtraTrees', '0.195', '0.205', '+0.010 \u00b1 0.018'],
        ['Logistic Regression', '0.156', '0.160', '+0.004 \u00b1 0.005'],
        ['SVM', '0.131', '0.131', '\u2248 0.000 \u00b1 0.004'],
    ]
    F.append(make_table(cellify(rows3), [42 * mm, 30 * mm, 30 * mm, 62 * mm]))
    F.append(Spacer(1, 2 * mm))
    F.append(bullets([
        '라벨B(팀 실제 타겟)에서는 대부분 모델이 농도피처로 개선(+0.01~+0.04) \u2014 skin에서 확인한 '
        '"농도 정보를 쓰면 B라벨에서 더 도움됨" 패턴과 방향 일치.',
        '라벨A는 전 모델에서 효과가 거의 없음(LR +0.001, SVM \u22480, XGB \u22120.002) \u2014 역시 skin과 '
        '같은 "라벨 정의에 따라 농도 효과가 갈린다"는 패턴 재확인.',
        '방법론: outer 5-fold \u00d7 inner 3-fold \u00d7 3seed \u00d7 6모델 \u00d7 2arm(baseline/concentration) '
        '= 360 outer, 모델당 20후보 전체격자 \u00d7 360 = 7,200후보, 둘 다 100% 완료(결측/부분실행 없음).',
    ]))

    F.append(Paragraph('5. 중요 \u2014 "검증 완료"가 "배포 완료"는 아님', styles['h3']))
    F.append(Paragraph(
        'final_verification.json에 <b>"final_model_trained_or_deployed": false</b>가 명시되어 있음. '
        '이번 작업은 "농도피처를 추가하면 라벨B 기준 개선되는가"를 엄밀한 nested CV로 검증한 것이고, '
        '실제 배포용 eye 체크포인트 교체(= skin_rf_checkpoint.joblib에 해당하는 eye 쪽 모델 파일 생성/교체)는 '
        '<b>이번 작업 범위에 포함되지 않음</b>. 레포 전체에서 eye 전용 production 모델 파일 자체가 아직 없음'
        '(`modeling/`에 skin_rf/는 있지만 대응하는 eye_rf/류 디렉터리 없음).', styles['warn']))
    F.append(bullets([
        '별도 실험(03번, comp_group_id 재탐지 민감도 비교)은 eye팀 스스로 "중간 구간: 제공된 기준으로 '
        '확정 불가"로 보류 처리해둠(증강 67행이 CV에 포함돼 production train-only 검증을 대체하지 못한다는 '
        '한계도 함께 명시) \u2014 이 부분은 아직 결론이 난 상태가 아님.',
        '종합: 방법론·재현성은 깨끗하게 통과했고 농도피처 채택 방향은 타당하다고 확인되지만, '
        'skin처럼 실제 재학습\u00b7체크포인트 교체\u00b7predict.py 반영까지 간 마지막 단계는 eye 쪽에 아직 남아있음.',
    ]))

    doc.build(F)


if __name__ == '__main__':
    build_pdf(r'C:\Users\UserK\학술제\세팅\Eye_production_더블체크_보고서_20261108.pdf')
    print('done')
