# -*- coding: utf-8 -*-
"""팀원이 작성한 "wmean_vs_max_쉽게_설명한_판단보고서"(기존 skin팀 PDF 비판·교정)를
레포 코드/데이터로 직접 검증하고, 최종 합의 문구와 다음 실험 프로토콜을 기록하는 보고서.
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
    'good': ParagraphStyle('good', fontName='Malgun-Bold', fontSize=9.5, leading=14,
                            textColor=GOOD, spaceAfter=1.5 * mm),
    'warn': ParagraphStyle('warn', fontName='Malgun-Bold', fontSize=9, leading=13,
                            textColor=WARN, spaceAfter=1.5 * mm),
    'quote': ParagraphStyle('quote', fontName='Malgun', fontSize=9.5, leading=14,
                             textColor=colors.black, leftIndent=4 * mm,
                             borderColor=BLUE, borderWidth=0, spaceAfter=2 * mm),
    'cell': ParagraphStyle('cell', fontName='Malgun', fontSize=7.8, leading=10.8),
    'cellhead': ParagraphStyle('cellhead', fontName='Malgun-Bold', fontSize=7.9, leading=10.8,
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
    F.append(Paragraph('[Skin] wmean vs max \u2014 팀원 교정 반영 및 최종 합의', styles['title']))
    F.append(Paragraph('COS팀 · 학술제 · 2026-11-09 · 대상: 팀원 "wmean_vs_max_쉽게_설명한_판단보고서"(da3f4d9 검토) '
                        '+ 기존 skin팀 집계방식_wmean_vs_max_채택논쟁_정리_20261109.pdf', styles['subtitle']))

    F.append(Paragraph('1. 결론부터', styles['h2']))
    F.append(Paragraph(
        '팀원이 제출한 4가지 교정 지적을 레포 코드·데이터로 직접 재검증한 결과 <b>전부 사실로 확인됨</b>. '
        '기존 skin팀 보고서(da3f4d9)의 "1위가 계속 바뀌었다" 서술 중 일부가 부정확하거나 과장돼 있었음을 인정하고, '
        '팀원이 제시한 최종 판단과 문구를 그대로 채택함.', styles['body']))
    F.append(Paragraph(
        '"현재 결과는 max에 유리한 탐색적 근거다. 교체를 확정할 공정한 비교가 아직 부족하므로 wmean을 잠정 유지한다."',
        styles['good']))
    F.append(Paragraph(
        '피해야 할 표현(팀원 지적, 동의함): "wmean이 이미 더 안전하거나 우수하다고 검증됐다" / "다음 실험 한 번이면 '
        '영구 확정된다" \u2014 둘 다 현재 증거 수준을 넘어서는 과장.', styles['bodysmall']))

    F.append(Paragraph('2. 팀원의 4가지 교정 지적 \u2014 레포로 직접 재검증', styles['h2']))
    rows = [
        ['#', '팀원 지적', '검증 방법', '검증 결과'],
        ['1', '"wmean·top만 비교하다 max를 추가"한 것은\n두 방식의 순위가 뒤집힌 사례가 아님',
         'skin_official_grid_recheck.csv(32조건)에서\n라벨B/RF/+conc의 wmean·top 값 직접 대조',
         '<b>확인함.</b> wmean(0.264) &gt; top(0.251)로 두 방식의\n순위 자체는 안 바뀜. max가 새로 들어와 둘 다\n제친 것뿐 \u2014 "뒤집힘" 서술은 과장이었음'],
        ['2', '과거 비교(10/6, 11/7 초반)는\n0.5 고정 기준선이었지, "균형 임계값"이\n아니었음',
         '진행상황.md 2026-10-31/11-05 기록\n직접 확인("threshold 0.5 유지 확정/재확인")',
         '<b>확인함.</b> 균형(balance-point) threshold는\n2026-11-07~08에 와서야 도입됨. 기존 보고서가\n초기 비교 행에도 "균형"이라고 적은 것은 오기'],
        ['3', 'nz20 필터 외에 트리 모델의 설정(class_weight)도\n같이 바뀌어서, 필터 하나의 효과로\n단정할 수 없음',
         'eval_aggregation_method_comparison.py(6번) vs\nrf_et_aggregation_concentration_grid.py(7\u20138번)\n소스코드 직접 대조',
         '<b>확인함.</b> RandomForest의 class_weight가\n"balanced_subsample"(6번) \u2192 "balanced"(7\u20138번)로도\n바뀜 \u2014 nz20만의 순수효과로 볼 수 없음'],
        ['4', '보존된 15회 반복 결과에는 wmean·top만 있고\nmin/max는 없음 \u2014 max 우위의\n재현 근거로 쓸 수 없음',
         'skin_official_grid_recheck_15rep.csv\n컬럼(method) 직접 확인',
         '<b>확인함.</b> 15rep 파일은 method\u2208{wmean,top}뿐.\nmax 결과(0.302)는 20회 1회만 존재, 15\u219220회\n재현성 검증이 안 된 수치'],
    ]
    F.append(make_table(cellify(rows), [8 * mm, 42 * mm, 46 * mm, 50 * mm]))
    F.append(Spacer(1, 2 * mm))
    F.append(Paragraph(
        '네 지적 모두 소스 코드·CSV·진행상황 기록으로 직접 재현 가능했고 전부 사실과 일치함. '
        'skin팀 쪽에서 반박할 근거 없음 \u2014 교정 그대로 받아들임.', styles['bodysmall']))

    F.append(Paragraph('3. 그래도 "max가 현재 유리하다"는 사실 자체는 바뀌지 않음', styles['h3']))
    F.append(bullets([
        '위 4가지는 전부 <b>"왜 지금 바로 못 바꾸는가"의 근거를 보강</b>하는 교정이지, "max가 안 좋다"는 반증이 아님.',
        '가장 엄밀한 단일 비교(nz20+그룹버그수정+class_weight 통일, 20회 반복, 라벨B/RF/+농도)에서 '
        'max(MCC 0.302, 표본 649) &gt; wmean(MCC 0.264, 표본 659)인 것은 여전히 사실.',
        '다만 두 숫자가 <b>서로 다른 표본(649 vs 659)과 한 번의 OOF 기반 임계값 선택(평가셋으로 기준선도 고르고 '
        '같은 평가셋으로 채점)</b>에서 나왔다는 한계가 있어, "공정하게 비교해도 우세한지"는 아직 확인되지 않음.',
    ]))

    F.append(Paragraph('4. 합의된 다음 실험 프로토콜 (팀원 제안, 채택)', styles['h2']))
    rows2 = [
        ['단계', '내용'],
        ['① 동일 표본·동일 분할', 'wmean·max 둘 다 계산 가능한 동일 649건만 사용. 라벨B·RandomForest·농도피처 포함 고정.'],
        ['② 임계값 분리', '특성 선택·임계값 결정은 학습(연습) 폴드 안에서만 수행(outer 5-fold \u00d7 inner 3-fold). '
         '같은 조성 제형이 학습·평가 양쪽에 섞이지 않게 comp_group_id 기준 그룹 분리 유지. 두 방식에 동일 규칙 적용.'],
        ['③ 전체 + 다성분 하위집단', '전체표본 점수뿐 아니라 다성분 제형만의 하위집단 결과도 함께 비교(단일성분은 '
         '집계방식 차이가 수학적으로 사라지므로).'],
        ['④ 사전 결정 기준', '결과를 보기 전에 "바꿀 만큼의 개선" 기준(필요 MCC 개선폭, Sens/Spec 허용범위)을 미리 '
         '정함. 평균 차이만 단순 대조해 사후적으로 결론 내리지 않음.'],
        ['⑤ 20회 반복의 의미 재정의', '20회 반복은 "새 표본 20개"가 아니라 "같은 자료를 20번 다시 나눈 것" \u2014 '
         '이번 결과도 "영구 확정"으로 표현하지 않음.'],
    ]
    F.append(make_table(cellify(rows2), [36 * mm, 130 * mm]))
    F.append(Spacer(1, 2 * mm))
    rows3 = [
        ['비교 결과', '결정'],
        ['max가 사전에 정한 개선 기준을 충족', 'max 전환 검토 \u2014 train.py/evaluate.py/predict.py 집계로직 재작성, 재학습, 체크포인트 교체'],
        ['차이가 불확실하거나 개선 기준 미달', 'wmean 유지 \u2014 현재 근거로는 변경할 이점이 충분하지 않음으로 기록하고 종료'],
    ]
    F.append(make_table(cellify(rows3), [60 * mm, 106 * mm]))

    F.append(Paragraph('5. 현재 상태', styles['h2']))
    F.append(bullets([
        '<b>production은 wmean 그대로 유지</b>(코드 변경 없음). 위 프로토콜의 실제 실행(outer5\u00d7inner3 nested CV '
        '스크립트 작성·실행)은 아직 안 함 \u2014 이것이 유일하게 남은 다음 작업.',
        '세팅/집계방식_wmean_vs_max_채택논쟁_정리_20261109.pdf(기존 skin팀 보고서)는 "1위가 계속 바뀌었다"는 '
        '서술이 §2의 교정 1·2·3을 반영하지 않은 상태로 남아있음 \u2014 이 문서가 그 교정판 겸 최종 합의 기록.',
        '세팅/진행상황.md가 다른 팀원 작업으로 dirty하여 이번 합의 내용은 아직 그 파일에 병합되지 않음(기존 '
        '병합 대기 항목에 합류).',
    ]))

    doc.build(F)


if __name__ == '__main__':
    build_pdf(r'C:\Users\UserK\학술제\세팅\wmean_vs_max_팀원교정반영_최종합의_20261109.pdf')
    print('done')
