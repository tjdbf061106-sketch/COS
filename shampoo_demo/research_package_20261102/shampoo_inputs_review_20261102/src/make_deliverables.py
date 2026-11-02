"""Create an Excel review workbook, standalone figures and a Korean report."""
from __future__ import annotations

import base64
import html
import importlib.metadata
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

from build_review import ROOT, digest, read_json, write_json


def table(data):
    return data.to_html(index=False,border=0,escape=True,na_rep='미확인',float_format=lambda x:f'{x:.3f}')


def img(path):
    return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii')


def figures(review,curves,representatives):
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    groups=sorted(review.polymer_thickener_group.unique())
    colors=dict(zip(groups,plt.get_cmap('tab10').colors))
    fig,ax=plt.subplots(figsize=(11,6.5),constrained_layout=True)
    for group,g in review.groupby('polymer_thickener_group'):
        ax.errorbar(g.turbidity_ntu,g.eta100_mPa_s,xerr=g.turbidity_error_as_reported,fmt='o',ms=5,
                    alpha=.8,capsize=2,color=colors[group],label=group)
    annotate=set(representatives.sample_id)|set(review.loc[review.flag_turbidity_plus_reported_error_crosses_target,'sample_id'])
    for row in review.itertuples(index=False):
        if row.sample_id in annotate:ax.annotate(f'S{row.sample_id}',(row.turbidity_ntu,row.eta100_mPa_s),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.axvline(100,linestyle='--',color='#bd6533');ax.axhline(100,linestyle='--',color='#bd6533')
    ax.set(yscale='log',xlabel='Turbidity (NTU); bars = source-reported error',ylabel='Viscosity at 100 s^-1 (mPa.s)',title='44 observed starting formulations | no new predictions')
    ax.legend(fontsize=8,loc='upper right');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'observed44_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,17),constrained_layout=True)
    for ax,group in zip(axes.flat,groups):
        g=curves[curves.polymer_thickener_group==group]
        for sid,curve in g.groupby('sample_id'):
            ax.plot(curve.shear_rate_s_inv,curve.mean_viscosity_mPa_s,lw=1.2,alpha=.85,label=f'S{sid}')
        ax.set(xscale='log',yscale='log',xlabel='Shear rate (s^-1)',ylabel='Viscosity (mPa.s)',title=group)
        ax.axvline(100,color='#777',linestyle=':',alpha=.5);ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.15)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.03,.9,'Original mean flow curves\n\n25 C; same study preparation protocol\n\nSource rheology classes retained\n\nDifferent recipes are not technical replicates\n\nNo interpolation beyond measured rates',va='top',fontsize=13)
    fig.suptitle('Complete flow-curve review | 44 formulations / 7 represented subsystems',fontsize=16)
    fig.savefig(out/'observed44_flow_curves.png',dpi=140);fig.savefig(out/'observed44_flow_curves.pdf');plt.close(fig)


def workbook(review,representatives,summary):
    registry=pd.read_csv(ROOT/'data/ingredient_registry_reviewed.csv')
    evidence=read_json(ROOT/'sources/supplier_evidence.json')
    source_rows=[{'자료':d['title'],'문서시점':d['document_date'],'페이지':d.get('page','웹페이지'),'URL':d['url'],'사용범위':d['use']} for d in evidence['documents']]
    short_names={'sample_id':'제형_ID','stock_recipe':'상업원료_투입량','polymer_grade':'고분자_원료등급','thickener_grade':'증점제_원료등급',
                 'eta10_mPa_s':'점도_10s-1_mPa.s','eta100_mPa_s':'점도_100s-1_mPa.s','eta500_mPa_s':'점도_500s-1_mPa.s',
                 'turbidity_ntu':'탁도_NTU','turbidity_error_as_reported':'탁도_원문오차값','turbidity_relative_error':'탁도_상대오차',
                 'rheology_type_source':'원문_유동유형','nearest100_cv':'100근처_측정CV','added_water_balance_pct_estimate':'조정전_물잔량_추정pct'}
    sheets={
        '읽는법':pd.DataFrame([
            ['범위','44개 처방 정밀 검토와 전체 812개 입력 보강. 모델 재학습/새 예측 없음.'],
            ['선정목표','실측 안정, eta100>=100 mPa.s, NTU<=100. 기존 예시 목표를 유지.'],
            ['통계해석','44개는 결과에 따라 고른 기술통계용 부분집합. 전체 학습셋으로 대체하지 않음.'],
            ['오차해석','탁도 원문오차와 점도 std는 그대로 기록. CI/SE/보간값의 정확한 불확실성으로 재해석하지 않음.'],
            ['원료활성분','원문 범위와 공급사 비교 범위를 구분. 실험 당시 배치 규격은 미확인.'],
            ['물','원료합의 잔량 + 일부 원료 내부 물. 최종 적정액과 미공개 원료 물은 미복원.'],
            ['입력세트',str(summary['feature_set_sizes'])],
            ['검증','입력값 보존, 희석정보, 단위, 라벨 누출, 범위 시나리오, 원본 곡선 재계산 검증.']
        ],columns=['항목','설명']),
        '44개_비교':review.rename(columns=short_names),
        '44개_원료투입량':pd.read_csv(ROOT/'review/formulations44_stock_amounts.csv'),
        '원료별_함량':pd.read_csv(ROOT/'review/formulations44_ingredients_long.csv'),
        '유변곡선_원측정':pd.read_csv(ROOT/'review/formulations44_flow_curves.csv'),
        '대표_검토제형':representatives.rename(columns=short_names),
        '동일조합_차이':pd.read_csv(ROOT/'review/same_combination_pairs.csv'),
        '원료_규격검토':registry,
        'PK45_민감도':pd.read_csv(ROOT/'data/pk45_sensitivity_effects_812.csv').query('sample_id in @review.sample_id'),
        '입력_정의':pd.read_csv(ROOT/'data/feature_dictionary.csv'),
        '입력_진단':pd.read_csv(ROOT/'data/feature_diagnostics_not_selection.csv'),
        '공급사_출처':pd.DataFrame(source_rows),
    }
    path=ROOT/'샴푸_44개처방비교_입력보강.xlsx'
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for name,data in sheets.items():
            data.to_excel(writer,sheet_name=name,index=False)
            ws=writer.sheets[name];ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
            for cell in ws[1]:
                cell.fill=PatternFill('solid',fgColor='174858');cell.font=Font(color='FFFFFF',bold=True);cell.alignment=Alignment(wrap_text=True,vertical='top')
            ws.row_dimensions[1].height=42
            for col in ws.columns:
                header=str(col[0].value)
                width=55 if any(t in header.lower() for t in ['note','definition','recipe','설명','url','combination','scope']) else min(28,max(15,len(header)*.8))
                ws.column_dimensions[col[0].column_letter].width=width
                for cell in col[1:]:
                    cell.alignment=Alignment(vertical='top',wrap_text=isinstance(cell.value,str))
                    if isinstance(cell.value,float):cell.number_format='0.000'
                    if header.startswith('flag_') and cell.value is True:cell.fill=PatternFill('solid',fgColor='FFF0C2')
    return path


def main():
    summary=read_json(ROOT/'summary.json');verification=read_json(ROOT/'verification.json')
    review=pd.read_csv(ROOT/'review/formulations44_comparison.csv')
    reps=pd.read_csv(ROOT/'review/subsystem_review_representatives.csv')
    curves=pd.read_csv(ROOT/'review/formulations44_flow_curves.csv')
    registry=pd.read_csv(ROOT/'data/ingredient_registry_reviewed.csv')
    featuresets=read_json(ROOT/'data/feature_sets.json')
    figures(review,curves,reps);book=workbook(review,reps,summary)
    concise=review[['sample_id','polymer_grade','thickener_grade','eta10_mPa_s','eta100_mPa_s','eta500_mPa_s','turbidity_ntu','turbidity_error_as_reported','rheology_type_source']]
    flagged=review[review.flag_turbidity_relative_error|review.flag_viscosity_near_target|review.flag_turbidity_near_target][['sample_id','eta100_mPa_s','turbidity_ntu','turbidity_relative_error','flag_viscosity_near_target','flag_turbidity_near_target']]
    rep_cols=['sample_id','polymer_grade','thickener_grade','eta100_mPa_s','turbidity_ntu','nearest100_cv']
    def sidlist(col):return ', '.join('S'+str(int(x)) for x in review.loc[review[col],'sample_id'])
    mainfindings=[
        f"44개 제형의 원료 행 176개와 전체 유변곡선을 대조했습니다. 원문 유동유형은 전단박화(SHEAR-THIN) 41개, 뉴턴 유체 3개입니다.",
        f"탁도 원문오차/측정값이 30% 이상인 제형: {sidlist('flag_turbidity_relative_error')}. 이 기준은 검토 표시이며 자동 제외 기준이 아닙니다.",
        f"점도 목표 100 mPa·s의 20% 이내 위쪽에 있는 제형: {sidlist('flag_viscosity_near_target')}. 탁도 100 NTU의 20% 이내 아래쪽에 있는 제형: {sidlist('flag_turbidity_near_target')}.",
        "S335의 탁도는 86, 원문 오차값은 39입니다. 두 수를 더한 125 NTU는 예시 목표 100을 넘습니다. 원문 오차를 사용한 민감도 표시이며 합성 신뢰구간 판정이 아닙니다.",
        "고분자·증점제 조합은 7개가 포함됩니다. 각 조합에서 측정오차가 작은 제형 하나씩 골라 검토 출발점으로 정리했습니다. 최고 성능 처방, 생산 적합성 또는 안전성 순위가 아닙니다.",
        "같은 원료 조합 안의 처방 차이 25쌍을 비교했습니다. 서로 다른 처방은 기술 반복시험이 아니며, 관측 차이를 원료 변경의 인과 효과로 해석하지 않습니다.",
    ]
    source_urls={d['id']:d['url'] for d in read_json(ROOT/'sources/supplier_evidence.json')['documents']}
    html_text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>샴푸 44개 처방 검토와 입력 보강</title>
<style>body{{font-family:Malgun Gothic,Arial,sans-serif;max-width:1240px;margin:38px auto;padding:0 24px;color:#18323d;line-height:1.75;background:#f8fafb}}h1,h2{{line-height:1.35}}h2{{margin-top:38px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{padding:18px;border:1px solid #d4e1e7;border-radius:12px;background:white}}.big{{font-size:30px;font-weight:bold;color:#167065}}.note{{background:#fff4dd;border-left:4px solid #b88623;padding:14px 18px}}table{{border-collapse:collapse;background:white;font-size:12px;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dce5e8}}.scroll{{overflow-x:auto}}img{{width:100%;height:auto}}a{{color:#176b98}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}}}</style>
<h1>샴푸 44개 처방 검토 · 물성 입력 보강</h1><p>요청한 2·3번 작업: 기존 실측 출발 제형 정밀 검토와 812개 전체 제형의 물성 입력 정비입니다. 모델 재학습이나 새로운 성능 주장은 포함하지 않습니다.</p>
<div class="cards"><div class="card">실측 처방 검토<div class="big">44개</div>원료 행 176개 / 고분자·증점제 7조합</div><div class="card">입력 데이터<div class="big">812 × 114</div>원료 18 → 요약 61 → 상호작용 114</div><div class="card">검증<div class="big">{verification['status']}</div>기존 기준선·측정 타깃·평가 분할 보존</div></div>
<p><a href="{book.name}">엑셀 비교표 열기</a> · <a href="README.md">재현 방법</a> · <a href="data/feature_sets.json">입력 세트</a></p>
<h2>실측 제형에서 확인한 점</h2><ul>{''.join('<li>'+html.escape(s)+'</li>' for s in mainfindings)}</ul>
<img src="{img(ROOT/'figures/observed44_comparison.png')}" alt="44 observed formulations with source reported turbidity error">
<h2>원료 규격 대조</h2><div class="note"><strong>Dehyton PK 45 활성분 차이</strong><p>연구 CSV에는 44–46%, BASF 공개 카탈로그에는 37–41%가 기재되어 있습니다. 이 원료는 812개 중 140개, 검토 대상 44개 중 14개에 사용됩니다. 원산지·원료 등급·실험 당시 배치가 동일한지 확인되지 않아 원문을 덮어쓰지 않았습니다. 별도 비교 시나리오에서 제형 전체 활성분 하한은 최대 0.931%p, 상한은 최대 0.665%p 낮아집니다. 이는 규격 가정의 영향이며 실제 처방 변화가 아닙니다.</p><a href="{source_urls['BASF_SURFACTANTS']}">BASF 자료 6쪽</a></div>
<p>Dehyquart A-CA의 연구 원문 INCI에는 Potassium이 있으나 <a href="{source_urls['BASF_ACA']}">BASF 제품 페이지</a>는 Cetrimonium Chloride로 표기합니다. 원문 명칭과 검토 명칭을 두 열로 보존했습니다. Dehyton AB 30 대표 SMILES의 전하 문제도 검토 항목으로 남겼으며, 이번 보강 입력은 이 SMILES로 분자 특성을 계산하지 않습니다.</p>
<p>같은 Polyquaternium-7이라도 Dehyquart CC7 Benz와 Salcare Super 7은 별도 등급으로 유지했습니다. 원문 슬라이드에서 분자량과 공중합 조성이 다르기 때문입니다. Luviquat Excellence의 <a href="{source_urls['BASF_LUVIQUAT']}">2012년 공급사 자료</a>에 있는 고형분 38–42%는 별도 참고 데이터로 기록했습니다. 고형분을 순수 고분자 활성분으로 간주하거나, pH 7 전하밀도를 pH 5.8에 그대로 적용하지 않았습니다.</p>
<h2>보강한 입력</h2><table><tr><th>세트</th><th>특성 수</th><th>포함 내용</th></tr><tr><td>stock18</td><td>18</td><td>원본 상업용 원료 투입 질량%</td></tr><tr><td>physics_core</td><td>61</td><td>역할·계면활성제 종류별 총량, 활성분 상·하한, 배합 균형, 공개된 염·산·보존제 기여분, 부분 물수지, 원료 등급</td></tr><tr><td>physics_interactions</td><td>114</td><td>종류 간, 고분자–증점제 등급 간, 계면활성제 종류–고분자/증점제 간 투입량 곱</td></tr></table>
<p>원본 범위를 사용하는 입력과 PK45 공급사 범위를 적용한 입력을 모두 저장했습니다. 활성분 범위를 중간값 하나로 바꾸지 않았습니다. 범위의 각 끝점은 부분 기여분이며, 모든 원료의 상한을 동시에 적용한 완전한 조성이나 확률구간이 아닙니다. 원료 내부 물·염·적정액이 완전히 공개되지 않았으므로 총 활성분·총 염농도·이온강도·최종 물 함량이 확정된 입력이라고 부르지 않습니다.</p>
<p>예측용 열에는 시험 결과, 실측 유동유형, 장비, 제형 ID·순번을 넣지 않았습니다. 44개만으로 입력을 만들지 않고 전체 812개를 처리했으며, 회귀의 결측 물성을 0으로 채우지 않았습니다. 변수의 상수 여부와 빈도 진단은 참고용이고 이 단계에서 전체 자료 기준으로 변수를 제거하지 않았습니다.</p>
<h2>조합별 검토 출발점</h2><p>원문 탁도 상대오차 → 100 s⁻¹ 근처 측정 CV → ID 순으로 선정한 기술적 검토 표본입니다. S337처럼 목표 경계에 가까운 표본도 표시를 유지합니다.</p><div class="scroll">{table(reps[rep_cols])}</div>
<h2>측정오차·목표 경계 검토</h2><div class="scroll">{table(flagged)}</div>
<h2>전체 유변곡선</h2><p>10·100·500 s⁻¹의 점도, 구간 기울기와 점도비를 계산했습니다. 보간은 원측정 범위 안에서만 수행합니다. 곡선의 표준편차는 반복 측정 원자료가 없으므로 표준오차나 보간값의 정확한 신뢰구간으로 변환하지 않았습니다.</p><img src="{img(ROOT/'figures/observed44_flow_curves.png')}" alt="Original flow curves for 44 formulations by subsystem">
<h2>44개 전체 비교</h2><div class="scroll">{table(concise)}</div>
<h2>범위와 다음 평가</h2><p>온도 25℃, 목표 pH 5.8±0.2, 혼합 35분·360 rpm 및 최종 36시간 관찰은 연구의 공통 프로토콜입니다. 제형별 최종 pH·원료 배치·적정액 투입량은 미보고 상태로 남겼습니다. 이번 결과는 장기 안정성이나 피부·안자극성 판단이 아닙니다.</p><p>다음 단계는 보존된 분할에서 18/61/114 입력과 활성분 두 시나리오를 비교하는 것입니다. 변수 선택·스케일링·모델 조정은 학습 폴드 내부에서만 수행해야 합니다. 44개는 목표 결과에 따라 선택된 검토 표본이므로 전체 학습/평가 모집단을 대신할 수 없습니다.</p>
<p>GitHub의 변경된 연구 방향(f049106)을 확인했습니다. 화장품 자극성 전이학습을 수행하지 않는다는 범위에 맞추었으며, 기존 skin·eye와 샴푸 기준선 파일을 수정하지 않았습니다. 이번 산출물은 로컬 폴더에 저장했습니다.</p>
<h2>검증</h2><ul>{''.join('<li>'+html.escape(s)+'</li>' for s in verification['checks'])}</ul><p><a href="verification.json">검증 결과</a> · <a href="review_protocol.json">분석 기준</a> · <a href="sources/supplier_evidence.json">공급사 근거 기록</a></p>
<p>샴푸 원자료: <a href="https://www.nature.com/articles/s41597-024-03573-w">Chitre et al. (2024)</a>, <a href="https://springernature.figshare.com/articles/software/Liquid_Formulations_Dataset/25451878">Figshare 데이터</a>, <a href="https://springernature.figshare.com/articles/dataset/Surfactants_Information/25451926">원료 정보</a>, <a href="https://springernature.figshare.com/articles/figure/Structures_of_Formulations_Ingredients/25451929">원료 구조</a>.</p></html>'''
    (ROOT/'report.html').write_text(html_text,encoding='utf-8')
    readme=f'''# 샴푸 실측 44개 검토와 입력 보강

요청한 2번(실측 제형 검토), 3번(입력 보강)을 완료한 별도 분석 폴더입니다. 기존 기준선은 그대로 유지했습니다. **모델을 재학습하지 않았으며 성능 개선을 주장하지 않습니다.**

## 열어볼 파일

- [보고서](report.html): 전체 결과와 그림이 포함된 단독 HTML.
- [엑셀 비교표]({book.name}): 44개 비교, 원료 투입량, 176개 원료 행, 원측정 곡선, 원료 규격, 입력 정의, 근거 출처.
- [전체 곡선 PDF](figures/observed44_flow_curves.pdf).
- [입력 세트 정의](data/feature_sets.json), [검증 결과](verification.json).

## 핵심 결과

- 실측 제형 44개, 원료 행 176개, 고분자·증점제 7조합 검토. 전단박화 41개, 뉴턴 유체 3개(원문 분류 유지).
- 탁도 상대오차 30% 이상: S17, S153, S154, S335, S639, S736. 이는 검토 플래그이며 자동 제외 사유가 아닙니다.
- S335의 탁도 86 + 원문오차 39 = 125 NTU로, 예시 목표 100을 넘습니다. S337·S349는 점도 목표 100 mPa·s 근처입니다.
- 조합별 검토 출발 제형 7개: {', '.join('S'+str(x) for x in summary['representative_sample_ids'])}. 선정 기준은 원문 탁도 상대오차, 100 s⁻¹ 근처 CV, ID 순입니다. 최적 처방/안전성 순위가 아닙니다.
- 전체 812개에 18/61/114개 입력을 생성했습니다. 44개만을 모델 학습에 사용하지 않습니다.
- Dehyton PK 45 활성분: 연구 원문 44–46%와 BASF 공개 자료 37–41%가 다릅니다. 140/812개, 14/44개 제형에 영향이 있어 별도 시나리오로 분리했습니다. 배치·원산지·등급 동등성은 미확인입니다.

## 실행

```powershell
python -m pip install -r requirements.txt
python src/build_review.py
python -m unittest discover -s tests -v
python src/verify_review.py
python src/make_deliverables.py
```

같은 저장소의 `shampoo_physics_20261102/` 기준선과 나란히 둡니다. `inputs/`에는 필요한 원본 사본을 포함했고, 검증은 기존 기준선 파일이 보존됐는지도 확인합니다. 공급사 PDF 전체는 재배포하지 않으며, 근거 숫자·페이지·URL·확인한 문서 해시를 `sources/`에 기록했습니다.

## 데이터 파일

| 파일 | 내용 |
|---|---|
| `review/formulations44_comparison.csv` | 제형별 처방, 10/100/500 s⁻¹ 점도, 측정오차, 검토 플래그 |
| `review/formulations44_stock_amounts.csv` | 원료 18개 투입량과 조정 전 물 잔량 추정 |
| `review/formulations44_ingredients_long.csv` | 성분별 상업 원료 투입량·공개 활성분 범위·등급 |
| `review/formulations44_flow_curves.csv` | 원래의 전단속도·평균 점도·표준편차 |
| `review/same_combination_pairs.csv` | 같은 원료 조합 안의 처방 차이 25쌍; 인과/반복시험 비교 아님 |
| `data/ingredient_registry_reviewed.csv` | 18개 원료의 원문 명칭, 검토 명칭, 범위, 등급, 공급사 차이 |
| `data/features_source_reported_812.csv` | ID + 114개 입력. 원문 범위 유지 |
| `data/features_pk45_supplier_sensitivity_812.csv` | 같은 구조의 PK45 공급사 범위 민감도 시나리오 |
| `data/feature_dictionary.csv` | 모든 입력의 단위·공식·한계 |
| `data/feature_sets.json` | 18/61/114개 열 목록. ID 제외 |
| `data/targets_and_groups_812.csv` | 라벨·결측·그룹을 별도 파일로 보존 |
| `data/outer_folds_preserved.csv` | 기존 3회 5-fold와 8개 subsystem 제외 분할, 바이트 단위 보존 |
| `data/supplier_context_not_core_features.csv` | Luviquat 고형분 등 과거 공급사 참고값. 주 입력에 포함하지 않음 |

## 해석·입력 규칙

- `source-claimed active`는 공개 CSV 열에서 환산한 값입니다. 실험 당시 배치의 활성분 실측이 아닙니다. 원료 투입량 10%에 활성분 44–46%면 제형 기여분은 4.4–4.6%입니다.
- Dehyton PK45 시나리오는 활성분 관련 열만 바꿉니다. 원료 투입량·실측 결과·원문의 물 함량은 그대로이며, 새로운 완전 조성으로 복원한 것이 아닙니다.
- 물·염·산·보존제 열은 공개된 부분 기여분입니다. 누락 성분을 0이라고 단정하지 않습니다. 고분자/증점제 내부 물, 적정액, 기타 미공개 성분은 미확인으로 남깁니다.
- 5종 계면활성제 분류는 저자의 타입 구분입니다. pH 5.8에서 측정한 전하량이 아닙니다. 종류별 투입량 곱은 상호작용을 학습하기 위한 사전 지정 대리변수이고 물리적 상호작용 실측이 아닙니다.
- Polyquaternium-7 두 등급은 분자량·공중합 조성이 다르므로 합치지 않습니다. 고분자 미확인 분자량을 임의 수치로 채우지 않습니다.
- Luviquat 38–42%는 2012년 자료의 고형분 참고값입니다. 순수 고분자 활성분과 같다고 가정하지 않고 주 입력에서 제외했습니다.
- Dehyquart A-CA의 검토 INCI는 공급사 표기 Cetrimonium Chloride입니다. 원문의 Potassium 포함 명칭은 별도 열로 보존했습니다. Dehyton AB30 대표 SMILES 전하 이슈는 아직 확정 수정하지 않았습니다.
- 투입량을 공개 성분 합으로 재정규화하지 않습니다. ID, 라벨, 물성 측정값, 장비, 표본 순서, 목표 충족 여부는 예측 입력에 없습니다.
- 희석·단위 환산·목표값 변경·grade 구분·0분모·보간 범위와 원본 보존을 검증했습니다. 모델 학습에 맞춘 정규화·결측대체·변수 선택은 이 단계에서 하지 않았습니다.
- 모든 통계·플래그는 44개에 대한 기술적 검토입니다. 이 제형들은 실측 물성으로 선택됐으므로 외부검증셋, 무작위 표본 또는 전체 학습셋이 아닙니다.

## 다음 단계

보존한 분할에서 18/61/114개 입력과 활성분 두 시나리오를 비교합니다. 모든 선택과 스케일링은 학습 폴드 안에서 수행합니다. 목표에 맞는 제형만 골라 학습하거나, 전체 자료의 변수 빈도로 먼저 필터링하지 않습니다. 현재 결과는 입력 준비 완료이지 정확도 향상 확인이 아닙니다.

GitHub `f049106`의 연구 방향을 반영했습니다. 화장품 자극성 전이학습과 피부/안자극 점수 생성은 수행하지 않았습니다. 기존 skin·eye, 기준선, 팀 문서는 수정하지 않았고 이번 자료를 GitHub에 게시하지 않았습니다.

## 출처

- [Chitre et al. (2024)](https://www.nature.com/articles/s41597-024-03573-w) 및 [샴푸 데이터](https://springernature.figshare.com/articles/software/Liquid_Formulations_Dataset/25451878), [원료 정보](https://springernature.figshare.com/articles/dataset/Surfactants_Information/25451926), [원료 구조](https://springernature.figshare.com/articles/figure/Structures_of_Formulations_Ingredients/25451929). Figshare 원본은 CC0.
- [BASF Standard Surfactants & Alcohols, 6쪽]({source_urls['BASF_SURFACTANTS']}). 문서 발행시점·실험 당시 배치 동등성은 미확인.
- [BASF Luviquat Polymer Grades, 2012-05, 4쪽]({source_urls['BASF_LUVIQUAT']}).
- [BASF Dehyquart A-CA 제품 정보]({source_urls['BASF_ACA']}).
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    packages=['numpy','pandas','matplotlib','openpyxl']
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in packages)+'\n',encoding='utf-8')
    manifest={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=ROOT/'manifest.json'}
    write_json(ROOT/'manifest.json',{'files':manifest,'excluded':['manifest.json','__pycache__']})
    print({'report':'report.html','workbook':book.name,'manifest_files':len(manifest)})


if __name__=='__main__':main()
