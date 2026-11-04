"""Build a Korean Word/HTML feasibility review from the local, read-only audit."""
from pathlib import Path
import hashlib
import html
import json
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET

import pandas as pd

OUT = Path(__file__).resolve().parent
summary = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
metrics = pd.read_csv(OUT / "classification_metrics_summary.csv")
blocks = []


def title(text): blocks.append(("title", text))
def heading(text): blocks.append(("heading", text))
def para(text): blocks.append(("paragraph", text))
def table(headers, rows): blocks.append(("table", (headers, rows)))


title("신규 원료 편입 프로토콜 — 실현 가능성 검토")
para("검토일: 2026-11-02 | 대상: 신규원료_적용범위_편입_프로토콜.docx 및 현재 shampoo_physics 데이터·모델·앱 코드")
para("판단: 원료 정보를 기록하고, 실측 자료를 추가한 뒤 제한된 범위에서 예측을 지원하는 절차는 구현 가능합니다. 다만 현재 문서는 연구 설계 초안입니다. 신규 원료의 실제 편입 성공, 최소 실험 수, 예측 성능을 아직 입증한 것은 아닙니다. 실험을 할 수 없다면 학술제의 핵심은 공개 실측 자료의 비교 연구와 적용범위 관리로 정하는 것이 현실적입니다.")
para("이미 812개 제형, 세 물성 모델, 그룹 평가 예측, 44개 실측 출발 처방이 있어 결과물이 없는 상태는 아닙니다. 남은 핵심은 성능의 한계를 숨기지 않고, 사전에 정한 비교 질문에 답하는 것입니다. 높은 성능이나 학술제 평가 결과를 보장할 수는 없습니다.")

heading("1. 현재 근거로 가능한 범위")
table(["항목", "판단", "조건·한계"], [
    ["원료 카드·규격·미확인 정보 관리", "바로 구현 가능", "원료·공급사·등급·배치·활성분 근거를 분리하여 기록"],
    ["기존 18개 상용 원료의 탐색적 물성 예측", "기준선 존재", "단기 안정성 및 실측 안정 제형에 조건부인 점도·탁도; AD의 성능 보증은 미검증"],
    ["미지원 원료·조건의 예측 보류", "구현 가능, 보완 필요", "현재 앱은 일부 미확인 원료를 연결하고 범위 밖에서도 수치를 출력"],
    ["보강 입력과 원료 정보 불확실성의 영향 비교", "현재 데이터로 연구 가능", "18/61/114개 입력이 준비됐지만 성능 비교는 아직 수행하지 않음"],
    ["신규 원료를 실제 예측 지원 범위에 등록", "실측 데이터가 있어야 검증 가능", "재학습용과 최종 평가용 새 처방을 구분; 알려진 원료 모의 실험만으로 대체 입증 불가"],
    ["최소 실험으로 성공하는 능동학습", "후속 연구", "탐색 비용 대비 이득을 같은 실험 예산의 무작위·공간 충진 선택과 비교해야 함"],
])
para("원 논문도 18개 원료, 812개 제형과 294개 안정 제형을 제시하고, 제한된 원료 및 구조 정보 때문에 학습 범위 밖 예측이 어려워진다고 설명합니다. 이는 이 프로젝트가 신규 원료 전체로 일반화됐다는 근거가 아닙니다. [1]")

heading("2. 직접 다시 확인한 데이터와 성능")
para("현재 데이터는 안정 294개·불안정 518개입니다. 점도 타깃은 293개, 탁도 타깃은 294개입니다. 모든 입력·학습 코드의 training_lock 해시가 일치하며, 저장된 OOF 예측에서 기존 분류 지표를 다시 계산해 일치함을 확인했습니다.")
metric_rows = []
for model, name in [("linear", "선형 모델"), ("extra_trees", "Extra Trees")]:
    row = metrics[(metrics.scheme == "combination_holdout") & (metrics.model == model)].iloc[0]
    metric_rows.append([name] + [f"{row[m+'_mean']:.3f} ± {row[m+'_sd']:.3f}" for m in ["Sensitivity", "Specificity", "MCC", "ROC-AUC"]])
table(["모델", "Sensitivity", "Specificity", "MCC", "ROC-AUC"], metric_rows)
para("양성=안정, 임계값=0.5, 원료 조합으로 분리한 3회 5-fold OOF 결과의 평균±반복 SD입니다. SD는 신뢰구간이 아닙니다. 양성을 불안정으로 표현하면 민감도와 특이도를 서로 바꾸고, 해당 점수·판정 정의도 일치시켜야 합니다.")
para("Extra Trees는 안정 제형 탐지율 약 55%, 불안정 제형 배제율 약 87%로 균형이 충분하다고 보기 어렵습니다. 선형 모델은 약 61%·83%로 간격이 작지만 안정 제형을 여전히 많이 놓칩니다. MCC·AUC 한 값만으로 승자를 정하지 않습니다. 두 값이 모두 낮아 간격만 작은 경우도 좋은 균형으로 보지 않습니다.")
para("Extra Trees의 로그 척도 R²는 점도 0.364, 탁도 0.239입니다. 예시 목표를 충족한 실측 출발 처방은 44개, 미측정 보간 후보는 990개이며 점추정 충족 99개·기술적 오차대까지 충족 0개입니다. 이는 정밀한 새 처방을 확정했다는 결과가 아닙니다.")
para("전체 475개 원료 조합 중 287개(60.4%)는 관측 처방이 하나뿐입니다. 원료별로 해당 원료를 포함한 처방은 112~452개, 안정 처방은 17~154개입니다. Plantapon Amino KG-L과 Dehyquart CC6는 안정 제형이 각각 17개여서 세부 조합까지 나눈 검증의 정밀도가 제한됩니다. 표본 수는 통과를 보장하지 않습니다.")

heading("3. 우선 고쳐야 할 부분")
table(["우선", "발견한 문제", "권장 수정"], [
    ["P0", "재학습과 평가를 위한 신규 자료 분리가 불명확", "신규 처방 A=개발·적응용, B=잠금 평가용으로 사전 분리. 도식의 ‘일부 조합 제외’를 본문과 명확히 일치시킴"],
    ["P0", "기존 모델은 원료별 고정 투입량 열 18개만 학습", "신규 원료를 별도 ID/입력 열로 추가한 모델 버전을 재학습. 모든 신규 처방을 평가용으로 빼면 새 열의 영향을 학습할 수 없음"],
    ["P0", "구조/INCI의 동일성이 상용 등급의 동일성으로 연결될 수 있음", "구조 검색과 물성 입력 승인을 분리. 명시적인 새 공급사·등급은 기존 열에 자동 대입하지 않고 보류"],
    ["P0", "개별 원료의 전체 농도 범위가 공동 조성의 보간 영역으로 표시됨", "‘관측 조합·개별 범위 일치’라는 사실만 표시. 공동 조성 거리·동일 조합 지지점·오차의 관계를 별도로 검증"],
    ["P0", "문서는 미통과 시 예측 보류, 앱은 범위 밖에서도 수치 출력", "미확인·미지원·조건 불일치일 때의 출력 정책을 하나로 고정하고 실패 입력 사례로 확인"],
    ["P1", "1:1 원료 대체와 최소 실험 수가 고정될 수 있음", "상용 원료 wt% 고정과 활성분 고정은 다른 비교임을 명시. 물·염·공급 용매의 차이도 기록하고 제약을 둔 혼합물 실험을 설계"],
    ["P1", "제조 반복과 측정 반복, 배치·조건 식별자가 구체적이지 않음", "처방 ID·제조 배치 ID·원료 로트·시험 ID·기술 반복 ID를 분리. 같은 처방/관련 배치는 같은 평가 그룹으로 관리"],
    ["P1", "통과 기준·필요 표본 수가 아직 없음", "민감도·특이도 각각의 요구 수준과 불확실성, 회귀 허용오차, 예측 보류율을 사용 목적에 맞춰 사전에 정함. 한 클래스뿐이면 판정 보류"],
    ["P2", "능동학습으로 ‘최소 실험’이 가능하다는 근거 없음", "초기 실험 설계와 단순 기준선을 먼저 사용. 같은 실험 예산·고정 평가셋에서 선택 전략을 비교한 뒤 별도 주장"],
    ["P2", "등록 단위가 원료 전체로 읽힐 수 있음", "원료×공급사×등급×조합×농도×제조·측정 조건×모델 버전의 제한된 지원 범위로 등록"],
])
para("신규 원료를 전혀 보지 않고 예측하는 별도 연구를 하려면 원료 간 공유 가능한 특성 표현이 필요합니다. 기능·활성분·고분자 특성 등의 입력은 후보지만, 상용 혼합물을 단일 SMILES로 바꾸는 것으로 해결되지는 않습니다. 기존 61/114개 입력에도 원료 ID별 열이 포함되므로 그 자체로 임의 신규 원료 지원을 뜻하지 않습니다.")

heading("4. 앱에서 재현한 구체적인 문제")
para("사례 A — 이름을 NEW_SUPPLIER_UNVERIFIED로 주고 Plantacare 2000의 대표 SMILES를 함께 넣으면 resolve_ingredient가 기존 Plantacare 2000에 status=resolved로 연결합니다. 메시지는 등급 확인 필요라고 알리지만 app.py의 차단 조건은 resolved 여부이므로 이 미확인 상태만으로는 차단되지 않습니다. 공급사 동일성이 확인됐다는 증거는 없습니다.")
para("사례 B — 관측이 한 번뿐인 제형 ID 38에서 Texapon SB 3 KC를 12.24%에서 전체 데이터의 상한인 16.24%로 바꿔도 applicability는 interpolation을 반환합니다. 같은 조합의 관측점이 한 개이므로 이 변경점은 그 조합에서 관측한 점들의 볼록껍질 안에 있지 않습니다. 실제 안정성·오차는 측정되지 않았으며 이 사례로 나쁜 물성이 생긴다고 주장하는 것은 아닙니다.")
para("사례 C — 저장된 주평가는 정확히 같은 원료 조합을 학습과 평가 사이에서 분리합니다. 학습에서 본 조합만 승인하는 규칙을 각 학습 fold 안에서 적용하면 평가 15개 fold 모두 승인 비율은 0%입니다. 현재 앱이 실제로 100% 거부한다는 뜻은 아닙니다. 현재 앱은 범위 밖에도 수치를 표시합니다. 이 결과는 향후 보류 규칙을 도입할 때 평가 분할과 실제 지원 범위의 정의를 함께 맞춰야 한다는 뜻입니다.")
para("개선: 알려진 원료의 새로운 조합 예측, 같은 조합에서 농도 변경, 완전히 새 원료 편입을 서로 다른 평가 문제로 정의합니다. 적용범위 필터·거리·스케일링·임계값은 각 학습 fold만으로 만들고, 평가 fold의 정답으로 조절하지 않습니다. 그룹 평가와 전처리 분리 원칙은 scikit-learn 공식 문서에도 설명돼 있습니다. [2]")

heading("5. 수정 프로토콜 제안")
table(["순서", "완료 조건"], [
    ["1. 목표와 지원 단위 고정", "신규 원료/공급사/등급 중 무엇이 바뀌는지, 단기 안정성 또는 특정 전단속도의 물성 중 무엇을 지원할지 명시"],
    ["2. 정보·비교 처방 등록", "원료 규격·활성분의 기준·물/염/고형분·등급·로트·문서 버전을 기록하고 미확인 값을 남김"],
    ["3. 실험·분할 사전 고정", "대조 처방, 제약된 농도 조합, 독립 제조 반복, 시험 순서를 설계. A 개발용과 B 잠금 평가용을 그룹 단위로 구분"],
    ["4. 측정과 QC", "제조 순서·혼합 에너지·실측 pH·적정액·온도·관찰시간·기기와 전단속도·단위·원시 반복값을 보존. 안정성부터 판단하고 결측 사유를 기록"],
    ["5. 개발용 자료로 모델 갱신", "기존 자료+A로 입력 스키마를 확장하고 재학습. 전처리·임계값·선택은 개발 자료 내부에서 결정. 기존 원료 성능 저하도 확인"],
    ["6. 잠금 평가와 제한 등록", "한 번 잠근 B로 네 분류 지표·회귀 오차·지원 비율을 평가. 부족하거나 기준 미달이면 보류. B로 다시 조정했으면 더 이상 최종 평가라고 부르지 않음"],
])
para("B가 별도 조합의 내부 보류셋인지, 다른 제조 배치·시험 시점에서 얻은 전향적 검증인지 구분합니다. 제품·시험 식별자로 독립성이 확인되지 않은 자료는 확증적 독립 외부검증으로 부르지 않습니다. 최소 실험 수는 혼합물 제약, 필요한 모델 복잡도, 배치 변동과 원하는 오차 정밀도에 따라 정하며, 몇 개만 측정하면 충분하다고 고정하지 않습니다. [3]")
para("분류에서는 안정·불안정 표본이 모두 필요합니다. 한 클래스만 있으면 일부 지표를 계산할 수 없으며 0 또는 1을 임의로 넣지 않습니다. 실측 안정 처방에서의 물성 오차와, 전체 후보 중 실제 불안정한 처방을 통과시키는 오류를 따로 보고합니다. 좋은 안정성 분류만으로 점도·탁도 지원까지 일괄 승인하지 않습니다.")

heading("6. 학술제에서 현실적인 핵심 결과")
para("권장 연구 질문: ‘제형의 입력 표현과 원료 정보의 불확실성이 샴푸 물성 예측의 균형과 적용범위 판단에 어떤 영향을 주는가?’ 현재 자료로 답할 수 있는 질문이며 개선이 없거나 특정 조건에서 실패하는 결과도 남길 수 있습니다. 예쁜 앱이나 프로토콜 도식만으로 연구 결과를 대신하지는 않습니다.")
table(["우선", "결과물", "현재 상태와 다음 작업"], [
    ["1", "기준선·균형 비교표", "완료된 OOF를 네 지표로 재확인. 양성 정의·분할·단위·실패 사례를 일관되게 제시"],
    ["2", "입력 표현·원료 불확실성 비교", "18개 투입량 대 61개 물성 특성 입력을 주 비교로 사전 고정. 114개 상호작용과 PK45 시나리오는 보조 비교. 동일 평가 분할·모델 조건 사용"],
    ["3", "예측 지원/보류의 근거 표시", "위 두 코드 사례를 수정하고, 원료·등급 미확인과 조성 범위 밖에서의 동작을 검증. 거부한 비율과 남은 표본의 성능을 함께 제시"],
    ["후속", "신규 원료 모의 편입", "공개 자료의 원료 하나를 가린 뒤 같은 원료의 일부 조합을 개발용으로 추가하고 나머지를 고정 평가. 실제 새 원료 검증과 구분"],
])
para("비교 모델을 고르거나 임계값을 튜닝한다면 내부 그룹 검증에서만 선택하고 외부 fold는 평가에 사용합니다. 이미 본 OOF에 맞춰 설정을 반복 선택하면 선택 편향이 생깁니다. 같은 자료로 수행한 추가 비교는 재사용한 내부 평가임을 명시해야 합니다.")
para("모의 편입은 전체 원료 목록과 포함 조건, 분할, 추가 자료 예산을 예측 전에 고정하고 원료별 결과를 모두 보고합니다. 유리한 원료 하나만 결과를 본 뒤 고르지 않습니다. 같은 처방이 여러 원료의 평가에 중복될 수 있으므로 원료별 결과를 서로 독립인 18개 실험처럼 합치지 않습니다. 이미 전체 자료로 학습된 저장 모델을 그대로 가져와 ‘가린 원료를 보지 않았다’고 주장하지 않고, 각 실험에서 처음부터 학습합니다.")
para("실제 제조·측정이 가능하면 위 비교 연구 뒤에 신규 원료 1종의 제한된 편입 사례를 추가할 수 있습니다. 실험이 어렵다면 신규 원료 부분은 모의 편입 결과 또는 후속 프로토콜로 제시합니다. 어느 경우에도 ‘어떤 새 원료든 예측하는 완성 모델’이나 ‘최소 실험으로 편입 성공’을 현재 성과로 쓰지 않습니다.")

heading("7. 저장된 근거와 검토 범위")
para("검토 결과는 test1 안에만 저장했습니다. 원본 Word 문서, 기존 앱·모델·데이터, GitHub는 수정하지 않았습니다. 이번에는 실행 계획과 현재 동작을 감사했으며 보강 입력 재학습이나 신규 원료 실험은 수행하지 않았습니다. 소스 파일의 검토 전후 해시가 같습니다.")
table(["파일", "내용"], [
    ["신규원료_프로토콜_실현가능성_검토.docx / 검토보고서.html", "동일한 검토 본문"],
    ["ingredient_support.csv", "18개 원료별 처방·클래스·물성·조합 수"],
    ["classification_metrics_by_repeat.csv / classification_metrics_summary.csv", "기존 OOF의 네 지표 재계산, 반복별 혼동행렬과 SD"],
    ["app_counterexamples.json / hypothetical_gate_coverage.csv", "현재 앱의 연결·보간 판정 사례, 가정한 조합 승인 규칙의 평가 범위"],
    ["audit_summary.json / source_inventory.json", "검사 결과, 원본·입력·코드 해시와 변경 없음 확인"],
    ["audit_feasibility.py / make_review.py", "감사와 문서 생성 재현 코드"],
])
para("실행: 현재 프로젝트와 원본 Word가 같은 경로에 있을 때 test1에서 python -B audit_feasibility.py 다음 python -B make_review.py를 실행합니다. 기존 연구 환경의 numpy, pandas, scikit-learn, joblib, rdkit가 필요합니다. Word 작성은 Python 표준 라이브러리만 사용합니다.")
para("[1] Chitre et al. (2024), 원 논문 초록·서지. 원료 수, 제형 수, 연구 범위 밖 예측 한계. https://pubmed.ncbi.nlm.nih.gov/38961122/")
para("[2] scikit-learn 공식 문서, 그룹 교차검증 및 학습 자료에서만 수행하는 전처리. https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data")
para("[3] NIST/SEMATECH e-Handbook, 혼합물 설계와 농도 제약·모델 적합성 평가. https://www.itl.nist.gov/div898/handbook/pri/section5/pri54.htm")
para("외부 자료는 설계 원칙을 확인하는 데 사용했습니다. 현재 모델의 수치·코드 문제·실현 가능성 판단은 로컬 파일을 직접 감사한 결과와 그에 대한 검토 의견입니다. 원 논문 전체 본문은 이번 웹 조회에서 접근 제한이 있었으므로 초록 이상의 새 사실을 인용하지 않았습니다.")

# Render the same structured content as readable HTML and a standard OOXML Word file.
html_body = []
for kind, value in blocks:
    if kind == "table":
        headers, rows = value
        html_body.append("<table><thead><tr>" + "".join("<th>" + html.escape(c) + "</th>" for c in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(str(c)) + "</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>")
    else:
        tag = {"title": "h1", "heading": "h2", "paragraph": "p"}[kind]
        html_body.append(f"<{tag}>{html.escape(value)}</{tag}>")
(OUT / "검토보고서.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>신규 원료 편입 검토</title><style>body{font:16px/1.7 'Malgun Gothic',sans-serif;color:#193349;max-width:1100px;margin:40px auto;padding:0 24px}h1{font-size:28px}h2{font-size:22px;margin-top:36px;border-bottom:2px solid #167987}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{border:1px solid #c9d5df;padding:9px;vertical-align:top}th{background:#e9f2f5}p{margin:14px 0}@media print{body{font-size:10pt;max-width:none;margin:0}h2{page-break-after:avoid}tr{page-break-inside:avoid}thead{display:table-header-group}}</style><body>" + "\n".join(html_body) + "</body></html>", encoding="utf-8")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)
def el(name, parent=None, **attrs):
    node = ET.Element("{" + W + "}" + name, {"{" + W + "}" + k: str(v) for k, v in attrs.items()})
    if parent is not None: parent.append(node)
    return node
doc = el("document")
body = el("body", doc)
def paragraph(parent, text, style=None):
    p = el("p", parent)
    if style:
        prop = el("pPr", p)
        el("pStyle", prop, val=style)
    r = el("r", p)
    t = el("t", r)
    t.text = str(text)
    return p
for kind, value in blocks:
    if kind != "table":
        paragraph(body, value, {"title": "Title", "heading": "Heading1", "paragraph": "Normal"}[kind])
        continue
    headers, rows = value
    tbl = el("tbl", body)
    pr = el("tblPr", tbl)
    el("tblW", pr, w=9360, type="dxa")
    borders = el("tblBorders", pr)
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        el(edge, borders, val="single", sz=4, color="C9D5DF")
    grid = el("tblGrid", tbl)
    for _ in headers: el("gridCol", grid, w=9360 // len(headers))
    for i, row in enumerate([headers] + rows):
        tr = el("tr", tbl)
        trpr = el("trPr", tr)
        el("cantSplit", trpr)
        if i == 0: el("tblHeader", trpr)
        for cell in row:
            tc = el("tc", tr)
            tcpr = el("tcPr", tc)
            el("tcW", tcpr, w=9360 // len(headers), type="dxa")
            if i == 0: el("shd", tcpr, fill="E9F2F5")
            paragraph(tc, cell, "TableText")
sect = el("sectPr", body)
el("pgSz", sect, w=11906, h=16838)
el("pgMar", sect, top=1080, right=1273, bottom=1080, left=1273, header=720, footer=720, gutter=0)
styles = el("styles")
for name, size in [("Normal", 21), ("Title", 36), ("Heading1", 27), ("TableText", 18)]:
    style = el("style", styles, type="paragraph", styleId=name)
    el("name", style, val=name)
    pp = el("pPr", style)
    el("spacing", pp, after=140, line=300, lineRule="auto")
    if name in ["Title", "Heading1"]: el("keepNext", pp)
    rp = el("rPr", style)
    el("rFonts", rp, ascii="Calibri", hAnsi="Calibri", eastAsia="맑은 고딕")
    el("sz", rp, val=size)
    el("szCs", rp, val=size)
    if name in ["Title", "Heading1"]:
        el("b", rp)
        el("color", rp, val="193349")
docx = OUT / "신규원료_프로토콜_실현가능성_검토.docx"
with ZipFile(docx, "w", ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
    z.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    z.writestr("word/_rels/document.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
    z.writestr("word/document.xml", ET.tostring(doc, encoding="utf-8", xml_declaration=True))
    z.writestr("word/styles.xml", ET.tostring(styles, encoding="utf-8", xml_declaration=True))
with ZipFile(docx) as z:
    assert z.testzip() is None
    for item in z.namelist(): ET.fromstring(z.read(item))
    restored = ET.fromstring(z.read("word/document.xml"))
    text = " ".join(t.text or "" for t in restored.iter("{" + W + "}t"))
    for term in ["812", "287", "0.546", "NEW_SUPPLIER_UNVERIFIED", "16.24%", "잠금 평가용", "재학습"]:
        assert term in text, term
inventory = json.loads((OUT / "source_inventory.json").read_text(encoding="utf-8"))
for path, expected in inventory["project_files"].items():
    assert hashlib.sha256((OUT.parent / path).read_bytes()).hexdigest() == expected, path
assert hashlib.sha256(Path(inventory["original_protocol"]).read_bytes()).hexdigest() == inventory["original_protocol_sha256"]
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "review_manifest.json"}
(OUT / "review_manifest.json").write_text(json.dumps({"files": manifest, "docx_package_xml_check": "PASS", "docx_word_visual_render": "not_performed", "source_files_unchanged": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "docx": str(docx), "html": str(OUT / '검토보고서.html'), "review_files": len(manifest) + 1, "source_files_unchanged": True}, ensure_ascii=False, indent=2))
