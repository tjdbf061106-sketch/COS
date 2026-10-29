# -*- coding: utf-8 -*-
"""선행연구(논문) 요약 PDF 6종 생성 - NTP ICE 기반 화장품 자극성 예측 도메인전이 연구"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold', 'C:/Windows/Fonts/malgunbd.ttf'))

title_style = ParagraphStyle('Title', fontName='MalgunBold', fontSize=15, leading=20,
                              alignment=0, textColor=colors.HexColor('#1a3d6d'), spaceAfter=8)
meta_style = ParagraphStyle('Meta', fontName='Malgun', fontSize=9.7, leading=15,
                             textColor=colors.HexColor('#444444'), spaceAfter=3)
link_style = ParagraphStyle('Link', fontName='Malgun', fontSize=9.7, leading=15,
                             textColor=colors.HexColor('#0b5fa5'), spaceAfter=10)
tag_style = ParagraphStyle('Tag', fontName='MalgunBold', fontSize=9, leading=13,
                            textColor=colors.white, alignment=1)
h1_style = ParagraphStyle('H1', fontName='MalgunBold', fontSize=12, leading=17,
                           spaceBefore=12, spaceAfter=6, textColor=colors.HexColor('#1a3d6d'))
body_style = ParagraphStyle('Body', fontName='Malgun', fontSize=10, leading=16.5,
                             spaceAfter=6, alignment=4)
note_style = ParagraphStyle('Note', fontName='Malgun', fontSize=8.5, leading=12.5,
                             textColor=colors.HexColor('#777777'), spaceAfter=4)


def tag_table(stage_text):
    t = Table([[Paragraph(f"관련 단계: {stage_text}", tag_style)]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#2a6f3d')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def build_paper_pdf(out_path, title_kr_hint, title_orig, authors, year, journal, url, doi,
                     summary_paras, usage_paras, stage_text, caveat=None):
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=20 * mm, bottomMargin=18 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    els = []
    els.append(Paragraph("선행연구 요약", ParagraphStyle('Kicker', fontName='Malgun', fontSize=9.5,
                                                     textColor=colors.HexColor('#888888'), spaceAfter=4)))
    els.append(Paragraph(title_orig, title_style))
    els.append(Paragraph(f"저자: {authors}", meta_style))
    els.append(Paragraph(f"발표: {year}년 · {journal}", meta_style))
    els.append(Paragraph(f"DOI: {doi}", meta_style))
    els.append(Paragraph(f"링크: {url}", link_style))
    els.append(tag_table(stage_text))
    els.append(Spacer(1, 8))

    els.append(Paragraph("핵심 내용 요약", h1_style))
    for p in summary_paras:
        els.append(Paragraph(p, body_style))

    els.append(Paragraph("우리 연구에 어떻게 활용할 수 있는가", h1_style))
    for p in usage_paras:
        els.append(Paragraph(p, body_style))

    if caveat:
        els.append(Spacer(1, 6))
        els.append(HRFlowable(width="100%", color=colors.HexColor('#dddddd')))
        els.append(Spacer(1, 4))
        els.append(Paragraph(caveat, note_style))

    doc.build(els)
    print("OK ->", out_path)


OUT_DIR = r"C:\Users\UserK\Desktop\학술제\선행연구_요약"

papers = [
    dict(
        out_path=OUT_DIR + r"\1_NTP_ICE_Bell2017.pdf",
        title_orig="An Integrated Chemical Environment to Support 21st-Century Toxicology",
        authors="Bell SM, Phillips J, Sedykh A 외 9명",
        year=2017,
        journal="Environmental Health Perspectives",
        url="https://pubmed.ncbi.nlm.nih.gov/28557712/",
        doi="10.1289/EHP1759",
        stage_text="① NTP ICE 데이터 확보",
        summary_paras=[
            "NTP 산하기관인 NICEATM이 개발한 웹 기반 통합 화학물질 환경(Integrated Chemical Environment, ICE)을 "
            "소개하는 원 논문이다. ICE는 curated in vivo 시험 데이터, 참고 화학물질 정보, Tox21/ToxCast "
            "고속대량스크리닝(HTS) in vitro 데이터, in silico 예측모델을 하나의 플랫폼에 통합한다.",
            "급성전신독성, 내분비 교란, 피부 감작성 등 다양한 endpoint를 질의(query)하고 서로 다른 데이터 "
            "소스를 결합할 수 있도록 설계되었으며, 동물실험 대체법(alternative method)의 개발·검증·실무 "
            "적용을 지원하는 것이 핵심 목적이다.",
            "오픈소스 워크플로우와 예측모델을 다운로드하여 로컬 환경에서 직접 활용할 수 있도록 공개하고 있어, "
            "규제기관·연구자가 재현 가능한 방식으로 대체시험법을 검토할 수 있게 한다.",
            "이 논문 자체는 특정 데이터셋의 구축 방법론을 상세히 다루기보다, 플랫폼과 리소스 전반을 소개하는 "
            "성격이 강하다.",
        ],
        usage_paras=[
            "우리 연구의 데이터 소스인 NTP ICE 자체를 인용할 때 반드시 필요한 1차 출처(origin paper)이다. "
            "서론에서 \"왜 NTP ICE 데이터를 학습 데이터로 사용하는가\"를 정당화할 때 이 논문을 인용하면 된다.",
            "다만 EPA 농약 제형의 피부/안 자극성 Category 1~4 라벨의 세부 구조(FIFRA 체계)는 이 논문만으로는 "
            "부족하므로, ICE 웹사이트에서 다운로드한 실제 데이터 딕셔너리(skin_irritation.xlsx, "
            "eye_irritation.xlsx)를 별도로 확인해 보완해야 한다.",
        ],
    ),
    dict(
        out_path=OUT_DIR + r"\2_CATMoS_Mansouri2021.pdf",
        title_orig="CATMoS: Collaborative Acute Toxicity Modeling Suite",
        authors="Mansouri K, Karmaus AL, Fitzpatrick J 외 90여명 (35개 국제 연구그룹 공동)",
        year=2021,
        journal="Environmental Health Perspectives",
        url="https://pubmed.ncbi.nlm.nih.gov/33929906/",
        doi="10.1289/EHP8495",
        stage_text="②③⑤ 라벨 체계 · 트리기반 ML 앙상블 · 적용범위(AD)",
        summary_paras=[
            "급성 경구독성(acute oral toxicity)을 동물실험 없이 예측하기 위해 국제 컨소시엄이 진행한 대규모 "
            "협업 모델링 프로젝트이다. 11,992개 화학물질의 LD50 데이터를 학습/평가셋으로 나누고, 35개 연구그룹이 "
            "총 139개의 개별 예측모델(다양한 머신러닝 알고리즘 포함)을 제출했다.",
            "각 개별 모델은 자신이 정의한 적용범위(applicability domain, AD) 내에서만 평가되었으며, 이 "
            "개별 모델들의 예측을 결합하는 컨센서스(consensus) 전략으로 최종 예측치를 산출했다.",
            "예측 대상 endpoint는 LD50 값, EPA 위해도 4단계 분류, GHS 위해도 5단계 분류, 고독성/무독성 "
            "이진분류 등 5가지이다. 최종 예측치는 NTP ICE와 오픈소스 도구 OPERA를 통해 80만개 이상의 "
            "화학물질에 대해 공개되어 있다.",
        ],
        usage_paras=[
            "EPA 위해도 다단계(Category) 분류 예측을 다룬 대표적 선행연구로, 우리 연구의 라벨 체계(EPA "
            "Classification Category 1~4)와 가장 유사한 구조를 가진다.",
            "각 개별모델이 \"자신의 적용범위 내에서만\" 평가된다는 설계는 우리 연구에서 ICE mixture 도메인과 "
            "화장품 도메인 간 Applicability Domain을 정의·평가하는 방식에 직접 참고할 수 있다.",
            "다중 모델의 예측을 결합하는 컨센서스 방식은 Random Forest/XGBoost 등 트리기반 앙상블 모델 설계 "
            "및 성능 비교 방법론에 참고할 수 있다.",
        ],
    ),
    dict(
        out_path=OUT_DIR + r"\3_STopTox_Borba2022.pdf",
        title_orig="STopTox: An in Silico Alternative to Animal Testing for Acute Systemic and Topical Toxicity",
        authors="Borba JVB, Alves VM, Braga RC 외 11명",
        year=2022,
        journal="Environmental Health Perspectives",
        url="https://pubmed.ncbi.nlm.nih.gov/35192406/",
        doi="10.1289/EHP9341",
        stage_text="③④⑤ 자극성 예측 모델 · 화장품/농약 겸용 적용 · 적용범위(AD)",
        summary_paras=[
            "농약·의약품·화장품 규제 평가에 필요한 급성독성 \"6-pack\" (피부 감작성, 피부 자극/부식성, 안 "
            "자극/부식성, 급성 경구·흡입·경피 독성)을 동물실험 없이 예측하는 QSAR 앙상블 모델 시스템을 "
            "제시한다.",
            "각 endpoint별로 공개된 데이터셋 중 가장 큰 규모를 큐레이션했으며(급성 흡입 독성 기준 약 "
            "681~8,978개 화합물), 분자 구조 기반 서술자(molecular descriptor)를 활용한 QSAR 앙상블 모델을 "
            "구축했다.",
            "외부검증셋에서 70~77%의 정확분류율(correct classification rate)을 달성했으며, 웹 포털"
            "(stoptox.mml.unc.edu)로 공개되어 누구나 화학물질의 독성을 스크리닝할 수 있다.",
        ],
        usage_paras=[
            "피부/안 자극성을 화장품·농약 두 도메인에 걸쳐 정확히 다룬 논문으로, 우리 연구의 ③④⑤ 단계"
            "(자극성 예측 모델 구축 → 화장품 데이터 적용 → 결과 도출)와 거의 일대일로 대응된다.",
            "QSAR 서술자 기반 접근을 트리기반 모델(RF/XGBoost)의 피처 엔지니어링에 참고할 수 있고, 외부검증 "
            "정확도(70~77%)를 우리 모델의 성능 벤치마크로 활용할 수 있다.",
        ],
    ),
    dict(
        out_path=OUT_DIR + r"\4_ExplainableML_Liu2025.pdf",
        title_orig="Using explainable machine learning to predict the irritation and corrosivity of chemicals on eyes and skin",
        authors="Liu Y, Liu Y, Zhang S 외 10명",
        year=2025,
        journal="Toxicology Letters",
        url="https://pubmed.ncbi.nlm.nih.gov/40180199/",
        doi="10.1016/j.toxlet.2025.03.008",
        stage_text="③⑤ ML 자극성 예측 · 해석가능성/적용범위 근거",
        summary_paras=[
            "안 자극성 3,316건, 피부 자극성 3,080건의 실험 데이터를 통합하여 다양한 머신러닝/딥러닝 모델을 "
            "개발한 2025년 최신 연구이다. 최고 성능 모델은 외부검증에서 안 자극성 balanced accuracy 73.0%, "
            "피부 자극성 75.1%를 달성했다.",
            "단순 예측을 넘어 데이터셋 수준-분자 수준-원자 수준의 다층적 해석(interpretability) 분석을 "
            "수행하여 독성과 관련된 구조 경고 프래그먼트(structural alert fragment)를 규명했고, 비전문가도 "
            "사용할 수 있는 시각화 인터페이스를 개발했다.",
            "대상 화학물질은 농약, 화장품, 의약품 전반을 포괄한다.",
        ],
        usage_paras=[
            "우리 연구와 가장 유사한 목표(피부·안 자극성/부식성 예측, 농약·화장품 겸용)를 다룬 최신(2025) "
            "논문으로, 성능 벤치마크(balanced accuracy 70%대)와 SHAP류 해석기법 적용 방식을 직접 참고할 수 "
            "있다.",
            "다만 이 논문은 transfer learning/도메인 전이를 명시적으로 다루지 않으므로, \"동일 도메인 내 "
            "예측\"과 \"도메인 전이 예측\"의 성능 차이를 비교하는 논거(우리 연구의 novelty 강조)로 활용할 수 "
            "있다.",
        ],
    ),
    dict(
        out_path=OUT_DIR + r"\5_TransferLearning_ODonovan2023.pdf",
        title_orig="Application of transfer learning to predict drug-induced human in vivo gene expression changes using rat in vitro and in vivo data",
        authors="O'Donovan SD, Cavill R, Wimmenauer F 외 9명",
        year=2023,
        journal="PLoS One",
        url="https://pubmed.ncbi.nlm.nih.gov/38032940/",
        doi="10.1371/journal.pone.0292030",
        stage_text="④ Transfer Learning / Domain Adaptation (핵심)",
        summary_paras=[
            "약물독성 예측 분야에서 종간(rat → human) 도메인 전이 문제를 다룬 연구이다. Open TG-GATEs 유전자 "
            "발현 데이터셋을 이용해, rat in vitro/in vivo 데이터를 소스 도메인, human in vitro 데이터를 "
            "타겟 도메인으로 설정했다.",
            "gradient reversal layer를 포함한 도메인 분류기(domain classifier)를 갖는 비지도 도메인 적응"
            "(unsupervised domain adaptation) 딥러닝 구조를 적용하여, 신경망의 특정 계층에서 rat/human "
            "도메인이 더 이상 구분되지 않을 정도로 도메인 갭을 줄였다.",
            "그 결과 새로운 약물 노출에 대한 인간 유전자발현 반응을 생리학적으로 타당하게 예측할 수 있었고, "
            "인간 in vitro 데이터를 함께 학습에 포함시키자 rat in vivo 반응 예측의 시간적 정확도도 "
            "향상되었다.",
        ],
        usage_paras=[
            "화학물질 독성예측에서 \"한 도메인(rat)에서 학습 후 다른 도메인(human)으로 전이\"하는, 정확히 "
            "우리 연구 구조(농약제형 → 화장품제형)와 동일한 문제 설정을 가진 사례이다.",
            "gradient reversal layer 기반 도메인 적응이나, 도메인 간 특징 분포 차이가 줄어드는지 확인하는 "
            "방식(예: hidden layer에서 도메인 구분 가능성을 측정)을 우리의 도메인 갭 정량화 지표 설계에 "
            "응용할 수 있다.",
        ],
        caveat="※ 이 논문은 트리기반 모델이 아닌 딥러닝(신경망) 기반 도메인 적응이므로, 방법론을 그대로 "
               "적용하기보다 \"도메인 갭을 측정·축소하는 개념적 프레임\"을 참고하는 데 적합하다.",
    ),
    dict(
        out_path=OUT_DIR + r"\6_ImbalancedTox21_Idakwo2020.pdf",
        title_orig="Structure-activity relationship-based chemical classification of highly imbalanced Tox21 datasets",
        authors="Idakwo G, Thangapandian S, Luttrell J 외 6명",
        year=2020,
        journal="Journal of Cheminformatics",
        url="https://pubmed.ncbi.nlm.nih.gov/33372637/",
        doi="10.1186/s13321-020-00468-x",
        stage_text="②③ 클래스 불균형 처리 · 트리기반 모델",
        summary_paras=[
            "독성 데이터셋에서 활성(독성양성) 화합물이 비활성 화합물보다 훨씬 적은 극단적 클래스 불균형 "
            "문제를 다룬 연구이다. Random Forest를 기본 분류기로 하여 (1) 처리 없음, (2) 랜덤 언더샘플링"
            "(RUS), (3) SMOTE, (4) SMOTEENN(SMOTE + Edited Nearest Neighbor로 합성표본 생성 후 오분류 표본 "
            "정제) 네 가지 방식을 비교했다.",
            "12개의 Tox21 바이오어세이 데이터셋에서 실험한 결과 SMOTEENN이 대체로 가장 우수한 성능을 "
            "보였으나, 불균형비가 약 28:1을 초과하면 효과가 급격히 저하되었다.",
            "단순 정확도보다 F1-score, Matthews 상관계수(MCC), Brier score가 불균형 데이터 평가에 더 "
            "일관된 지표임을 제시했다.",
        ],
        usage_paras=[
            "EPA Category 1~4 라벨은 실제로 특정 category(특히 저자극/Category 4)에 데이터가 몰려 극심하게 "
            "불균형할 가능성이 높으므로(skin invivo mixture 데이터 기준 Category 4가 약 80%), 이 논문의 "
            "SMOTE/SMOTEENN 적용 방식과 성능 저하 임계치(28:1)를 우리의 클래스 불균형 처리 설계에 직접 "
            "참고할 수 있다.",
            "평가지표 선정(F1-score, MCC)에도 이 논문의 결론을 그대로 적용할 수 있으며, \"불균형비가 일정 "
            "수준을 넘으면 오버샘플링도 한계가 있다\"는 결론은 우리 데이터의 불균형 정도를 사전에 진단해야 "
            "한다는 시사점을 준다.",
        ],
    ),
]

for p in papers:
    build_paper_pdf(**p, title_kr_hint=None)

print("\n총", len(papers), "개 PDF 생성 완료 ->", OUT_DIR)
