# COS 학술제 — 인계 문서 (2026-11-06 01:00)

새 세션에서 이어서 작업하기 위한 자립 문서. ChemExpo 데이터 수집 단계부터 기록.

---

## 0. 한 줄 상태

**제형 단위 EPA 피부 자극성 학습 데이터 추가 조달은 실패했고, 대신 "농약↔화장품 공유 보존제 22종이 양성에 농축되어 있다"는 통계적 발견을 얻었다. 다만 그 발견을 feature로 넣어도 모델 성능은 측정 변동폭 안에서만 움직인다.**

---

## 1. 배경 (이전 결론 요약)

- **보유 데이터**: ICE skin 541제형 / 양성 30건(5.5%) / Cat4 81.8%. 특성 = 성분 함량 가중 RDKit descriptor 217개 + Morgan FP 2048 + 공통 8개 = 2,273차원
- **진단된 병목 3개** (노트북 10절): ① 라벨 노이즈 κ≈0.1 ② 조성 기재 불완전(선언 조성 중앙값 31.5%, 미기재 inert 약 2/3) ③ 두 도메인 성분 어휘 단절
- **외부 데이터 조사 결론** (`claude/C_skin_formulation_data.md`): EPA 등급 + skin + 제형별 행, 세 조건을 다 만족하는 공개 데이터는 ICE 외에 없음. Corvaro 2017은 집계만(양성 5건), Kolle 2017은 제형별 행은 있으나 **조성 컬럼이 없음**(라벨 CLP)
- 핵심 통찰: **라벨이 아니라 조성이 병목이다.** 논문들이 제형 조성을 영업비밀로 비공개함

---

## 2. ChemExpo (EPA) 확인 결과

교수님이 준 링크: https://comptox.epa.gov/chemexpo/pucs/

### 2-1. PUC는 제품 리스트가 아니라 분류 체계
3단계 계층(General Category → Product Family → Product Type). 전체 476개 PUC.

| General category | 누적 제품수 |
|---|---|
| **Personal care** | **99,720** |
| Home maintenance | 31,996 |
| Cleaning products and household care | 16,350 |
| **Pesticides** | **2,009** |

Personal care 하위: makeup and related 29,248 / hair styling and care 16,904 / facial cleansing and moisturizing 12,993 / body hygiene 7,659 / general moisturizing 6,326 / sunscreen 5,787 / deodorant 2,409 / hair coloring 2,391

### 2-2. 다운로드 URL (워크스페이스에서 EPA 접속 불가 — 수동 다운로드 필요)
```
PUC 메타데이터     https://comptox.epa.gov/chemexpo/dl_pucs/
전체 조성 (60MB+)  https://comptox.epa.gov/chemexpo/dl_co_chemicals/
PUC별 조성         각 PUC 상세 페이지의 "Download Products and Chemical Weight Fractions" 버튼
                   Personal care = /chemexpo/puc/137/ , face wash = /puc/182/ , hair styling = /puc/205/
```

### 2-3. 치명적 한계 — 함량이 거의 없다
받은 파일: `ChemExpo_Personal_care_products_and_weight_fractions_20261105.csv` (PUC 137만, 7.9MB)

| 항목 | 값 |
|---|---|
| 행 / 고유 제품 | 36,174 / **2,254** |
| 큐레이션 성분 (DTXSID 있음) | 20,029행 (55.4%) / **고유 1,341종** |
| 제품당 성분 수 | 중앙값 **14**, 평균 16, 최대 82 |
| **함량(Central Weight Fraction) 있는 행** | **665 / 36,174 (1.8%)** |
| 함량 있는 제품 | 183 / 2,254 (8%) |
| **성분 5개 이상 + 조성 합계 0.80~1.05** | **13개** (그중 2개는 산업용 오분류 → 실질 11개) |

→ **완전 조성 화장품 제형은 11개.** 기존 타깃 도메인 27종보다 적다.
→ 원인: 화장품 라벨은 성분을 함량 내림차순으로만 적고 %를 안 적는다. 구조적 문제라 전체 bulk(99,720)를 받아도 비율은 비슷할 것
→ **따라서 Donahue 2011(안구: 에탄올>10%/계면활성제>10%)과 Lechuga 2025(피부: HLB·농도·이온특성) 베이스라인 계획은 둘 다 무효**

### 2-4. 데이터 품질 문제
- PUC 분류가 72% Automatic(NLP) → 산업용 제품(`way oil`, `nitriding stop-off coating`)이 Personal care에 섞임
- 같은 제품에 `Deionized water`가 서로 다른 값으로 중복
- 조성 합계가 최대 4.011(400%)인 제품 존재

### 2-5. 미탐색 — Pesticides 2,009건
ChemExpo에 농약 제품 2,009건의 조성이 있다. ICE의 미기재 inert를 메울 가능성(병목 ②). 제품 식별자(제품명/EPA 등록번호) 매칭 가능성은 **아직 확인 안 함**.

---

## 3. 핵심 발견 — 공유 보존제 22종의 양성 농축

상세: `claude/D_shared_ingredient_transfer_finding.md`

### 3-1. 겹침
ICE skin 270종 ∩ ChemExpo Personal care 1,341종 = **22종** (ICE 기준 8.1%)

노트북 7.2절의 기존 서술 "구조 수준 겹침 2/67(에탄올, 과산화수소뿐), ID 블록에서 화장품은 전부 영벡터, identity 기반 전이는 구조적으로 불가능" → **틀렸다.** 기존 수치는 화장품 27제품의 성분 67종을 모수로 한 것.

### 3-2. 양성 농축 (화학 계열별)

| 그룹 | 보유 제형 | 양성률 | 미보유 양성률 | OR | Fisher p |
|---|---|---|---|---|---|
| **[A] 보존제·살균제 13종** | **21** | **42.9% (9/21)** | 4.0% | **17.93** | **2.5e-07** |
| [B] 농약활성성분·용매·향료·황 9종 | 18 | 0.0% (0/18) | 5.7% | 0.00 | 1.00 |
| [C] 공유 22종 전체 | 38 | 23.7% (9/38) | 4.2% | 7.17 | 7.5e-05 |

**신호는 전부 보존제·살균제 계열에 있다.** 농약 활성성분(malathion, chlorpyrifos, DEET)과 황은 양성 0.

### 3-3. 성분별 (ICE 제형 수 / 양성 / 양성률, 기준선 5.5%)

| CASRN | 물질 | ICE제형 | 양성 | 양성률 | %AI 중앙 | 화장품 제품수 |
|---|---|---|---|---|---|---|
| 2682-20-4 | MIT | 6 | 5 | **83%** | 2.275 | **188** |
| 26172-55-4 | CMIT | 4 | 4 | **100%** | 0.790 | **166** |
| 52-51-7 | Bronopol | 3 | 3 | **100%** | 7.500 | 2 |
| 7173-51-5 | Didecyldimethylammonium chloride | 5 | 2 | 40% | 0.760 | 2 |
| 118-52-5 | 1,3-Dichloro-5,5-dimethylhydantoin | 2 | 2 | 100% | 27.400 | 1 |
| 55406-53-6 | IPBC | 5 | 1 | 20% | 3.000 | 1 |
| 16079-88-2 | 1-Bromo-3-chloro-5,5-DMH | 1 | 1 | 100% | 60.000 | 1 |
| 89415-87-2 | 1,3-Dichloro-5-ethyl-5-methyl-DMH | 1 | 1 | 100% | 10.600 | 1 |
| 7704-34-9 | Sulfur | 8 | 0 | 0% | 59.675 | 7 |
| 121-75-5 | Malathion | 3 | 0 | 0% | 50.000 | 1 |
| 2921-88-2 | Chlorpyrifos | 2 | 0 | 0% | 34.475 | 1 |
| 64-17-5 | Ethanol | 1 | 0 | 0% | 7.500 | 141 |
| 6440-58-0 | DMDM hydantoin | 1 | 0 | 0% | 13.400 | 97 |
| (나머지 8종) | geraniol, isopropanol, H2O2, cinnamaldehyde, chloroxylenol, zinc pyrithione, dioctyl quat, octyl decyl quat | 1~3 | 0 | 0% | — | 1~80 |

**MIT는 화장품 188제품, CMIT는 166제품에 들어있다** → 타깃 도메인에 실제로 존재.

### 3-4. 수정된 해석
농약→화장품 전이의 다리는 "농약 활성성분"이 아니라 **보존제·살균제**다. isothiazolone·할로겐화 하이단토인·4급 암모늄은 독립적으로 알려진 피부 자극/감작 물질 계열이다. 기존 제로샷 전이 붕괴는 전이가 불가능해서가 아니라 **타깃 표본이 27개뿐이라 공유 성분 담은 제품이 거의 없었기 때문**일 수 있다.

### 3-5. 한계 (반드시 함께 보고할 것)
- 표본 작음: 양성 중 보유 9개, 전체 보유 21~38개. OR 신뢰구간 넓음
- 양성의 70%(21/30)는 공유 성분 없음 → 이 경로로 설명되는 건 양성의 30%
- 인과 아님: 공유 성분 보유가 원인인지, 그 성분을 쓰는 제형 유형이 자극성인지 구분 안 됨
- **순환 위험**: [A]/[B] 그룹을 화학 계열로 정의했지만, 성분별 양성률 표를 **먼저 본 뒤** 그룹을 나눴다. 가설 생성적 결과로 보고해야 하며 확증이 아니다. 외부 근거(isothiazolone이 규제 대상 자극물질이라는 독립 지식)가 유일한 뒷받침
- CASRN 완전일치 기준. 구조 유사체까지 보면 겹침이 더 클 수 있음
- ChemExpo 쪽은 PUC 137(2,254제품)만. 하위 PUC 포함 시 1,341종이 늘어나고 겹침도 늘어날 것 → **현재 결과는 과소추정 가능**

---

## 4. feature 투입 실험 결과 — 성능은 안 오른다

재현 코드: `/home/claude/exp_biocide.py` (아래 §5-3에 전문 보존)
프로토콜: StratifiedGroupKFold 5-fold × 5seed, comp_group_id 기준, class_weight balanced, threshold 0.30, 모든 변환 fold 내부 fit

데이터: 541행 / 양성 30 / group 526 / 보존제 보유 제형 21

### LR (C=0.1)
| 구성 | 차원 | Sens | Spec | MCC | AUC |
|---|---|---|---|---|---|
| (a) 기존 descriptor+FP (nonzero≥20 필터) | 2273→528 | 0.420±0.061 | 0.868±0.007 | 0.185±0.041 | 0.707±0.040 |
| (b) 기존 + 보존제 13종 (필터 적용) | 2286 | 0.420±0.061 | 0.868±0.007 | 0.185±0.041 | 0.707±0.040 |
| (c) 보존제 13종 only | 13 | 0.987±0.030 | 0.007±0.005 | -0.020±0.083 | 0.586±0.026 |
| (d) 보존제 13종 + 함량 | 26 | 0.960±0.028 | 0.009±0.002 | -0.066±0.066 | 0.542±0.026 |
| **(a) 필터 없음** | 2273 | 0.340±0.043 | 0.907±0.010 | 0.184±0.041 | 0.652±0.049 |
| **(b) 필터 없음** | 2286 | 0.360±0.043 | 0.910±0.009 | **0.203±0.044** | **0.667±0.047** |

### RF (300 trees, min_samples_leaf=2)
| 구성 | 차원 | Sens | Spec | MCC | AUC |
|---|---|---|---|---|---|
| (a) 기존 | 2273 | 0.313±0.018 | 0.922±0.002 | 0.187±0.015 | 0.743±0.018 |
| (b) 기존 + 보존제 13종 | 2286 | 0.313±0.018 | 0.922±0.002 | 0.187±0.015 | 0.743±0.018 |
| (c) 보존제 13종 only | 13 | 1.000±0.000 | 0.000±0.000 | 0.000±0.000 | 0.542±0.021 |

### 중요한 함정 — (b)=(a)는 허수였다
`nonzero≥20` 필터(B 리포트 권고)가 **보존제 13개를 전부 제거**했다. 각 성분이 1~6개 제형에만 나타나기 때문:
```
보존제 13개 각각의 nonzero 수: [6, 4, 3, 1, 2, 1, 1, 5, 3, 1, 5, 2, 2]
nonzero>=20 통과: 0/13
```
필터를 끄면 MCC 0.184→0.203, AUC 0.652→0.667로 오르지만 **표준편차(±0.04) 안**이다.

### 해석
통계적 농축(OR 17.93, p=2.5e-07)은 실재하지만 **21/541 제형에만 해당해서 전체 성능을 못 움직인다.** B 리포트가 feature 변환 10종에서 본 것과 같은 결과다. 보존제 13종 단독으로는 거의 전부 양성 예측(Spec≈0)이라 모델이 안 된다.

**단, (b) 필터 없음이 네 지표 전부에서 (a)보다 좋다는 점은 발표에 쓸 수 있다** — 유의하지 않지만 방향이 일관되고, 희소 feature를 nonzero 필터로 버리면 안 된다는 교훈이 된다.

---

## 5. 파일과 재현 정보

### 5-1. GitHub
```
https://github.com/tjdbf061106-sketch/COS
clone: GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/tjdbf061106-sketch/cos
```
폴더: `skin irritation/`, `eye irritation/`, `modeling/`, `pretraining/`, `shampoo_demo/`, `선행연구/`, `세팅/`, `주차 별 진행상황/`

`skin irritation/` 주요 파일:
| 파일 | 내용 |
|---|---|
| `skin_irritation.xlsx` | ICE 원본 |
| `skin_filtered.csv` | 1,799행 × 28열. formulation_id, casrn, chemical_name, **percent_active_ingredient**, unknown_pct, smiles 등 |
| `skin_formulation_level_with_comp_group.csv` | 544행. formulation_id, **final_label**(EPA Cat 1~4), comp_group_id, is_composition_twin |
| `skin_vectorized.csv` | 544 × 2,289. 메타 24 + desc_* 217 + fp_w_* 2048 |
| `casrn_smiles_cache.csv` | **270종** casrn/cid/smiles/status |
| `casrn_descriptors_full.csv` | CASRN별 descriptor |

주의: 노트북이 쓰는 `work/` 모듈(`cosmetic_domain.py`, `transfer.py`, `domain_gap.py`, `ceiling.py` 등)과 **화장품 27종 데이터는 레포에 없다** — 로컬 `C:/Users/DS/COS/work`에만 있음

### 5-2. ChemExpo 파일 (수동 다운로드본)
- `ChemExpo_PUCs_20261105.csv` — 477행, PUC 메타데이터
- `ChemExpo_Personal_care_products_and_weight_fractions_20261105.csv` — 36,174행, PUC 137만

### 5-3. 재현 코드 (exp_biocide.py)
```python
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score

P = ".../skin irritation/"
THR = 0.30
SEEDS = [0, 1, 2, 3, 4]

# 화장품과 공유되는 보존제·살균제 계열 13종
BIOCIDE = ["2682-20-4", "26172-55-4", "52-51-7", "6440-58-0", "118-52-5",
           "16079-88-2", "89415-87-2", "55406-53-6", "13463-41-7", "88-04-0",
           "7173-51-5", "5538-94-3", "32426-11-2"]

def norm(s):
    m = re.match(r"^0*(\d{2,7})-(\d{2})-(\d)$", str(s).strip())
    return f"{int(m.group(1))}-{m.group(2)}-{m.group(3)}" if m else None

def load():
    d = pd.read_csv(P + "skin_vectorized.csv", low_memory=False)
    ing = pd.read_csv(P + "skin_filtered.csv", low_memory=False)
    ing["n"] = ing["casrn"].map(norm)
    for cas in BIOCIDE:
        sub = ing[ing["n"] == cas]
        d[f"bio_{cas}"] = d["formulation_id"].isin(set(sub["formulation_id"])).astype(float)
        pct = sub.groupby("formulation_id")["percent_active_ingredient"].max()
        d[f"biopct_{cas}"] = d["formulation_id"].map(pct).fillna(0.0)
    base = ["MolWt","LogP","TPSA","HBD","HBA","RotatableBonds","AromaticProportion","ESOL_LogS"]
    base += [c for c in d.columns if c.startswith(("desc_", "fp_w_"))]
    d = d.dropna(subset=["binary_label"]).dropna(subset=base)
    return d.reset_index(drop=True), base, [f"bio_{c}" for c in BIOCIDE], [f"biopct_{c}" for c in BIOCIDE]

def run(d, cols, model="lr", nonzero_filter=True):
    y = d["binary_label"].astype(int).to_numpy()
    g = d["comp_group_id"].to_numpy()
    X = d[cols].fillna(0).to_numpy(dtype=np.float64)
    rows = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        for tr, te in cv.split(X, y, groups=g):
            Xtr, Xte = X[tr], X[te]
            if nonzero_filter and Xtr.shape[1] > 50:
                keep = (Xtr != 0).sum(axis=0) >= 20
                if keep.sum() < 5: keep = np.ones(Xtr.shape[1], bool)
                Xtr, Xte = Xtr[:, keep], Xte[:, keep]
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            m = (LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, random_state=seed)
                 if model == "lr" else
                 RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                        min_samples_leaf=2, random_state=seed, n_jobs=-1))
            m.fit(Xtr, y[tr]); oof[te] = m.predict_proba(Xte)[:, 1]
        pred = (oof >= THR).astype(int)
        tp = ((pred==1)&(y==1)).sum(); fn = ((pred==0)&(y==1)).sum()
        tn = ((pred==0)&(y==0)).sum(); fp = ((pred==1)&(y==0)).sum()
        rows.append(dict(sens=tp/(tp+fn), spec=tn/(tn+fp),
                         mcc=matthews_corrcoef(y, pred), auc=roc_auc_score(y, oof)))
    r = pd.DataFrame(rows)
    return {k: (r[k].mean(), r[k].std()) for k in ["sens","spec","mcc","auc"]}
```

### 5-4. 겹침 재계산 코드
```python
# ICE 270종 vs ChemExpo 큐레이션 성분 겹침
ice = pd.read_csv(".../casrn_smiles_cache.csv"); ice["n"] = ice["casrn"].map(norm)
ce  = pd.read_csv("ChemExpo_Personal_care_...csv", low_memory=False)
ce["n"] = ce["Curated CAS Number"].map(norm)
OV = sorted(set(ice["n"].dropna()) & set(ce["n"].dropna()))   # -> 22종
```

---

## 6. 다음 수 (우선순위)

1. **ChemExpo 하위 PUC 전체 수집 후 겹침 재계산.** 현재 1,341종은 PUC 137만. makeup 29,248 / hair styling 16,904 등을 포함하면 성분 어휘가 수천 종이 되고 겹침 22종이 늘어난다. **현재 결과가 과소추정일 가능성이 높아 이게 1순위**
2. **ChemExpo Pesticides 2,009건 ↔ ICE 544제형 식별자 매칭 시도.** 되면 병목 ②(조성 기재 31.5%)를 직접 공격. 조사 전체에서 가장 큰 수가 될 수 있음
3. **구조 유사체 겹침** — CASRN 완전일치가 아니라 Morgan FP 유사도(예: Tanimoto ≥0.7) 기준으로 겹침 재계산. 22종이 더 늘어날 것
4. **타깃 도메인을 ChemExpo로 교체** — 화장품 27종 대신 ChemExpo 2,254제품(성분 목록 기준, 균등 가중)으로 도메인 갭/AD 재분석. 노트북의 AD가 27개 기준이라 집계 방식에 따라 0/27~24/27로 흔들렸던 문제가 완화됨
5. **MIT 함유 화장품 188제품에 제로샷 예측** → 제로샷을 처음으로 의미 있는 표본에서 평가
6. **nonzero 필터 재검토** — B 리포트의 `nonzero≥20` 권고는 희소하지만 신호 있는 feature를 버린다. 보존제처럼 1~6개 제형에만 나타나는 feature를 살릴 별도 경로 필요

---

## 7. 내가 이번에 틀렸던 것 (같은 실수 반복 방지)

| 주장 | 정정 |
|---|---|
| "comp_count=1은 순물질과 같다" | 아니다. ICE comp_count=1은 "선언 활성성분 1개 + 미기재 inert 약 69%". 순물질(100%)과 다른 객체. `undeclared_pct`가 출처 지시자가 되어 모델이 화학 대신 데이터 출처를 학습함 |
| "순물질 양성 편입하면 비용 0" | 아니다. eye 순물질 양성률 58% vs 혼합물 19%라 교란 발생 |
| "Raabe & Costin에서 양성 37건 확보" | 리뷰 논문. 제형별 표 없음. 25건은 Kolle 재인용, 207건은 Corvaro 재인용 |
| "Cat2→Cat4 뒤집힘 1/5 = 제형 라벨 노이즈" | 단일성분 물질 재현성 연구(Rooney 2021) 결과. 제형 아님 |
| "Corvaro가 EPA 라벨 보유" | EPA 등급은 있으나 집계만. 제형별 표 없음. 양성 5건 |
| "ChemExpo로 27 → 수천 종" | 함량 기준으로는 27 → **11개**. 성분 목록 기준으로는 2,254제품 |
| "Donahue 2011로 피부 베이스라인" | Donahue는 **안구 전용**. 피부는 Lechuga 2025(HLB·농도·이온특성→Zein)이고 둘 다 함량 필수라 11개에만 적용 가능 |

---

## 8. 프로젝트 문서 목록

| 경로 | 내용 |
|---|---|
| `claude/B_feature_threshold_cv_results.md` | feature 차원축소·threshold·반복 CV 재검증. 권고: nonzero<20 제거(444차원) + threshold 0.30 |
| `claude/C_formulation_irritation_data_sources.md` | 제형 단위 자극성 데이터 소스 조사 (안구 포함, 15개 소스) |
| `claude/C_skin_formulation_data.md` | 피부 전용. Corvaro 원문 확인 결과 포함. 조달 노선 종료 결론 |
| `claude/D_shared_ingredient_transfer_finding.md` | 공유 성분 22종 발견. 본 문서 §3의 상세 |
| `COS_실험.ipynb` | 메인 실험 노트북 (10절 결론 포함) |
| `중간보고서_COS (1).docx` | 제출본 |
| `modeling/modeling/src/*.py` | PyTorch 멀티태스크 하네스 |
