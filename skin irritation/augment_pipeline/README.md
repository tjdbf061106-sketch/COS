# skin 데이터 보강 파이프라인 — `skin_보강후보_통합.xlsx` 통합 (2026-11-06)

제품 SDS 급성독성표에서 수집한 보강후보 125건(`skin_보강후보_통합.xlsx`, `추가후보` 시트)을
기존 `skin_vectorized.csv`(544건)에 합쳐 652건으로 늘린 과정. 최종적으로 **성능은 중립이었지만
("더 많은 실제 데이터 반영"이 성능 지표보다 중요하다는 판단으로) 채택 확정**됨.

## 실행 순서

```
py 01_resolve_cas_by_pubchem.py    # 신규 CASRN -> PubChem(xref/RN -> CID -> IsomericSMILES)
py 02_resolve_name_by_pubchem.py   # CAS 없는 성분 -> PubChem 이름검색(제네릭 용어는 자동 제외)
py 03_build_augmented_dataset.py   # descriptor/FP 계산 + 투입비율 가중평균 집계 -> skin_vectorized_augment_125.csv
```

이미 한 번 실행되어 `casrn_smiles_cache.csv`/`casrn_descriptors_full.csv`/`skin_vectorized.csv`에
결과가 반영돼 있음(원본은 `*_pre_augment_*` 파일로 백업됨). 다시 돌리면 이미 병합된 걸 다시
찾으려 해서 대부분 "신규 0건"으로 나오는 게 정상 — 재현 기록용으로 보존.

## 핵심 발견 — 함량 텍스트 파싱 버그

성분함량 시트의 `함량종류` 컬럼 중 "원문 참조"(457건 중 265건, 58%)는 **값이 없다는 뜻이 아니라
`함량_원문`에 텍스트로 숫자가 그대로 있다는 뜻**이었다(예: "15%"). 처음엔 이걸 놓쳐서 전부 NaN
처리 -> 동일가중치 폴백으로 조성 가중치가 깨졌고, 양성 후보 다수(N021/N066/N067/N095)가 영향을
받았다. `pct_parser.py`(OCR 오류·범위·퍼센트 접미사 처리, 파싱 성공률 97%)로 수정.

## 최종 결과

- 125건 중 111건 구조 해결 성공(양성 6/9, 음성 105/116). 해결 불가 14건은 전부 UVCB(지방산칼륨염
  비누 등 단일 SMILES로 표현 불가능한 성분만 가진 경우).
- 반복 30회 group 5-fold 기준, 원본(541건) vs 증강(652건) 성능 차이는 통계적으로 없음(|t|<1.2,
  `skin irritation/rf_feature_compare/`의 비교 방법론과 동일).
- 성능 향상 근거는 없었지만 실데이터 반영 자체를 우선해 채택.

## 파일

| 파일 | 역할 |
|---|---|
| `pct_parser.py` | 함량_원문 텍스트 -> 숫자(%) 파서 |
| `01_resolve_cas_by_pubchem.py` | CASRN -> SMILES(PubChem xref) |
| `02_resolve_name_by_pubchem.py` | 성분명 -> SMILES(PubChem 이름검색) |
| `03_build_augmented_dataset.py` | descriptor/FP 계산 + 제형 단위 가중평균 집계 |

## 백업된 원본 파일 (레포 루트 `skin irritation/`)

- `skin_vectorized_pre_augment_544.csv`
- `casrn_smiles_cache_pre_augment_270.csv`
- `casrn_descriptors_full_pre_augment_264.csv`
