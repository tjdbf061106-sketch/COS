# ChemExpo Personal care 조성 수집 (2026-11-06)

팀원(장다예) 인계문서(`COS_handoff_20261106.md` §6-1)의 "①ChemExpo 하위 PUC 전체 수집" 작업.
팀원 환경에서는 EPA 사이트 접속이 안 돼 PUC137(일부 하위 카테고리만, 2,254제품) 하나만 받았는데,
이 세션에서는 직접 접속이 가능해 **Personal care 전체 하위 카테고리**를 긁어왔다.

## 수집 방법

```
https://comptox.epa.gov/chemexpo/dl_co_chemicals/   # 전체 PUC 조성 bulk 파일(63MB zip, 압축해제 시 1GB)
```
위 파일을 받아 `PUC General Category == "Personal care"`로 필터링(청크 단위 처리, 원본 1GB는
용량 문제로 로컬/레포에 보존하지 않음 — 필요시 위 URL에서 재다운로드 가능).

## 결과

| 항목 | PUC137만(팀원 1차) | Personal care 전체(이번) |
|---|---|---|
| 고유 제품 | 2,254 | **77,535** |
| 함량 있는 행 비율 | 1.8% | 0.3%(비율은 더 낮아짐 — 다른 하위카테고리는 공개도가 더 낮음) |
| **완전 조성 제품**(성분≥5, 함량합 0.80~1.05) | 11 | **108** (10배) |

하위 카테고리(PUC Product Family)별 완전조성 제품 분포: hair styling 14, body hygiene 13,
fragrance 12, general moisturizing 10, nails 9, oral/dental 7, specialized bath 7,
makeup 6, hair coloring 5, sunscreen 4, 기타 소수.

## 파일

| 파일 | 내용 |
|---|---|
| `ChemExpo_PUCs.csv` | PUC 분류체계 메타데이터(477행) |
| `ChemExpo_personal_care_full_composition_108.csv` | 완전 조성 제품 108개의 전체 성분 행(1,297행) — **다음 분석(②성분어휘/③도메인갭/④HLB)의 입력 데이터** |

## 한계

- 함량 비율 자체는 PUC137보다도 낮아짐 — "범위를 넓히면 품질이 좋아진다"가 아니라 "절대 수가 늘어난다"는
  의미. 화장품 라벨이 구조적으로 %를 안 적는 문제(§2-3)는 그대로.
- "완전 조성" 기준(성분≥5, 합 0.80~1.05)은 팀원이 1차 수집에서 쓴 것을 그대로 적용. 더 느슨한/엄격한
  기준으로 다시 볼 수 있음.
- 원본 1GB bulk 파일은 로컬에 보존하지 않음 — 재현 필요시 위 URL에서 재수집.
