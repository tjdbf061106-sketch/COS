# E. EPA 제형 라벨의 근거 품질 — 라벨 노이즈의 기전

작성 2026-11-06. 대상: 독립수집 `skin_보강후보_통합` 165건(v2 57 + v3 108)의 양성 10건 + 신규 수집 4건.

## 1. 요약

EPA 제형 skin 등급은 단일 시험 결과가 아니라 **규제 판정**이다. 독립수집 양성 10건의 근거를 1차 EPA 문서로 확인한 결과, 단일·무이의 시험 등급을 가진 건은 4건이고 그중 3건은 시험 ID 없는 SDS 유래다. 나머지 6건에는 시험 간 불일치, 기관 간 불일치, 미검토 수용, 타제품 bridging이 있다. 이것이 병목 ①(라벨 노이즈 κ≈0.1)의 기전이다.

## 2. 근거 상태별 양성 10건

| 후보 | 제품 | Cat | 근거 상태 |
|---|---|---|---|
| N068 | Safer Moss and Algae | II | 같은 제품 피부시험 2건이 **Cat I과 Cat IV**. EPA가 가중해 II 부여 |
| N101 | Safer Insecticidal Soap | II | 피부시험 5건 중 1건 Cat I, 4건 더 약함. Cat II는 EPA의 "concession" |
| N065 | Demon EC | II | EPA Cat II vs **CDPR Cat III** (기관 간 불일치) |
| N066 | Troysan Polyphase 587 | I | **self-validated** — Cat I이므로 EPA 미검토 수용 |
| N067 | Troysan Polyphase 597 | I | skin은 acceptable, dermal·inhalation은 타제품 bridging |
| N100 | M-Pede | I | skin 단일시험, 나머지 acute는 타제품(42697-1) |
| N095 | HD-64 | I | 시험번호 미기재 |
| N021 | Spectracide | II | SDS 요약, 시험 미명시 |
| C002 | RESOURCE | II | SDS 표, 시험번호 없음 |
| C004 | DISTANCE | II | SDS 표, 시험번호 없음 |

N068이 가장 강한 사례다. 동일 제형·동일 엔드포인트에서 4점 척도상 **최대 불일치**(I vs IV)가 관측됐다.

## 3. 등급 생성 경로 4종

신규 수집에서 네 번째 경로가 추가로 확인됐다.

1. 검토된 단일 시험
2. 불일치 시험들의 기관 가중 판정 (N068, N101)
3. self-validation — Cat I이라서 미검토 (N066)
4. **pH 규칙 waiver** — 피부 시험 없이 등급 부여. `EPA 5813-RNN` PUMA는 pH 12.066으로 skin·eye 시험이 waive되고 40 CFR 158.500에 따라 Cat I이 자동 부여됐다. 관찰된 피부 반응이 아니다.

경로 3·4는 **피부 관찰 없이 생성된 양성 라벨**이다. 모델에 넣으면 화학이 아니라 규제 규칙을 학습 대상으로 삼게 된다.

## 4. 기존 정정 사항과의 연결

인계문서 §7에서 "Cat2→Cat4 뒤집힘 1/5 = 제형 라벨 노이즈"는 Rooney 2021 단일성분 재현성 연구 결과이므로 제형 근거가 아니라고 정정했다. 본 문서가 그 자리를 채우는 제형 수준 1차 근거다. 인용 출처는 외부 논문이 아니라 본 팀이 수집한 EPA PPLS·reregistration notice다.

## 5. 한계 (반드시 함께 보고)

- **ICE 541제형에 직접 적용 불가.** ICE에는 MRID·근거 경로 메타데이터가 없어 κ≈0.1이 waiver·self-validation 때문이라고 ICE에서 측정할 수 없다.
- 보일 수 있는 범위는 "관찰 없는 등급 부여 경로가 실재하며, 독립수집 양성 10건 중 6건이 단일 검토시험이 아니다"까지다. 기전의 **존재**는 확인되고 ICE 내 **비율**은 추정이다.
- 표본 10건. 양성에 한정한 검토이며 음성 쪽(특히 `secondary_SDS_summary` 56건)은 미검토다.
- 등급 역산 금지: signal word는 5개 경로 중 최악 등급으로 결정되므로 label 문구에서 skin 등급을 추정할 수 없다. acute toxicity review 표가 있는 건만 등급 확정으로 취급해야 한다.

## 6. 다음 수

1. 165건 전체에 근거 경로 플래그(검토단일/기관가중/self-validated/waiver/SDS유래) 부여
2. 수집 스키마에 waiver·self-validation 플래그 상설화
3. `EPA 67702-22` Neudorff 비누 농축 47% 등급 확정 — Cat I/II면 47%·49%·49% 세 제형의 등급이 갈리는 사례가 되어 §2에 추가

## 부록. 신규 수집 4건

양성 2건 채택(`skin_보강후보_v4_신규.xlsx`). 기존 165건과 등록번호·제품명·별칭·출처URL 네 축 전부 불일치.

| 등록번호 | 제품 | 조성 | skin Cat | 근거 |
|---|---|---|---|---|
| EPA 3377-40 | ALBEMARLE BEQ442-8 Biocide | ADBAC 40.0% + ADEBAC 40.0% + inert 20.0% | I | MRID 451135-01, 실제 rabbit 시험 |
| EPA 10324-80 | MAQUAT 5.5-M | 4급암모늄 4종 합 5.5% + inert 94.5% | I | MRID 443584-02, Acceptable |

보류 2건: `5813-RNN` PUMA(waiver 유래), `67702-22` Neudorff(등급 미확정).

**부수 소득**: 4급암모늄 양성 3건의 함량이 5.12%(N095) / 5.5% / 80%인데 전부 Cat I이다. 함량이 아니라 보유 여부가 등급을 지배하며, §4 표에서 이진 지시자 (c)에 함량을 더한 (d)가 더 나빴던 것(MCC −0.020 → −0.066)과 방향이 일치한다. 이진 feature가 맞는 형태라는 근거.

**기각 노선**: RED/BRAD 집계(ADBAC RED EPA739-R-06-009 등)는 시험물질이 a.i. 1건뿐이고 제형별 행이 없다(Corvaro와 동일 구조). EPA-HQ-OPP-2016-0093 회고분석도 제품별 행 없음.
