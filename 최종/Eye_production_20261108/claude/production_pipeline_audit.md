# Eye production 파이프라인 및 농도 피처 영향 분석

## 확인된 생성 체인

1. `시행착오/2. vectorized/최종/build_vectors.py`가 성분 벡터를 가중평균하고 parquet·메타데이터를 작성한다. `export_eye_vectorized.py`가 이들을 ID 순서 검증 후 CSV로 합친다. `준비/테스트/reference/build_vectors.py`는 보존본이다. 가중평균의 `raw_weights / raw_weights.sum()`은 그대로 유지해야 한다.
2. 병합의 기준 입력은 `준비/vectorized/eye_vectorized.csv` 606행이다. 전체 원본 export에서 이 606행을 선별·복사한 실행 명령까지는 이번 조사로 확인하지 못했다. 병합 시에는 이 파일의 SHA256을 `준비/eye_modeling/protocol.json`의 `input_sha256`과 대조한다.
3. `준비/테스트/prepare_experiment.py` → `outputs/eye_candidate_vectors.csv`에서 기존 증강 54행을 가져온다. V4 13행은 `prepare_v4_integration.py` 및 `merge_v4_workbooks.py` 계열을 통해 최종 후보 엑셀의 `V4특징벡터`에 보존되어 있다. V4는 원문 단위를 검증한 벡터이므로 단순 농도 파서로 재계산하면 안 된다.
4. `준비/production/_eye_merge_audit/merge_eye_production.py`가 606 + 54 + 13 = 673행을 병합한다. 증강에는 `AUG_` ID와 `evaluation_role=augmentation_train_only`를 부여하고, 기존 연결 그래프·제외 노드를 경유한 연결까지 보존한다. 후보의 `comp_known_pct`, `comp_undeclared_pct`, `comp_pct_sum_used` 등을 출력의 `known_pct`, `undeclared_pct`, `pct_sum_used`로 매핑한다.
5. `update_eye_comp_count.py`가 성분 수를 검증된 고유 CAS 기준으로 변경한다.
6. `generate_eye_comp_groups.py`가 원본 CAS:농도 조합으로 composition 그룹을 채운다. **여기의 소수 6자리 농도 표기와 실험 03의 feature 벡터 6자리 반올림은 별개 규칙이다.**
7. 마지막 `_split_group_fix/.../report.json`에 기록된 수정이 검토용 거대 그룹을 분할 연결에서 제거하고, comp/structure 연결 기준으로 425개 그룹을 만든다. 이 마지막 수정의 실행 스크립트는 해당 감사 폴더에서 발견되지 않았다.

### 현재 CSV의 계보 검증

현재 `production데이터/eye_vectorized.csv`의 실제 SHA256은
`811268d4813fcdd68e85458641bf5823239cae04e1fa871766c77e04f0221e7f`로 마지막 split 수정 보고서의 `after_sha256`과 일치했다.

감사 보고서에 기록된 출력 해시 순서:

- 최초 병합: `3b6fe726540554f88051862c5dca0c0b2a97f613ff71af8b400da47932650b74`
- count 수정: `a9788e95700ec39d520f8082326c7d9efc0e3ba74aa086f2cac4f1c59756fc2c`
- composition 수정: `722b42563fd8b4590992ba1be1180e47a16bb3822ff79b219e3c485a25ea32aa`
- split 수정: 위 현재 해시.

따라서 준비 CSV와 production CSV의 해시 차이는 오류의 증거가 아니라 병합 및 후속 메타데이터 수정의 결과다. 모든 과거 입력 파일 해시를 이번 조사에서 재검증한 것은 아니다.

**재현성 문제:** 위 감사 스크립트들은 현재 보관 위치와 `DEST=AUDIT.parent`, `PREP=DEST.parent/'준비'` 상대경로가 맞지 않는다. 보관본을 지금 위치에서 그대로 실행하면 실제 production 입력·출력을 가리키지 않는다. 과거 보고서의 절대경로와 현재 스크립트의 계산 경로를 구별해야 한다. 원본 CSV를 지워서 과거 병합을 재실행하면 안 된다.

## 피처 네 개를 덧붙이면 끝인가?

데이터 재벡터화는 필요 없다. 하지만 CSV 열 추가만으로 모델에는 반영되지 않는다.

- 현재 EPA 학습기 `feature_columns()`는 정확히 `fp_*` 2048개 + `desc_*` 217개, 총 **2265개**만 선택한다. `conc_*`는 무시된다.
- 이전 병합 스키마의 **2280개**와 현재 EPA 입력 **2265개**는 서로 다르다. 이름 명세, 차원 검증, 학습·추론 입력 선택을 함께 바꿔야 한다.
- 실험 11의 실제 정의는 `log10(pct + .001)`, `pct / 100`, `pct >= 90`, `undeclared_pct / 100`이다. 잘린 지시문의 `conc_pct`를 원시 %로 해석하지 않고 실험 11 조합을 그대로 써야 한다.
- 실제 673행에서 known_pct 결측은 16행, pct_sum_used 대체 후에도 9행이 결측이다. undeclared_pct 결측은 17행이다. 실험 11은 후자를 0으로 처리한다.
- 실험 11은 전체 데이터 중앙값으로 9행을 메운다. production CV에서는 중앙값을 **각 inner/outer 학습 fold에서만 fit**해야 한다. 고정된 전체 중앙값을 CSV에 채워 넣으면 검증 전처리 누수가 생긴다. 배포 시에는 최종 학습 중앙값을 모델과 같이 저장해야 한다.
- 농도에 따라 그룹을 다시 풀지 말고, 기본 fp/desc의 6자리 반올림으로 만든 보수적 그룹을 두 비교군에 똑같이 적용해야 한다.

## 증강 67행의 평가 처리

병합 스크립트는 역할 표식만 기록하며 학습 fold를 만들지 않는다. 실험 03의 현재 `load_eye → build_jobs → nested_splits → collect`는 673행 전부를 test/validation에 포함한다. 실험 11도 같은 방식이다. 따라서 기존 +0.030은 606행 train-only production 평가 결과가 아니다.

production 검증에서는:

- outer test와 inner validation, 임계값 선택용 OOF는 기존 평가 606행만 포함한다.
- 증강은 학습에만 넣고 해당 validation/test와 그룹이 연결된 증강행은 그 학습 fold에서도 제외한다.
- paired baseline/농도 arm에 동일한 folds, seed, full grid, 모델과 임계값 정책을 적용한다.
- 증강이 inner validation 또는 threshold OOF에 섞이지 않는지 행 ID와 그룹으로 검사한다.

## 서명·검증 영향

기존 감사 기록의 과거 해시를 새 해시로 덮어쓰면 안 된다. 입력 CSV와 skin CSV의 실행 전후 해시, 기본 feature·행 순서·라벨·역할·그룹 보존, 농도 수식, schema 버전, fold별 결측 대체값, 코드/설정 해시를 새로운 실행 manifest에 남겨야 한다. CSV를 바꾼다면 모델 입력 해시·캐시 키·feature 명세 및 추론 경로도 함께 갱신해야 한다. 현재 실험 03이 원본 CSV를 사용 중이므로 이 조사에서는 production CSV를 변경하지 않았다.

## 비용 추정과 상태

- 농도 변환 자체와 단위 테스트: 1~2시간 규모.
- train-only 중첩 CV, fold 내 중앙값 학습, schema/추론/서명 연결까지 검증 가능한 구현: **반나절 규모**로 추정. 단순 4열 CSV 추가와는 다르다.
- full-grid 반복 실험의 벽시계 시간은 별도이며 측정 전 확정할 수 없다. 축소 실험 6후보에서 full 20후보면 후보 적합 작업은 약 3.3배다. 실제 모델별 후보 수·동시 실행 부하에 따라 달라진다.
- `concentration_features.py`에 네 농도 피처와 학습 중앙값 기반 변환을 구현했다. 기존 실험 11도 동일 수식을 공유하며, 과거 실험용 전체 중앙값 방식이라는 주석을 붙였다. 희석 스케일링은 production 검증에 포함하지 않는다.
- `run_concentration_production.py`는 606행 평가/67행 train-only, 3개 seed, 5×3 중첩 분할, 6개 모델별 20후보 전체 격자, paired baseline/농도 arm을 구현한다. 후보 및 outer 결과는 설정·코드·원본 해시에 묶인 체크포인트로 저장된다. 기존 CSV나 00~11 캐시는 덮어쓰지 않는다.
- 완료 시 실행 디렉터리에 2269개 모델 피처를 명시한 `eye_vectorized_concentration.csv`와 `export_manifest.json`을 생성한다. 이는 전체 학습용 파생 CSV이며, CV에서는 이 CSV를 읽지 않고 원래 농도 메타데이터로 fold별 중앙값을 학습한다. 배포 모델 학습·교체는 이 검증 실행에 포함하지 않는다.
- 검증: ruff 통과, 그룹화/농도 수식/학습 중앙값 격리/증강 분리/실제 sklearn fit/체크포인트 및 실제 CSV export 보존 테스트 총 11개 통과. `--prepare-only`로 전체 production 분할과 설정을 검증했다. 최종 설정 경로: `results/concentration_production/f91f3c88787bec6b/config.json`.
- 실제 LR / 라벨 A / seed 42 / outer fold 0의 baseline·농도 두 arm에서 **각 20후보 전체 격자 적합**이 완료되었다. 각 arm의 학습 550행, 평가 121행과 후보 체크포인트 20개, 평가·임계값 선택에 증강행이 없는 것, 소스/코드 해시 일치를 재검증했다. 농도 arm의 outer 학습 중앙값은 28.895%다.
- **전체 360개 fold 작업 중 저장된 106개를 검증했으며 최종 성능 결과는 아직 없다.** 체크포인트 체크섬·각 20후보 완료·평가 및 임계값 선택의 증강행 제외·기록된 분할 일치·eye/skin 입력 해시 보존을 확인했다. 전체 학습은 계속 실행 중이다.
- 실험 03의 반올림 재실행은 완료됐다. 전체 ΔMCC는 1판 -0.007519에서 2판 -0.014205로 변했다. 라벨 A는 +0.000925 → -0.009772, B는 -0.015964 → -0.018638이다. 제공된 기준의 중간 구간이므로 전체 재실행 필요 여부를 확정할 수 없으며, 영향이 거의 없다는 기존 결론도 확정할 수 없다. 결과는 `03_grouping_fix_edition_comparison.csv`와 `03_grouping_fix_verdict.json`에 저장했다.
- 00~02, 04~11 캐시는 아직 1판 결과다. 기존 노트북·보고서의 paired 비교는 03 1판에 고정하여 2판 분할과 섞이지 않게 했다. 03은 증강을 포함한 673행 CV이므로 이 결과를 production train-only 성능 검증 완료로 해석하면 안 된다.
