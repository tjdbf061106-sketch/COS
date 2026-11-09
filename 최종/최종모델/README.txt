Eye A 최종 모델 — 예측 실행 패키지

최종 모델은 Logistic Regression + 성분 함량 가중평균이다.
양성(1)은 EPA I·II, 음성(0)은 EPA III·IV다. 음성이 모든 눈 자극의 부재를 뜻하지는 않는다.
별도 농도 특징 없이 구조 특징 2265개를 사용한다.

확정 설정
- C=0.0001, solver=lbfgs, max_iter=5000, tol=0.0001, random_state=42
- 학습 집합별 균형 sample_weight 사용
- 판정: 양성 클래스 점수 >= 0.4814867474694138이면 1
- 결측값 0 대체 + StandardScaler + LR + 특징 순서 + 임계값을 joblib 파일에 함께 저장
- 전체 673행(기존 606 + 추가 67)으로 최종 재학습: 양성 125 / 음성 548

실행
검증 환경은 Python 3.14.6이며, 사용한 패키지 버전은 requirements-lock.txt에 고정되어 있다.
이 폴더에서 실행한다. Windows에 py 런처가 있으면 python 대신 py를 사용할 수 있다.

python -m pip install -r requirements-lock.txt
python verify_package.py
python predict.py --input "input_features.csv" --output "eye_A_predictions.csv"

input_features.csv에는 feature_names.json의 2265개 열이 있어야 한다.
입력 열 순서는 이름으로 자동 정렬한다. formulation_id가 있으면 출력에 보존한다.
출력 파일은 새 파일 이름으로 지정한다.
출력: eye_A_score, eye_A_prediction, decision_threshold.
pipeline.predict()의 기본 임계값 0.5 대신 동봉한 predict.py의 최종 판정을 사용한다.

SMILES와 성분 함량으로 예측하기
이 폴더가 Python 모듈 경로에 있는 환경에서 다음 API를 사용할 수 있다.

from predict import load_bundle, vectorize_components, predict_features
bundle = load_bundle()
features = vectorize_components(bundle['feature_names'], ['CCO', 'O'], [20, 80])
result = predict_features(bundle, features)

위 구조·함량은 API 호출 예시다. 함량은 구조가 확인된 성분 사이에서 정규화한다.
다성분 입력은 상대 함량이 필요하다. 미공개 성분의 구조나 함량을 임의로 만들지 않는다.
단일 성분은 함량을 생략할 수 있다. A 모델에는 절대 농도 특징이 없으므로 같은 구조의
단일 성분은 절대 농도를 바꾸어도 동일한 입력 벡터가 된다.

선정·평가 근거
LR/RF/XGBoost × 가중평균/특징별 최댓값을 동일 그룹 분할로 비교하여 A의 LR+가중평균을 선택했다.
기존 606행을 평가하고 추가 67행은 학습 전용으로 사용한 3 seeds × outer5/inner3 결과:
민감도 0.70435 / 특이도 0.67753 / MCC 0.30633 / ROC-AUC 0.71893.
comparison_metrics.csv에는 A의 전체·단일표현성분·다성분 하위집단 비교 결과를 담았다.
단일표현성분은 구조가 표현된 성분이 하나라는 뜻이며, 실제 순물질이라는 뜻은 아니다.

최종 설정은 기존 LR 후보 20개를 3 seeds × 5 grouped folds로 다시 비교해 정했다.
추가 67행은 설정 선정 중에도 검증에 넣지 않았고, 검증 그룹과 연결된 추가행은 학습에서 제외했다.
hyperparameter_search.csv의 AUC는 설정을 고르는 개발 점수이며 새 독립 테스트 성능이 아니다.
임계값은 고유 606행의 반복 OOF 예측 1818개로 정했다. 반복 예측은 독립 표본 1818개가 아니다.
추가 자료는 평가가 금지된 것이 아니라 이번 실험에서 학습 전용으로 정한 것이다.
673행 전체를 대상으로 한 교차검증 성능은 이 패키지에서 보고하지 않는다.

검증
verification.json은 원본 최종 학습의 감사 기록이다: 후보 AUC·선정·임계값 재계산,
673행 전처리 통계 확인, 저장 전후·별도 프로세스 예측 일치, 원자료 26행의 특징 재구성을 통과했다.
verify_package.py는 파일 해시, 실행 환경, 가상 입력 6개의 특징·점수·판정, 별도 폴더에서의
예측 실행을 확인한다. 이 가상 입력 검사는 성능 평가가 아닌 소프트웨어 동작 검사다.
모델 바이너리는 감사가 완료된 원본 최종 모델과 같은 SHA256을 유지한다.

파일 안내
- eye_A_LR_wmean.joblib / predict.py: 최종 모델과 예측 코드
- requirements-lock.txt: 실행 패키지 버전
- metadata.json / feature_names.json: 학습 정의·설정과 특징 순서
- selected_parameters.json / hyperparameter_search.csv: 최종 설정 선정 근거
- comparison_metrics.csv: A의 6조합 비교 성능
- verification.json: 원본 학습·저장 감사 결과
- smoke_test_cases.json / verify_package.py / checksums.json: 패키지 단독 실행 확인

예측에는 이 폴더와 명시된 실행 환경을 사용한다. 기존 실험 폴더를 불러오지 않는다.
학습 캐시·전체 학습 자료·실험용 노트북은 이 실행 패키지에 포함하지 않았다.
기존 Eye 연구 자료는 ../Eye_production_20261108/ 폴더에서 별도로 보존된다.
이번 최종 학습 원본은 로컬의 final_A_LR_wmean_20261109 폴더에 보존되어 있다.
전체 학습을 새 PC에서 재현하는 패키지는 아니며, 서비스 자동 배포도 수행하지 않았다.
독립 화장품 외부 성능과 보정된 실제 자극 확률은 아직 검증하지 않았다.
