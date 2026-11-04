# Pretraining: mixture-003

혼합물 실험 `mixture-003`에서 사용한 **성분 구조 인코더와 혼합물 자극성 예측기**입니다. 이전 `bc-audit-005` 체크포인트 패키지를 교체했습니다.

1. **추가 사전학습(MLM)**: 화장품 성분 SMILES의 가려진 토큰을 복원합니다. 자극성 라벨을 사용하지 않습니다.
2. **혼합물 예측**: 성분별 ChemBERTa 표현을 평균 또는 함량 가중평균으로 합치고, 혼합물 전체의 피부·눈 EPA 라벨로 분류기를 학습합니다. 성분에 혼합물 라벨을 복사하지 않습니다.

MLM은 `mixture-002`에서 학습했고, 코퍼스·분할·제외 성분과 체크포인트 해시가 동일함을 확인해 `mixture-003`에서 재사용했습니다. 이번 업로드를 위해 MLM을 다시 학습한 것은 아닙니다. 인코더는 혼합물 지도학습 중 고정했습니다.

## 저장 내용

| 경로 | 내용 |
|---|---|
| `best/` | 추가 사전학습 가중치와 설정·토크나이저 8개 파일 |
| `mixture_heads/` | 피부·눈 × 단순 평균·함량 가중평균, 총 4개 혼합물 분류기 |
| `mixture_predictor.py` | 감사가 끝난 성분 입력으로 저장된 모델을 실행하는 코드 |
| `examples/known_mixture.json` | 실제 학습 제형의 입력 예시. 외부 검증 사례가 아님 |
| `input_schema.json` | 384차원 성분 표현, 11개 불확실성 변수와 집계 규칙 |
| `data/mlm_corpus.csv` | 고유 SMILES와 train/validation 구분 |
| `data/excluded_candidate_identities.json` | 추가 MLM에서 제외한 후보 혼합물 성분 identity 그룹 |
| `manifest.json`, `manifest.json.sha256`, `losses.*` | MLM 학습·재로딩 기록과 체크포인트 해시 |
| `checkpoint_expected.json`, `provenance.json` | 기본 모델 출처와 학습 계보 |
| `protocol.json`, `evaluation_summary.json`, `verification_summary.json` | 비교 원칙, 평가 결과, 독립 검산 요약 |
| `requirements.txt`, `environment_lock.json` | 실행 환경의 패키지 버전 |
| `package_manifest.json` | 배포 파일 해시와 패키지 실행 검증 결과 |

원본 XLSX, 대형 특징 행렬, 교차검증 모델 390개, 다른 기준 모델과 탐색 분석 모델은 포함하지 않습니다. 저장된 모델의 재사용 패키지이며, 원자료부터 전체 실험을 다시 실행하는 코드 묶음은 아닙니다.

## 모델과 데이터

- 기본 모델: [DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM), revision `ed8a5374f2024ec8da53760af91a33fb8f6a15ff`.
- 추가 MLM: 고유 SMILES **1,144개**, 표준화 identity **1,140그룹**. 학습 1,030개, 검증 114개.
- 후보 혼합물의 구조 identity 263그룹을 새 MLM에서 제외했습니다. 원래 대규모 사전학습 자료의 중복 여부는 미확인입니다.
- 3 epochs, 195 optimizer steps, batch 16, learning rate `5e-5`, seed `20261104`.
- 고정 검증 마스크의 MLM 손실: `0.163980 → 0.114275`. 다른 코퍼스·검증 마스크를 쓴 과거 실행의 손실과 직접 비교하지 않습니다.
- 혼합물 분류: eye **547개**(양성 98), skin **494개**(양성 23). EPA I/II=1, III/IV=0이며 0은 완전 무자극을 뜻하지 않습니다.
- 분류기는 학습 자료에만 맞춘 imputer/scaler와 balanced LogisticRegression입니다. 배포 분류기는 endpoint·집계 방식별 전체 적격 자료로 재적합했습니다.

## 사용

기존 연구 환경의 패키지 버전은 `environment_lock.json`에 기록되어 있습니다. 저장소 루트에서 실행합니다.

```bash
python pretraining/mixture_predictor.py --input pretraining/examples/known_mixture.json --endpoint eye_epa --pooling weighted
```

입력은 **중복·성분 정체·구조 출처 점검을 마친 제형별 성분 목록**입니다. 표현 가능한 SMILES뿐 아니라 미해결 성분과 알려진 0% 성분도 보존해야 불확실성 변수를 재구성할 수 있습니다. 이 코드는 원료명을 조회하거나 성분 정체를 독립적으로 인증하지 않습니다.

네 분류기는 함량이 알려진 주 분석용입니다. 미상 함량은 오류로 반환하며 0이나 1로 채우지 않습니다. 함량 미상에 대한 별도의 평균 전용 탐색 모델은 포함하지 않습니다.

연결된 유기 구조의 입체정보 미지정은 그대로 인코딩하고 표시합니다. 염·다성분 구조, 분포형 원료, 확인되지 않은 조회 구조 등은 학습 프로토콜에 따라 미해결 성분으로 남깁니다. 가중평균은 표현 가능한 성분의 공개 함량 합으로 정규화하며, 미공개 잔여 처방을 복원한 값이 아닙니다.

`best/`는 `AutoModelForMaskedLM.from_pretrained("pretraining/best", local_files_only=True)`로도 불러올 수 있습니다. 혼합물 예측에는 성분 집계와 `mixture_heads/`의 분류기도 필요합니다.

## 평가와 한계

단위 테스트 21개와 독립 검산 44개가 통과했습니다. 원본 최종 CSV 입력, 새 특징 벡터, 교차검증 모델 390개, OOF 예측 40,824건과 성능 계산을 확인했습니다. 배포 패키지 실행 검증은 `package_manifest.json`에 따로 기록합니다.

조성 그룹·성분 identity·Murcko scaffold를 분리해 피부·눈을 따로 평가했습니다. 피부 주 분석의 scaffold 시험 fold 2개는 한 클래스뿐이어서 판별 성능 지표를 계산하지 않았습니다. 미공개 성분의 실제 중복까지 보장하지 않으며, scaffold 분리는 혼합물 지도학습 기준입니다.

원자료는 **FIFRA/ICE 농약 제형의 부분 공개 활성성분 조성**입니다. 미공개 성분, 시험 농도·노출 시간, percentage의 질량/부피 기준이 남아 있습니다. 실제 화장품 제형의 독립 외부 검증은 확보하지 못했습니다. 추가 사전학습의 일관된 성능 향상도 입증되지 않았습니다.

`evaluation_summary.json`의 성능은 교차검증 fold 모델의 OOF 결과입니다. 배포한 전체자료 재적합 분류기의 별도 외부 성능이 아닙니다. 예시 입력의 출력은 로딩·계산 재현 확인용입니다.

## 파일 확인

```python
from pathlib import Path
import hashlib, json
root = Path("pretraining")
manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
for name, expected in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print("Package files verified")
```
