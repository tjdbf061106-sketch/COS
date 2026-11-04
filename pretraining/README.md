# Pretraining

화장품 성분 SMILES로 **MLM(Masked Language Modeling) 추가 사전학습**한 ChemBERTa 체크포인트입니다. 실행 `bc-audit-005`에서 선택한 모델과 재사용에 필요한 설정·토크나이저·학습 기록을 보관합니다.

이 모델은 SMILES의 가려진 토큰을 예측합니다. 제형의 피부·안자극성 분류 헤드는 포함하지 않으며, MLM 검증 손실 감소가 화장품 자극성 예측 성능 향상을 입증하지는 않습니다.

## 저장 내용

| 경로 | 내용 |
|---|---|
| `best/model.safetensors` | 추가 사전학습 후 선택한 가중치, 13,721,840 bytes |
| `best/config.json` | `RobertaForMaskedLM` 모델 설정 |
| `best/`의 나머지 6개 파일 | 토크나이저, vocabulary, special tokens |
| `losses.csv`, `losses.json` | 에포크별 학습·검증 손실, 학습 전 검증 손실 |
| `manifest.json`, `manifest.json.sha256` | 학습 설정, 선택 에포크, 원본·산출물 해시, 저장 후 재로딩 검증 기록 |
| `checkpoint_expected.json` | 기본 모델의 출처·고정 revision·검증 정보 |
| `environment_lock.json` | 학습 당시 Python·패키지 버전·CPU 환경 기록; 컴퓨터별 절대경로 제외 |

기존 기본 가중치는 원본 출처에서 받을 수 있으므로 중복 보관하지 않습니다. 원본 코퍼스, 학습 코드, optimizer 상태와 다른 실험 산출물은 포함하지 않습니다. 이 폴더는 저장된 모델의 재사용용이며, 전체 재학습을 재현하거나 중단된 optimizer 상태를 그대로 복원하는 패키지는 아닙니다.

## 학습 기록

- 기본 모델: [DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)
- 기본 revision: `ed8a5374f2024ec8da53760af91a33fb8f6a15ff`
- 파라미터 수: 3,428,808개. 모델 이름의 `77M`을 파라미터 수로 해석하지 않습니다.
- MLM 코퍼스: 학습 1,014구조, 검증 114구조. 각 코퍼스의 SHA-256은 manifest에 보존됩니다.
- 3 epochs, 192 optimizer steps, batch size 16, learning rate `5e-5`, seed 42.
- 학습 입력 최대 길이 128토큰. 특수·패딩 토큰을 제외한 15% 마스킹, 80/10/10 치환.
- 고정된 검증 마스크의 토큰 가중 손실로 epoch 3을 선택했습니다.

| 단계 | 학습 손실 | 검증 손실 |
|---|---:|---:|
| 추가 학습 전 | — | 0.704944 |
| Epoch 1 | 0.499904 | 0.618820 |
| Epoch 2 | 0.398025 | 0.573998 |
| Epoch 3, 선택 | 0.362902 | 0.530612 |

## 로컬에서 불러오기

`torch`, `transformers`, `tokenizers`, `safetensors`가 설치된 환경에서 저장소 루트를 작업 디렉터리로 사용합니다. 학습 당시 버전은 `environment_lock.json`에 기록되어 있습니다.

```python
from pathlib import Path
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = Path("pretraining/best")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(checkpoint, local_files_only=True)
model.eval()

# 가려진 SMILES 토큰에 대한 logits를 계산하는 로딩 예시입니다.
inputs = tokenizer(
    f"CC{tokenizer.mask_token}",
    return_tensors="pt",
    truncation=True,
    max_length=128,
)
with torch.inference_mode():
    logits = model(**inputs).logits
print(logits.shape)
```

## 파일 무결성 확인

아래 코드는 manifest 자체와 manifest에 기록된 체크포인트·손실 파일 10개의 SHA-256을 확인합니다. `.gitattributes`는 이 파일들의 줄바꿈 자동 변환을 막아 원본 바이트를 보존합니다.

```python
from pathlib import Path
import hashlib
import json

root = Path("pretraining")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
expected_manifest = (root / "manifest.json.sha256").read_text().split()[0]
assert digest(root / "manifest.json") == expected_manifest
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for relative_path, expected in manifest["artifact_hashes"].items():
    assert digest(root / relative_path) == expected, relative_path
print("Manifest and all checkpoint/loss files verified.")
```
