# 팀 공유용 Claude 프롬프트 

> 교수님 피드백 "Claude 활용 workflow" 3단계 산출물.
> 팀원 누구든 이 프로젝트 관련해서 Claude(Claude Code든 claude.ai든)에게 작업을 시킬 때, 아래 프롬프트를 세션 맨 처음에 붙여넣고 시작할 것.

---

## 복사해서 쓸 프롬프트

```
우리는 2026학년도 미래인재대학 학술제 팀 COS로, NTP ICE 혼합물(농약 제형) 데이터로
화장품 제형의 피부·안 자극성(EPA Classification)을 예측하는 Multi-task PyTorch 모델을
개발 중이야.

작업 시작 전에 반드시 아래 3개 파일을 먼저 읽고 맥락을 파악해줘:
1. Desktop\학술제\세팅\진행상황.md — 전체 진행 이력, 지금까지 결정된 것과 미결 사항
2. Desktop\학술제\세팅\멀티태스크_모델링_정리.md — 모델 아키텍처 설계 배경(왜 이 구조로 정했는지)
3. Desktop\학술제\.claude\skills\irritation-pipeline\SKILL.md — 데이터 스키마·라벨링 규칙·작업 규칙 요약
   (이 폴더 안에서 Claude Code를 실행 중이면 자동으로 로드되어 있을 거야, 안 보이면 직접 읽어줘)

현재 상태 요약:
- skin: CASRN one-hot 방식 폐기 확정(2026-10-06) → RDKit SMILES descriptor 방식(염 제거 적용)으로 재구성 완료, **`skin irritation\skin_vectorized.csv`가 현재 쓰는 파일**(재구성 과정은 `skin_feature_rebuild.ipynb` 참조). 이전 CASRN one-hot 버전 파일은 완전히 삭제됨 — 과거 기록에 "v2"나 "DEPRECATED"란 이름이 보이면 지금 이 파일을 가리키는 것.
- eye: 아직 CASRN one-hot(`eye_vectorized.csv`) 그대로 — skin과 동일한 방식으로 재구성 예정
- Multi-task 모델 코드 스켈레톤: modeling\src\ (datasets.py/model.py/splits.py/train.py/evaluate.py) — skin 쪽은 v2 descriptor 스키마에 맞게 업데이트 완료, 아직 실행은 안 해봄(PyTorch 미설치 환경에서 로직만 검증)

[오늘 하고 싶은 작업]
→ 여기에 실제로 시킬 작업을 적을 것. 예시:
  - "modeling 폴더 학습 코드 pip install부터 실행까지 해주고 결과 F1/ROC-AUC 보여줘"
  - "eye도 skin처럼 SMILES 기반 descriptor로 재벡터화해줘"
  - "eye_vectorized.csv의 Mixture 608 vs 1주차 확정 579 수치 불일치, 원본 eye_filtered.csv까지 대조해서 원인 밝혀줘"
  - "화장품 전성분표 공개 데이터 찾아서 수집 시작해줘"

작업 규칙(지켜줘):
- 파일 수치나 스키마는 추측하지 말고 반드시 직접 열어서(pandas 등으로) 확인한 뒤 말할 것
- 데이터/코드에서 이상하거나 미결인 점을 발견하면 조용히 넘어가지 말고 세팅\진행상황.md에 기록하고 알려줄 것
- 큰 설계 변경, 기존 결정 뒤집기, 삭제성 작업은 먼저 설명하고 승인받은 뒤 진행할 것
- 작업 끝나면 세팅\진행상황.md를 그 결과로 업데이트할 것
```

---

## 왜 이 프롬프트 구조인가

- **3개 파일을 먼저 읽게 하는 이유**: 팀원 4명이 각자 다른 시점/다른 세션에서 Claude를 쓰다 보니, 매번 새 세션은 과거 맥락을 모름. 이력(진행상황.md)·설계근거(모델링 정리)·당장 지켜야 할 규칙(SKILL.md)을 구조적으로 분리해서, 매번 같은 3곳만 보면 누가 언제 질문해도 동일한 답이 나오게 함.
- **"오늘 하고 싶은 작업" 자리를 비워둔 이유**: 날마다 작업 내용이 다르므로 고정 태스크를 박아두지 않음 — 대신 예시로 지금 당장 유효한 작업 후보(학습 실행/스코프 불일치 조사/EDA/화장품 데이터 수집)를 보여줘서 팀원이 뭘 적어야 할지 감 잡게 함.
- **작업 규칙 4줄**: 이번 세션에서 실제로 효과를 본 것들(파일 직접 대조 확인, 미결사항 기록, 승인 없는 설계변경 금지, 작업 후 문서 갱신)을 그대로 팀 전체 규칙으로 승격.
