# 선행연구 자료

## ChemBERTa-2

- 논문: Walid Ahmad, Elana Simon, Seyone Chithrananda, Gabriel Grand, Bharath Ramsundar. *ChemBERTa-2: Towards Chemical Foundation Models*. ELLIS Machine Learning for Molecule Discovery Workshop, 2021; arXiv:2209.01712, v1 (2022).
- 원문: https://arxiv.org/abs/2209.01712
- 라이선스: arXiv 기록에 표시된 CC BY 4.0. 재배포 시 저자·출처를 표시합니다.
- 파일: [ChemBERTa-2 관련 논문.pdf](ChemBERTa-2%20관련%20논문.pdf)

논문은 SMILES 기반 분자 표현 사전학습을 다룹니다. 주요 방법은 SMILES 토큰을 가렸다 복원하는 masked language modeling(MLM)과, SMILES로 계산한 분자 특성을 예측하는 multi-task regression(MTR)입니다. PubChem의 7,700만 분자를 이용해 사전학습 규모와 방법을 비교하고, MoleculeNet 과제에서 후속 미세조정 성능을 평가합니다. 과제에 따라 사전학습 개선의 전이 정도가 달랐다고 보고합니다.

이 자료는 화장품 원료나 화장품 자극성에 대한 직접 검증 결과를 의미하지 않습니다. 팀의 화장품 적용은 별도 데이터 감사와 성능 검증이 필요합니다.
