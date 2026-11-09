# 외부학습 최소 배포 자료

`Eye_STopTox_20261108/`에는 필요한 파일 **34개**만 포함했습니다.
현재 결과는 상위 [종합 노트북](../eye_all_experiments_v2.ipynb) 6~7부에 통합되어 있습니다.

- 보고서 입력: OOF·seed별 지표·paired Δ·그래프, 화장품 관측 감사표와 사례 집계, 규약·검증 JSON.
- 실제 학습 기록: RF 보조모델 `stoptx_teacher.joblib`, 원 Table S4 관측 CSV, 선별 학습 CSV,
  행별 제외 감사표, 성분 점수·보조 특성, 고정 환경, 핵심 학습 코드 4개.
- 완료 평가 기록: 90 outer·1,800후보 JSON을 담은 `검증체크포인트.zip`.
- 현재 파일 무결성: `package_checksums.json`은 이 배포에서 선택한 나머지 33개 파일의 SHA256입니다.

원문 PDF·다운로드 로그·중복 보고서/노트북·이동/백업 기록은 로컬에 보존하고 제외했습니다.
선별한 원본 33개 파일은 로컬과 바이트가 같으며 checksum 목록만 최소 배포에 맞춰 생성했습니다.
전체 원 패키지 checksum과 제외 내역은 상위 `publication/refresh_provenance.json`에 기록했습니다.

보고서 재검증은 상위 폴더에서 `python verify_results.py`로 실행합니다.
원본 학습 코드 `common.py`, `prepare.py`, `run_transfer.py`, `verify_transfer.py`는 방법과
고정된 계산 절차를 검토하기 위한 snapshot입니다. 원 PC의 전체 ICE·성분 출처·정부 원료 목록·
원래 코드와 경로 의존이 남아 있으므로 이 최소 패키지만으로 전체 학습을 다시 실행할 수는 없습니다.
학습을 재현할 때는 로컬 원 환경과 전체 원 패키지를 사용해야 합니다.
포함된 보조모델은 화장품 최종 배포 모델이 아닙니다.

과거 `verification.json`의 원본 노트북 보존 판정은 종합본 갱신 이전 시점의 기록입니다.
STopTox GHS 관측으로 실제 학습했지만 독립 화장품 EPA 정량 검증은 적격 0행으로 미수행입니다.
