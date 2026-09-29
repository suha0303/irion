# 시뮬레이션 문서

[현재 시뮬레이터 구조](../../simulation/README.md), [v4.2 시뮬레이션 명세](../specifications/CODEX_REBUILD_PACKAGE_V4_2/01_SIMULATION_CODEX_SPEC.md), [공통 데이터 계약](../specifications/CODEX_REBUILD_PACKAGE_V4_2/00_SHARED_DATA_CONTRACT.md)을 따른다.

부품·비행 설정은 `../../simulation/profiles/`, 물리 소스와 관련 시험은 `../../simulation/sim/`, 동결 생산자와 검증 증거는 `../../simulation/artifacts/`에 있다. 배터리 공개 참조자료는 `../../simulation/data/public/`이다. 수집 CLI는 [Data](../../Data/README.md)로 분리했다.

별도 기체 프로파일 확장은 [다른 multirotor 프로파일](VEHICLE_PROFILES.md)을 따른다. 지원 범위는 고정 수직축4..8모터이며 기존 X500 경로와 AI 학습은 유지한다. 예제 quad/hex는 공학 fixture로, 실제 비행이나 해당 기체의 AI 성능을 검증한 결과가 아니다.

현재 수집 설정은 `collection`, 프로파일 폴더는 `simulation/profiles/collection/`이다. 구현은 `collection_config.py`와 `run_assessment.py`, 관련 도구는 `generate_collection_header.py`, `run_collection.py`, `assess_run.py`를 사용한다. 과거 회차를 뜻하는 이름은 새 실행 경로로 사용하지 않으며 기존 데이터·봉인 패키지·시험 증거 안에서는 원래 의미를 보존한다.
