# 데이터 생성·전처리 문서

[다른 컴퓨터 설치·운영](PORTABLE_SETUP.md), [데이터 폴더](../../Data/README.md), [수집 구성과 정책](../../Data/generation/pretrain/README.md), [AI 전처리 명세](../specifications/CODEX_REBUILD_PACKAGE_V4_2/02_AI_PREPROCESSING_CODEX_SPEC.md)를 따른다.

기본300run 구성·수집16병렬·나머지4병렬·2배속·작업별 난수 seed·선택 게이트를 유지한다. 선택한 배율 전체에 정상 train 기준 공통 scaler를 사용한다. 개별 실패는 총3회까지 시도하며 불완전 데이터셋을 학습용으로 인수하지 않는다. 최종 파일은 검증된 압축본이고 원시/중간 자료 정리는 소유권·검증 정책에 따른다. locked final test는 별도 승인 전까지 성능 확인에 쓰지 않는다.

수집·정제는 `Data/generation/.venv`의 CPU 환경, 실행 기록은 `Data/operations/pretrain/`을 사용한다. 사용자 데이터·환경·소켓·작업 상태는 Git에 올리지 않으며 새 컴퓨터에서는 설치 후 새 작업을 만든다. AI 학습 환경과 원격 실행 제한은 그대로 유지한다.

새 수집은 `DRONE_COLLECTION` 프로토콜, `collection` 시뮬레이션 설정과 `collection/x500` 데이터 풀을 기록한다. 기본 기체는 X500이며 다른 기체의 AI 인수를 자동 허용하지 않는다. 봉인된 전처리 구현의 과거 모듈·배터리 이름은 명시적 호환 경계에서만 사용하며 기존 데이터의 식별자·계보·해시는 보존한다.
