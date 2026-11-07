# 최신 코드 검수 범위

2026-11-15 기준 551개 자체 코드 파일, 116,466행을 목록화했다.
Python 478, TypeScript/TSX 31, C++ 31, 셸 7, 진입 스크립트 4개다.
전체 파일 경로·분류·변경 전 SHA는 로컬
`ai/reports/code-review-20261115/inventory-before.json`에 기록한다.

## 최신 코드와 보존 의존성

| 분류 | 파일 수 | 처리 |
|---|---:|---|
| 현재 수정 가능한 소스·테스트·도구 | 374 | 정적 검수·리팩토링·회귀 검사 |
| 생산자 봉인에 포함된 현재 시뮬레이션 | 127 | 전수 검수, 기존 실행 봉인의 바이트 보존 |
| 최신 16종 모델의 해시 검증 소스 | 2 | 전수 검수, 가중치와 결합한 소스 유지 |
| 독립 평가 버전 스냅샷 | 6 | 검수·호출 호환 검사, 과거 판정 구현 유지 |
| 동결 전처리·명세 검증 사본 | 42 | 현재 의존 관계·해시 검사, 스타일 수정 제외 |

시뮬레이션 소스와 설치된 바이너리는 `producer-sources.json` 및
`runtime-manifest.json`으로 연결된다. 의미가 같은 포맷 변경도 이 봉인을
깨뜨리므로, 스타일 때문에 기존 통과 증거를 다시 쓰지 않는다. 발견한 문제는
명시하고, 현재 입력 계약·물리·모델 계산을 바꾸는 변경과 분리한다.

GitHub 업로드용 `irion-upload/{ai,Data,simulation,docs}`는 개발 원본의
중복 사본이다. 검수는 개발 원본에서 한 번 수행하고 검증 후 동기화한다.
UI 원본은 `irion-upload/UI`이며 GitHub에서는 `UI`다.

외부 PX4·설치된 Python/Node 패키지, 빌드 출력, 원시 로그, 학습 산출물,
이전 릴리스의 내장 코드는 최신 자체 코드 수에 포함하지 않는다.

## 과거 자료 정리

- 2026-10-31 업로드 사본 `upload-branches/`와 과거 7일 커밋 계획을
  `docs/archive/development/20261031/`로 이동했다. 삭제하지 않았다.
- 기존 `docs/archive/`, `ai/reports/`, `ai/artifacts/`, `ai/models/`,
  `ai/releases/`, `Data/processed/`, `simulation/runs/`의 과거 증거는
  원래 경로를 보존한다. 내부 경로·해시 참조가 있어 일괄 이동하지 않는다.
- 최신 모델은 `training19-final-20261108`의 16종이다. Git에 있는
  `ai/results/final-evaluation-20261103`의 18종 결과는 역사 자료다.
- 이름에 `legacy`나 이전 회차가 있더라도 현재 CLI·체크포인트·복구 코드에서
  참조하는 코드는 삭제하지 않는다. 현재 동작을 지원하는 호환 코드로 검수한다.

## 스타일 적용

[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)의
4칸 들여쓰기, 80열 목표, 모듈 import, 의미 있는 이름, 명확한 예외·자원
수명, 필요한 공개 인터페이스의 Args/Returns/Raises 설명을 기준으로 삼는다.
Ruff는 포맷·구문·import 검사, Pylint는 추가 오류·가독성 진단에 사용한다.
자동 포맷 통과만으로 설계 검수나 전체 Google 규칙 준수를 주장하지 않는다.

기존 공개 API의 직접 import/re-export, pytest fixture import, NumPy/PyTorch의
관례적 수식 이름, CLI 시작 시 경로 설정, 늦은 선택 의존성 import는 각 사용
맥락을 검토한다. 기존 데이터 키·파일명·모델 state-dict 키는 스타일 때문에
바꾸지 않는다. 동결 파일의 진단과 수정 가능한 소스의 미해결 진단을 구분한다.

Pylint는 `ai/.local/code-review-tools`에 별도 설치한다. 기존 실행 환경과
서버 환경은 변경하지 않는다. 전체 기능 검사는 큰 마일스톤 경계에서 실행하고,
실패 로그와 처음 사용한 잘못된 검사 환경도 그대로 기록한다.

추가 보존 의존성: `Data/generation/tools/collection_support.py`는
`collect_long_runs.py::HELPER_SHA`가 고정 검증하므로 스타일 변경에서 제외한다.
분류 목록의 current 1개에 대한 추가 예외이며 기준선 목록은 사후 수정하지 않는다.
