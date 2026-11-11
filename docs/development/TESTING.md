# 현재 코드 검사 실행

저장소 루트에서 해당 마일스톤에 영향을 받는 묶음을 한 번 실행한다.
실패 원인을 수정한 뒤에는 영향받은 검사만 다시 실행한다. 실행 중인
소스 export 시험은 파일 변경을 감지하므로, 시험 도중 소스를 편집하지 않는다.

## 공개 clone에서 가능한 검사

```bash
./Data/generation/drone-data setup --tests
PYTHONPATH=Data/generation:ai:ai/src:simulation/sim/src \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
Data/generation/.venv/bin/python -B -m pytest -q \
  -c Data/generation/pyproject.toml --strict-markers \
  -m "not local_evidence" Data/generation/tests \
  ai/tests/test_pretrain_portability.py ai/tests/test_pretrain_source_lock.py
```

`local_evidence`는 Git에 없는 승인·회차 기록, 별도 AI 환경 또는 설치된
native simulator를 사용하는 검사다. 기본 로컬 전체 시험은 이 항목을
제외하지 않는다. 공개 CI가 통과해도 실기체·GPU 학습 성능의 통과는 아니다.

프런트엔드는 `UI/frontend`에서 `npm ci --ignore-scripts` 후
`npm run format:check`, `npm test`, `npm run build`를 실행한다.
현재 CI의 Python 검사 도구는 Ruff 0.11.13이며 `ruff check ai
Data/generation UI/backend`와 `ruff format --check ai Data/generation
UI/backend`를 사용한다.

## 기존 개발 환경의 전체 회귀 검사

AI 환경은 `ai/.venv`, 수집 환경은 `Data/generation/.venv`다. 같은 환경으로
합치거나 전역 Python을 변경하지 않는다. 아래 명령은 보존된 로컬 자료가
있는 개발 환경을 전제로 한다.

```bash
PYTHONPATH=ai:ai/src:Data/generation:simulation/sim/src \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
ai/.venv/bin/python -B -m pytest -q ai/tests

PYTHONPATH=Data/generation:ai:ai/src:simulation/sim/src \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
Data/generation/.venv/bin/python -B -m pytest -q \
  -c Data/generation/pyproject.toml Data/generation/tests
```

일부 시험은 임시 로컬 UDP/UNIX 소켓, 자식 프로세스, `/proc`의 소유권
확인이 필요하다. 샌드박스에서 이를 막으면 `PermissionError`, 빈 프로세스
정보 또는 초기화 정지가 발생할 수 있다. 소프트웨어 실패와 구분하고,
필요한 로컬 권한이 있는 환경에서 해당 시험만 다시 확인한다.

UI 백엔드는 자체 환경과 **실제 AI 개발 루트**를 함께 지정한다. 예를 들어
현재 중첩 업로드 체크아웃을 사용하는 개발 디렉터리에서는 다음과 같다.

```bash
ARION_AI_ROOT="$PWD" \
PYTHONPATH=irion-upload/UI/backend:ai:ai/src:Data/generation:simulation/sim/src \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
irion-upload/UI/.venv/bin/python -B -m pytest -q irion-upload/UI/backend/tests
```

일반 clone에서는 UI 경로가 `UI/`다. `ARION_AI_ROOT`는 모델 산출물과
`ai/.venv`가 실제로 있는 루트여야 한다. 중첩 업로드 사본에 코드만 있으면
가장 가까운 코드 경로를 찾더라도 AI 환경을 사용할 수 없다.

## 실패 증거와 보존 경계

로그·JUnit XML·사용한 환경·최초 실패를 `ai/reports/`의 회차별 디렉터리에
남긴다. 전체 실행과 수정 후 부분 재검사의 중복 제거 합계는 한 번의 전체
통과와 구분해 보고한다. 신규 GPU 학습이나 실제 비행을 시험 명령에 넣지 않는다.

M13에서 확인한 봉인 시험 22개 실패와 모델 소스·전처리 보존 조건은
[검수 결과](CODE_REVIEW_RESULTS.md)에 있다. M14에서는 봉인된 원문을
유지하면서 시험 의존성을 해당 버전에 맞춰 연결했다. verifier 교체나
skip/xfail·봉인 재작성은 하지 않았다. [원인과 수정](CODE_REVIEW_M14.md).

## M13에서 사용한 시뮬레이션 검사 환경

현재 개발 PC의 시뮬레이션 환경은 Python 3.12이며 pytest가 없다. 의존성을
바꾸지 않고 Data 환경에 이미 설치된 시험 도구 경로를 **뒤에 추가**했다.
시뮬레이션의 NumPy/Gazebo 등 기존 의존성 선택은 앞 순서를 유지한다.
이 방법은 같은 Python 3.12를 쓰는 현재 개발 PC용이다.

```bash
cd /home/sa/CapstoneV3
PYTHONPATH=simulation/sim/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
simulation/.venv/bin/python -B -c \
'import sys, runpy; sys.path.append("/home/sa/CapstoneV3/Data/generation/.venv/lib/python3.12/site-packages"); runpy.run_module("pytest", run_name="__main__")' \
  -q simulation/sim/tests simulation/audit/native_consumption_v2
```

M13 결과는 233 통과·22 실패였다. M14에서 같은 명령으로 **261개 통과**를
확인했다(기존 255개와 새 경계 검사 6개). `simulation/conftest.py`가 원래
시험의 API/역사 정책을 명시적으로 연결하므로 표준 pytest 실행을 사용한다.
AI Python 3.10은 이 시험의
`datetime.UTC` 요구와 맞지 않으므로 대신 사용하지 않는다. 최신 소스의
전체 검사 결과와 실행 명령은 [M13 최종 검증](CODE_REVIEW_M13.md)에 있다.

전압 오차 재현과 입력군별 민감도 진단은 [M15](CODE_REVIEW_M15.md)에 있다.
진단은 고정 모델과 준비된 입력을 읽으며 출력은 반드시 새 파일에 저장한다.
