# 시뮬레이션

`sim/`은 Python 제어·관측·C9·물리 플러그인 소스와 시험, `profiles/`는 X500 배터리·추진·환경 설정, `contracts/`는 원시 데이터 스키마다. `config/producer-sources.json`은 배포하는 소스·설정의 잠금이고 `artifacts/runtime-manifest.json`은 이 컴퓨터에서 설치·빌드한 결과를 식별한다. `artifacts/fixtures/`와 `artifacts/sources/`는 시험 fixture와 참조자료다.

```bash
./simulation/sim/tools/bootstrap.sh --check
./simulation/capstone-sim --help
```

설치·수집의 공통 진입점은 [설치 안내](../docs/data/PORTABLE_SETUP.md)를 따른다. `.venv/`는 Python3.12 시뮬레이션 환경, `vendor/`는 설치기가 받는 고정 PX4 소스, `build/current/`는 현재 물리 플러그인 빌드, `runtime/gazebo-transport/`는 프로젝트 전용 Gazebo 전송 라이브러리다. 이 환경과 실행 파일은 Git에 넣지 않고 각 컴퓨터에서 설치한다.

`.cache/instances/`는 PX4 인스턴스 잠금, `runs/`는 시뮬레이터가 관리하는 임시 원시 run이다. 최종 데이터는 상위 `Data/` 또는 사용자가 선택한 저장소에 둔다. 과거 생산자 잠금과 증거는 로컬 보존 자료로 유지한다. 배터리·추진·물리·비행·C9·입출력 계약을 설치 편의 때문에 바꾸지 않는다. 직접 비행·수집은 사용자가 선택한 범위에서만 실행한다.

수집용 설정은 `--configuration collection`이며 입력은 `profiles/collection/`에 있다. 기체 선택을 생략하면 X500이다. 직접 `run`의 기본 reference 설정과 수집 콘솔의 장시간 수집 설정은 서로 다른 설정이므로 이름 변경만으로 배터리나 물리값을 바꾸지 않는다.

모든 새 run은 `run.json`에 실제 PX4 `instance`를 기록한다. 완료·실패한 run의 `control --operation reset` 또는 `set_scenario`는 그 번호를 유지하며 새 run을 만든다. 옛 기록에 번호가 없거나 유효하지 않으면 0번을 추측하지 않고 거절한다. 이때 기존 증거를 보존하고 사용자가 `run --instance <번호>`로 별도 실행을 선택한다.

관련 문서는 [시뮬레이션 문서](../docs/simulation/README.md), 데이터 수집은 [Data](../Data/README.md)를 따른다.

다른 기체는 [기체·추진·배터리 프로파일 안내](../docs/simulation/VEHICLE_PROFILES.md)의 `profile inspect` → `profile prepare` → 명시적 `run --vehicle-profile` 순서로 추가한다. 고정 수직축 4~8로터를 지원하며, [quad/hex 예제](profiles/examples/README.md)를 제공한다. 기존 X500 수집과 AI 계약은 유지하고 새 기체 로그는 simulator 전용으로 구분한다. 프로파일 준비는 비행·수집·학습을 시작하지 않는다.
