# 드론 시뮬레이션·데이터와 X500 이상탐지

PX4·Gazebo에서 X500 비행로그를 수집하고, 정상 train 기준 공통 scaler를 사용하는 학습용 데이터셋으로 정제한다. 수집부터 독립 품질 검산·압축까지 로컬 프로그램이 실행하며 LLM 연결은 필요하지 않다.

| 폴더 | 목적 |
|---|---|
| [simulation/](simulation/README.md) | 시뮬레이션·물리·프로필·계약·네이티브 빌드 |
| [Data/](Data/README.md) | 수집·정제 프로그램과 사용자가 선택한 데이터 저장소 |
| [ai/](ai/README.md) | 모델·학습·보정·추론·평가·원격 GPU 실행 |
| [docs/](docs/README.md) | 명세·운영·개발 안내와 과거 보존 자료 |

## 다른 컴퓨터에서 시작

지원 기준은 **Ubuntu 24.04 x86_64와 Python 3.12**다. 비행 수집에는 Gazebo Harmonic·PX4 빌드 의존성과 user systemd가 필요하다. 자세한 사전 조건과 단계별 실행은 [설치·이동 안내](docs/data/PORTABLE_SETUP.md)를 따른다.

```bash
# 프로젝트를 클론한 폴더에서 실행한다. 설치·점검은 수집을 시작하지 않는다.
./Data/generation/drone-data setup --simulation
./Data/generation/drone-data doctor

# 사용자가 수집·시험·기존 작업 화면·작업 종료를 선택한다.
./Data/generation/drone-data menu
```

정제용 Python은 `Data/generation/.venv`, 시뮬레이션은 `simulation/.venv`에 설치한다. AI 학습용 `ai/.venv`와 서버 환경은 별개이며 데이터 수집을 위해 PyTorch나 GPU를 요구하지 않는다. 이미 설치한 컴퓨터에서는 `./Data/generation/drone-pretrain`으로 같은 메뉴를 열 수 있다.

기본 구성은 **300run × 선택 배율**, 비행16병렬·나머지4병렬·2배속이다. 게이트는 선택 사항이다. seed는 작업마다 새로 생성하고 직접 고정할 수도 있다. 선택한 배율 전체를 하나의 데이터셋으로 만들며 정상 train만 사용해 공통 scaler를 계산한다. 개별 run은 최대3회 시도하고 실패 증거를 보존한다. [기능·저장·중지 정책](Data/generation/pretrain/README.md)을 따른다.

현재 수집 설정의 이름은 `collection`, 실행 기록의 프로토콜은 `DRONE_COLLECTION`이다. 기체를 지정하지 않으면 X500을 사용하며 수집 프로파일은 `simulation/profiles/collection/`에 있다. 과거 회차 이름은 기존 데이터·봉인된 전처리 패키지의 계보를 읽을 때만 사용한다.

최종 저장 위치는 메뉴에서 지정하며 기본값은 프로젝트의 `Data/`다. 원시·중간 파일은 성공 산출물의 복원 가능성을 검증한 뒤 해당 작업 소유분만 정리한다. 개인 데이터셋·학습 가중치·실행 상태·로컬 설치 환경은 Git 소스에 포함하지 않는다. `export` 명령으로 다른 위치에서 설치할 소스 묶음을 만들 수 있다.

## AI 학습

서버에서 클론한 뒤 `./ai/drone-ai configure`로 서버·기존 Python·데이터·출력 경로를 입력한다. 환경 진단·짧은 GPU 학습/저장/복원 시험·진행 화면은 [서버 사용 안내](docs/ai/PORTABLE_SETUP.md), 비교 구성은 [실험 계획](docs/ai/EXPERIMENT_PLAN.md)을 따른다. 본학습과 다음 회차는 자동 시작하지 않는다. 데이터와 모델·컴퓨터별 설정은 Git 소스와 분리한다.

과거 `v1`·`v7` 등의 운영 이름은 현재 진입 명령이나 설치 경로로 사용하지 않는다. 계약의 schema 식별자·외부 패키지 버전·기존 데이터 계보와 보존 증거는 의미를 유지한다. 범주 이동용 호환 링크는 사용하지 않는다.

기체를 지정하지 않으면 X500을 사용한다. 다른 기체는 [프로파일 안내](docs/simulation/VEHICLE_PROFILES.md)에 따라 시뮬레이션에서 명시적으로 선택한다. 데이터·AI 인수는 현재 X500 전용이다. `drone-data`·`drone-pretrain`은 현재 실행 파일명이며 옛 이름의 호환 링크는 만들지 않는다.
