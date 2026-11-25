# ARION V0.2 · 드론 비행 분석

PX4·ArduPilot 비행 로그와 MAVLink 수신 데이터를 분석하는 오프라인 데스크톱 앱입니다. 비행 중 예측 4종과 비행 후 판정 4종을 기본 제공하며, 설치 후 CPU에서 실행합니다.

## 설치

[GitHub Releases에서 ARION V0.2.0-rc.3 받기](https://github.com/suha0303/irion/releases/tag/v0.2.0-rc.3)

| 운영체제 | 설치 방법 | 검증 환경 |
| --- | --- | --- |
| Ubuntu x64 | `.deb`와 `.sha256`을 받은 뒤 아래 명령으로 설치합니다. 앱 메뉴에서 **ARION**을 엽니다. | Ubuntu 24.04.5 노트북 |
| Windows x64 | `.exe`를 실행한 뒤 시작 메뉴에서 **ARION**을 엽니다. 설치본에는 코드 서명이 없습니다. | Windows Server 2025 GitHub Actions |
| macOS | Apple Silicon은 `macos-arm64.dmg`, Intel은 `macos-x64.dmg`로 끝나는 파일을 받습니다. DMG를 열고 ARION을 Applications로 옮깁니다. | macOS 15 GitHub Actions |

```bash
sha256sum -c ARION-0.2.0-rc.3-ubuntu-amd64.deb.sha256
sudo apt install ./ARION-0.2.0-rc.3-ubuntu-amd64.deb
```

Windows 설치본은 `Get-FileHash .\ARION-0.2.0-rc.3-windows-x64.exe -Algorithm SHA256`으로 첨부된 `.sha256`과 대조할 수 있습니다. 자세한 절차는 [설치 안내](docs/ai/DESKTOP_RELEASE.md)와 [검증 체크리스트](docs/ai/ARION_RELEASE_CHECKLIST.md)에 있습니다.

Mac DMG는 Apple 공증 전 테스트용입니다. 같은 폴더에 DMG와 해당 `.sha256`을 두고 `shasum -a 256 -c ARION-0.2.0-rc.3-macos-arm64.dmg.sha256`을 실행합니다. Intel은 파일명에서 `arm64`를 `x64`로 바꿉니다. 해시가 일치하고 첫 실행이 차단되면 **시스템 설정 → 개인정보 보호 및 보안 → 그래도 열기**에서 ARION에 예외를 줄 수 있습니다. [Apple 첫 실행 안내](https://support.apple.com/ko-kr/102445)

## 사용 방법

- **비행 로그 분석**은 **로그 모니터링 → 비행 로그 가져오기**에서 PX4 `.ulg` 또는 ArduPilot `.bin`을 열어 시작합니다. 프로파일과 사용 구간을 저장해 분석하면 예측 그래프와 비행 후 판정을 볼 수 있습니다.
- **실시간 비행 모니터링**에서 MAVLink UDP를 연결합니다. 수신 중에는 10초 버퍼를 거친 비행 중 모델 결과를 확인하고, **비행 종료 · 결과 분석**을 누르면 저장된 원문으로 전체 비행을 분석합니다.
- **기체별 파인튜닝**은 학습·보정·평가 로그를 따로 지정합니다. 출력층을 파인튜닝하고 판정 기준을 보정한 후보를 함께 저장합니다. 검증 대기 후보는 자동 적용하지 않습니다.

## 검증 범위

Ubuntu에서 외부 네트워크 없이 로그 분석, PX4 SITL 이륙·착륙과 MAVLink 기록 마감, 정상 CUAVv5 로그로 파인튜닝 후보 검증을 확인했습니다. Windows와 macOS에서는 합성 ULog 8종 분석과 루프백 MAVLink 원문 저장을 자동 검증했습니다. 정상 비행에 오경보가 남아 있으며 실기체 이상 검출 성능은 추가 검증이 필요합니다. [V0.2 평가 결과](ai/results/new19-20261121/README.md), [Ubuntu 추가 검증](docs/ai/ARION_RELEASE_CHECKLIST.md#ubuntu-노트북-검증-기록)

## 개발과 연구

이 저장소는 앱 소스와 PX4·Gazebo X500 비행 데이터 수집, 전처리, 모델 학습·평가 코드를 함께 관리합니다.

| 폴더 | 목적 |
|---|---|
| [simulation/](simulation/README.md) | 시뮬레이션·물리·프로필·계약·네이티브 빌드 |
| [Data/](Data/README.md) | 수집·정제 프로그램과 사용자가 선택한 데이터 저장소 |
| [ai/](ai/README.md) | 모델·학습·보정·추론·평가·원격 GPU 실행 |
| [UI/](UI/README.md) | 비행 모니터링 대시보드·재생·실시간 수신·모델 결과 표시 |
| [docs/](docs/README.md) | 명세·운영·개발 안내와 과거 보존 자료 |

### 수집 환경 설정

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

### AI 학습

서버에서 클론한 뒤 `./ai/drone-ai configure`로 서버·기존 Python·데이터·출력 경로를 입력한다. 환경 진단·짧은 GPU 학습/저장/복원 시험·진행 화면은 [서버 사용 안내](docs/ai/PORTABLE_SETUP.md), 비교 구성은 [실험 계획](docs/ai/EXPERIMENT_PLAN.md)을 따른다. 본학습과 다음 회차는 자동 시작하지 않는다. 데이터와 모델·컴퓨터별 설정은 Git 소스와 분리한다.

과거 `v1`·`v7` 등의 운영 이름은 현재 진입 명령이나 설치 경로로 사용하지 않는다. 계약의 schema 식별자·외부 패키지 버전·기존 데이터 계보와 보존 증거는 의미를 유지한다. 범주 이동용 호환 링크는 사용하지 않는다.

기체를 지정하지 않으면 X500을 사용한다. 다른 기체는 [프로파일 안내](docs/simulation/VEHICLE_PROFILES.md)에 따라 시뮬레이션에서 명시적으로 선택한다. 데이터·AI 인수는 현재 X500 전용이다. `drone-data`·`drone-pretrain`은 현재 실행 파일명이며 옛 이름의 호환 링크는 만들지 않는다.

### 평가 기록

- [V0.2 모델 실행과 앱 연결](ai/learning/new19/README.md)
- [기능과 실기체 검증 체크리스트](docs/ai/LOCAL_REAL_FLIGHT_CHECKLIST.md)

이전 평가 기록:

- [최종 324run 데이터 수집·정제·품질 결과](Data/results/final-test-20261103/README.md)
- [18모델 최종 평가 비교표·그래프·학습 기록](ai/results/final-evaluation-20261103/README.md)

위 이전 18모델 최종 평가는 2026-11-04 01:08 KST에 완료됐다. 각 결과 폴더에 원본 해시, 검산 범위와 해석상의 한계를 함께 기록했다.
