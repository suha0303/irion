# 수집·정제 설치와 로컬 운영

소스를 다른 폴더나 다른 컴퓨터로 옮긴 뒤 환경을 새로 설치해 X500 수집·정제를 실행한다. 사용자 이름·기존 외장하드·이 컴퓨터의 Python 설치 경로에 의존하지 않는다. 이 절차는 모델 학습을 시작하지 않는다.

## 지원 환경과 준비물

- 지원 기준: Ubuntu 24.04 x86_64, Python 3.12와 venv/pip, Git, user systemd와 cgroup v2.
- 비행 수집: Gazebo Harmonic의 개발 패키지, CMake·Ninja·GCC/G++, pkg-config와 PX4 빌드 의존성. 시스템 패키지는 사용자가 설치한다. 설치기는 sudo나 시스템 Python 변경을 실행하지 않는다.
- 최초 설치: Python 패키지와 고정된 PX4·Gazebo 소스를 받을 네트워크. 이후 수집은 LLM/API 연결이 필요하지 않다.
- 저장 공간: 선택한 최종 데이터 저장소와 `simulation/runs/` 임시 공간. 최종 압축 용량만으로 수집·정제 중 필요한 공간을 산정하지 않는다.

Windows·macOS 네이티브 실행이나 다른 Ubuntu 버전은 이 절차의 지원 확인 범위에 포함하지 않는다. 새 컴퓨터에서 전체 다운로드·빌드·실비행을 완료한 검증과 현재 컴퓨터의 소스·환경 점검은 구분한다.

## 설치와 진단

소스를 클론하거나 `export`한 소스를 압축 해제한 프로젝트 루트에서 실행한다. 이 프로젝트의 virtualenv와 빌드 디렉터리는 복사하지 않는다.

```bash
# 정제 전용 Python 환경 설치. GPU·PyTorch는 설치하지 않는다.
./Data/generation/drone-data setup

# 새 컴퓨터에서 비행 수집도 사용할 때: native 환경과 생산자 설치·빌드.
./Data/generation/drone-data setup --simulation

# 설치 상태·의존성과 현재 생산자 확인. 비행은 시작하지 않는다.
./Data/generation/drone-data doctor
```

네이티브 시스템 의존성만 먼저 확인하려면 `./simulation/sim/tools/bootstrap.sh --check`를 실행한다. 누락된 패키지와 사용자가 실행할 설치 명령을 확인할 수 있다. 병렬 빌드 수는 이 스크립트의 `--jobs N`으로 지정한다. `--adopt-existing`은 현재 장비의 검증된 기존 자산을 안정 경로에 인수하는 유지보수 옵션이며 새 컴퓨터의 일반 설치에는 필요하지 않다.

Python 환경은 `Data/generation/.venv/`와 `simulation/.venv/`에 분리한다. `ai/.venv/`와 원격 서버 환경을 수정하지 않는다. 생산자 소스 manifest는 `simulation/config/producer-sources.json`을 기준으로 하며 설치된 네이티브 빌드·runtime은 해당 컴퓨터에서 확인한다. 다른 장비의 실행 파일 해시를 그대로 요구하지 않고 소스·설정과 설치 결과의 식별을 나눠 기록한다. 이미 시작된 작업의 설정·입력·소스는 변경하지 않는다.

진단이 실패하면 필요한 의존성·경로·권한을 먼저 해결한다. setup이나 doctor가 성공했다는 사실만으로 비행 품질 검증이나 본수집 완료를 뜻하지 않는다. 선택 게이트·유한 시험은 메뉴에서 사용자가 실행한다.

## 수집과 정제

```bash
./Data/generation/drone-data menu
# 설치 후 같은 메뉴를 직접 열어도 된다.
./Data/generation/drone-pretrain
```

메뉴는 새 수집, 유한 시험, 기존 작업 화면, 작업 종료를 제공한다. 기본값은 300run × 선택 배율, 비행16병렬, 검증·압축/추출/전처리/검산 각각4병렬, 2배속이다. 비행 병렬 수는 PX4 포트·인스턴스 계약에 따라1..16, CPU 단계는 양의 정수를 받는다. 메모리 추정치·고정 RSS 상한은 기본 해제하되 최소 가용 RAM1GB와 저장소·소유권·품질 검사는 유지한다. swap을 허용한다. 병렬 수는 다른 컴퓨터의 실제 메모리와 CPU에 맞춰 선택한다.

설치 후 짧은 실제 동작 시험은 프로젝트 루트에서 다음 공개 명령으로 실행한다. `smoke`는 아직 사용하지 않은 작업 이름이어야 한다.

```bash
./Data/generation/drone-pretrain init --name smoke --mode test --levels 1 \
  --flight-workers 1 --export-workers 1 --workers 1 --io-workers 1 --audit-workers 1
./Data/generation/drone-pretrain start Data/operations/pretrain/tasks/smoke
./Data/generation/drone-pretrain watch Data/operations/pretrain/tasks/smoke
```

이 시험은 120초 비행 3개로 train/validation/calibration의 추출·전처리·공통 scaler·독립 검산을 확인한 뒤 `TEST_COMPLETE_STOPPED`로 끝난다. 본수집·학습은 시작하지 않는다. `start`가 메모리 정책과 서비스 환경을 생성하므로 내부 명령 `_worker`를 직접 호출하지 않는다. 화면을 닫아도 작업은 독립 실행된다.

기능과 경계는 다음과 같다.

- 게이트는 선택 사항이다. 정상 조건 범위·기동·경로·바람·탑재량·잔량·온도와 이상·정상 대조의 역할을 유지한다.
- seed를 생략하면 작업마다 OS 난수에서 새 seed를 만들며 직접 지정하면 재현 가능한 구성을 고정한다. 배율별 수집 seed는 달라지고 선택한 배율 전체를 하나의 데이터셋으로 집계한다.
- 일반 정상은15분, 저SOC는3분을 기본으로 한다. 단순 시간 배율로 물리 배터리 용량을 늘리지 않는다.
- 수집과 묶음별 추출·정제 후 전체 집계, 독립 품질 검산, 최종 압축본 검증을 수행하고 멈춘다. 정상 train에만 맞춘 공통 scaler를 validation/calibration/개발평가에도 적용한다. 정답은 모델 입력과 분리한다.
- 개별 안전 실패는 최대2회 재시도해 총3회까지만 실행한다. 실패 기록을 남기고 다음 run으로 넘어가되 자원·공유 안전 실패는 신규 시작을 중단하는 기존 정책을 따른다. 불완전 구성을 정상 학습 자료로 인수하지 않는다.
- 성공 산출물의 복원 가능성과 해시 검증 후 해당 작업 소유의 성공 원시·중간 파일만 정리한다. 실패 증거, 다른 작업 자료와 소유권이 불명확한 소켓은 보존한다.

최종 데이터 저장소의 기본값은 프로젝트 `Data/`다. 메뉴 또는 `init --storage-root /선택한/경로`로 다른 저장소를 지정할 수 있다. 작업 생성 시 저장 경로·mount·device를 고정하며 선택한 장치가 사라지면 다른 디스크로 자동 전환하지 않는다.

새 수집 기록은 프로토콜 `DRONE_COLLECTION`, 설정 `collection`, 데이터 풀 `collection/x500`을 사용한다. 기본 기체는 X500이며 프로파일은 `simulation/profiles/collection/`이다. 예전 회차 이름은 현재 수집 옵션이 아니다. 현재 봉인 패키지의 실제 모듈 폴더는 `preprocessing_core`·`preprocessing_tools`다. 이름 변경 전 원본은 `docs/archive/maintenance/generic-package-20261018-01/`에 보존하고 변경 계보와 새 파일 해시를 기록했다. 전처리 계산·설정과 기존 데이터의 배터리/schema 식별자는 유지한다. 실행기의 현재 전처리 연결은 `pretrain/resources/admission/`에서 제공한다.

## 화면·로그·중지

작업 기록은 `Data/operations/pretrain/tasks/<task>/`에 두고 `operator/state.json`, `operator/run.log`와 각 단계의 상세 산출물을 기록한다. 최종 데이터는 선택한 저장소의 `processed/pretrain/<task>/prepared-portable/`에 생성된다. 설치·정제 소스와 개인 데이터는 별개다.

```bash
# 현재 작업을 한 번 확인한다.
./Data/generation/drone-pretrain status

# 사용자가 보는 자동 갱신 화면. LLM 호출 없음.
./Data/generation/drone-pretrain watch
```

화면에서 q/Ctrl+C는 화면만 닫는다. 작업은 user systemd 서비스에서 계속 실행한다. 로그아웃하면 user manager가 종료될 수 있으며 PC 종료 시 작업은 유지되지 않는다. 설치기는 linger 설정을 변경하거나 작업을 자동 재시작하지 않는다. 작업 중지는 메뉴의 작업 종료를 사용하며 기록된 소유 작업만 대상으로 한다. 임의 `pkill`이나 전체 사용자 작업 종료로 정리하지 않는다.

단계100%는 전체 완료와 다르다. 최종 상태·검산 결과·서비스 종료를 확인하기 전에는 학습 준비 완료로 간주하지 않는다. 수집·정제 뒤 학습, 데이터 증량, 새 seed·모델 추가나 다음 회차를 예약하지 않는다.

## 수집 완료 후 캡슐에서 통합 복구

모든 run의 추출·전처리·개별 검산·캡슐 저장은 끝났으나 전체 통합 전에 중단된 작업은, 원래 작업을 재시작하지 않고 새 이름으로 복구한다. 다음 명령은 기존 캡슐을 검증·복원하고 전체 train 기준 공통 scaler와 최종 품질 검산·압축 보관까지 수행한 뒤 멈춘다. 비행이나 AI 학습은 실행하지 않는다.

```bash
./Data/generation/drone-pretrain recover-capsules \
  Data/operations/pretrain/tasks/<중단된-작업> --name <새-복구-작업>
./Data/generation/drone-pretrain watch Data/operations/pretrain/tasks/<새-복구-작업>
```

원본 작업이 실제로 종료됐는지, 계획된 모든 run의 영수증·검산·동결 기록이 일치하는지 확인한다. 일부 수집이나 캡슐이 누락된 경우에는 진행하지 않는다. 원본 캡슐·실패 증거는 보존하며, 정리 대상은 복구 작업이 새로 만든 임시 파일뿐이다. 복구 중 오류가 나면 기록하고 멈추며 자동으로 다시 제출하지 않는다. 네이티브 충돌 시에도 Python 스택이 서비스 로그에 남도록 faulthandler를 활성화한다.

캡슐 복구는 `PYTHONMALLOC=debug` 메모리 진단 설정을 작업에 고정한다. 이 설정은 메모리 손상 탐지를 돕지만 운영체제·인터프리터 오류를 해결했다는 보장은 아니다. 최종 파일은 기존 독립 검산과 해시 검증을 모두 통과해야 인수한다.

## 소스 배포와 기존 자료

```bash
# 현재 체크아웃 밖의 새 경로를 사용한다. 기존 파일은 덮어쓰지 않는다.
./Data/generation/drone-data export --output /tmp/Capstone-source.tar.gz
```

배포 대상은 실행 소스·설정·계약·필수 fixture·동결 전처리 패키지다. Python 환경, PX4 다운로드·네이티브 빌드, 개인 데이터, 학습 가중치, 실행 상태, 로컬 보고서·과거 보존 폴더는 제외한다. 소스 묶음은 `SOURCES.json`으로 포함 파일의 해시를 기록한다. `.gitignore`도 사용자 산출물과 소스를 구분한다. Git 저장소 초기화·원격 push는 자동 수행하지 않는다.

새 컴퓨터에 기존 완성 데이터셋을 인수할 때는 최종 압축본·평가자 정답·receipt를 함께 복사하고 원래 해시와 split·scaler를 유지한다. 실행 중 task나 PID·소켓·mount 정보를 다른 장비로 재개하지 않는다. 기존 데이터와 증거의 옛 schema·revision·외부 패키지 버전은 당시 계보로 보존하고 현재 진입 명령·운영 경로에서는 버전별 분기를 사용하지 않는다.

AI 학습은 별도다. 서버에는 학습 소스 snapshot·환경 profile·데이터 receipt를 함께 올리고 승인된 Slurm 자원 안에서 실행하는 기존 흐름을 유지한다. 이 설치 절차는 서버나 모델 구조를 변경하지 않는다.
