# X500 AI 프로젝트 작업 지침

기존 `docs/specifications/CODEX_REBUILD_PACKAGE_V4_2/AGENTS.md`와 공통 계약/AI 명세의 지침을 유지한다.
로컬 AI 환경은 `ai/.venv`, 코드/설정/시험/산출물은 `ai/`에 둔다.
시뮬레이션 코드·원시 run·C9 및 데이터 입출력 계약을 GPU 준비 때문에 변경하지 않는다.
기존 평가/model/scaler/threshold/GPU 인수인계 패키지와 통과·실패 증거를 보존한다.

## 원격 GPU 자원 제한 (사용자 지정, 2026-10-14)

사용자가 이 작업 중 계산 노드를 DIS05에서 **DIS01**로 변경 승인했다. 최신 DIS01 고정 조건을 적용한다.

- 승인된 접속: `ssh coss43@155.230.81.91 -p 10000`.
- 비밀번호는 사용자가 터미널에서 직접 입력하거나 이미 승인된 SSH 키 인증을 사용한다.
  비밀번호·개인키를 요청/수집하거나 코드·설정·문서·로그·Git에 저장하지 않는다.
- SSH host-key 검사와 실행 환경의 권한 제한을 우회하지 않는다.
- Slurm **p02 / DIS01 / 단일 노드 / GPU 1개**만 사용한다. 다른 파티션/노드로 우회하지 않는다.
  자동 명령에는 `--partition=p02 --nodelist=DIS01 --nodes=1 --ntasks=1 --gres=gpu:1`을 명시한다.
- GPU 작업은 이 프로젝트 전체에서 한 번에 하나만 실행한다. 여러 GPU1개 작업도 동시 제출 금지.
  제출 전 본 계정의 pending/running 작업과 프로젝트 실행 잠금을 확인한다.
- SSH 로그인 노드에서 학습/GPU 연산을 실행하지 않는다. 할당 완료된 동일 계산 작업 안에서
  hostname, job ID/owner/partition/nodes/GPU 수, Python/PyTorch, CUDA 가용성과 visible GPU1개를 검증한다.
- Slurm의 `CUDA_VISIBLE_DEVICES`를 설정·덮어쓰기·삭제하지 않는다. 물리 GPU 번호 선택 금지.
- 할당 불일치/가시 GPU 수 불일치/미확인 자원 상태에서는 학습하지 않는다. 자원 부족은 대기로 보고한다.
- 타 사용자/기존 작업을 중단하거나 변경하지 않는다. 취소는 이번 작업이 직접 생성하고
  owner와 job ID를 확인한 작업에만 가능하다. wildcard/user 전체 취소 금지.
- Python3.10/PyTorch2.1/A6000 48GB는 안내값이다. 실제 patch/CUDA build/환경/경로를 확인한다.
  sudo/시스템 Python 변경/전역 설치/공용 PyTorch·CUDA 업그레이드 금지.
  추가 설치가 필요하면 먼저 사용자 전용 환경의 최소 변경안을 제시한다.
- 초기 원격 점검은 작은 GPU 연산과 고정 정상 입력의 한 배치 forward/backward까지만 수행한다.
  전체 데이터 학습/여러 실험/기존 lifecycle 전체 실행을 자동 시작하지 않는다.
- 작업 종료 후 자체 생성 할당의 반납을 확인하고 접속/할당/연산/모델 시험을 별도로 보고한다.

## 승인된 원격 호환성 추가 (2026-10-14)

사용자가 AI 쪽에 PyTorch **2.5.1+cu121** 호환을 추가하여 기존
`/abr/coss43/miniconda3/envs/mamba/bin/python`(Python3.10.18)을 사용하도록 명시 승인했다.
`ai/remote/environment-mamba25.json`의 명시적 profile로 새 시험을 구분한다.
기존 로컬2.1.2 환경/v6패키지/통과 증거는 그대로 유지한다. 서버 기존 Conda 환경은 변경하지 않는다.
새 venv 설치안은 실행하지 않는다. 한 배치 통과를 전체 학습·복원 호환성 통과로 확대하지 않는다.

## 목적별 폴더 배치 (사용자 지정, 2026-10-17)

- `simulation/`: 시뮬레이션 소스·프로필·계약·PX4·빌드·생산자 잠금과 검증 자료. 내부 상대 구조와 동결 파일 내용을 유지한다.
- `Data/generation/`: 수집·학습 전 전처리 실행기와 그 설정·동결 패키지. 수집·정제 전용 Python 3.12 환경은 `Data/generation/.venv`이며 GPU/PyTorch를 설치하지 않는다. 작업·잠금·중복 방지 기록은 `Data/operations/`에 둔다.
- `Data/processed/` 등: 선택한 저장 루트의 학습용 데이터. 기본 루트는 프로젝트 `Data/`이며 외장 저장소는 필수가 아니다.
- `ai/`: 모델·학습·보정·추론·평가·원격 환경·시험·현재 회차 상태·운영 보고. AI Python 환경은 `ai/.venv`다.
- `docs/`: 명세·현재 계획·운영·개발 지침·과거 보존 자료. 오래된 회차는 `docs/archive/`에 보존한다.

새 작업은 위 실제 경로를 기준으로 작성한다. 사용자 지시에 따라 범주 이동용 호환 링크는 제거했다. 코드·설정·시험은 실제 경로를 사용하고 옛 위치를 재생성하지 않는다. 입력·출력의 임의 symlink 허용으로 검사를 완화하지 않는다. 원격 자원·승인·보안·품질·회차 정지 조건은 위 지침을 그대로 유지한다. 수집 명령은 `./Data/generation/drone-pretrain`, AI 명령은 `./ai/drone-ai`(모델 비교는 `train` 하위 명령), 시뮬레이션 명령은 `./simulation/capstone-sim`이다.

수집·정제 설치/진단은 `./Data/generation/drone-data setup --simulation`과 `doctor`를 사용한다. 지원 기준은 Ubuntu 24.04 x86_64다. 현재 생산자의 검토 소스는 `simulation/config/producer-sources.json`, 컴퓨터별 빌드 봉인은 `simulation/artifacts/runtime-manifest.json`이다. 과거 `v1…v7` 파일 선택으로 되돌리지 않는다. 과거 증거·동결 패키지의 파일 내용과 데이터 schema 버전은 보존한다. 설치기는 실행 중인 수집/정제의 잠금을 존중하며 실제 수집이나 학습을 시작하지 않는다.

## 기체 프로파일 확장 (사용자 승인, 2026-10-17)

기존 X500 동작을 유지하면서 다른 기체·추진계·배터리 프로파일을 추가하는 시뮬레이션 확장을 승인했다. 현재 지원은 고정 수직 회전축의 4~8로터 멀티콥터이며, 양의 호버 할당과 운용 여유를 검증한 프로파일만 준비한다. 원래 v4.2의 X500 범위 명세는 역사 자료로 보존하고, 현재 추가 범위는 `docs/simulation/VEHICLE_PROFILES.md`를 따른다.

새 기체의 프로파일 검증·자산 생성·컴파일·수치 시험은 실제 SITL 비행·수집 승인과 다르다. 별도 `profile prepare` 뒤 사용자가 지정한 `run --vehicle-profile`만 해당 기체를 선택한다. 기존 수집 CLI의 기본 기체는 X500이다. 새 기체의 로그는 별도 metadata/raw 계약과 `ai_compatible: false`를 사용하며 기존 X500 AI/데이터 인수 경로에 자동으로 혼합하지 않는다. AI 학습·평가 구현은 이 확장으로 변경하지 않는다.

2026-10-18 범용명 정리: 현재 데이터 명령은 `drone-data`·`drone-pretrain`, native 플러그인은 `libDronePowertrain.so`, 생성 헤더는 `vehicle_generated.hpp`다. 기체를 선택하지 않으면 기존 X500이다. X500 자체의 profile ID/계약·과거 증거·AI 모델 구현은 보존한다. 일반 doctor는 source_lock을 갱신하지 않으며 native build는 설치기의 공유 잠금을 사용한다.

## AI 클론 실행 정리 (사용자 승인, 2026-10-18)

현재 실행 패키지는 `ai/console`, `ai/learning`, `ai/execution`, `ai/pipeline`이다. 서버·사용자·포트·기존 Python·데이터·출력 경로는 `drone-ai configure`에서 직접 입력하며 `ai/.local/profiles/`는 Git에서 제외한다. `setup`과 `doctor`는 기존 환경을 읽기 전용으로 검사하며 공유 환경을 설치·변경하지 않는다. 옛 `x500-compare` 진입점은 보존 후 폐기했고 호환 링크를 만들지 않는다.

이번 승인 범위는 실제 정상 train/validation 각4개 창, LSTM·Transformer 32hidden·2층·batch2·seed0, 합계16 optimizer step의 저장·별도 프로세스 복원 기능시험이다. 본학습·보정·탐지 성능 평가 승인이 아니다. `smoke`는 서버 tmux와 단일 Slurm 작업이 수행하며 LLM 감시·자동 재시도·자동 후속 호출이 없다. 종료·반납 검산 `reconcile`은 기존 job만 읽고 기존 실패 기록을 보존한다. 본학습용 1,200run 인수와 회차 승인은 별도다. 상세 절차는 `docs/ai/PORTABLE_SETUP.md`를 따른다.
