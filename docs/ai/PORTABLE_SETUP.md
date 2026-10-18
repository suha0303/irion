# 서버 클론에서 AI 준비·시험

서버에 프로젝트를 클론한 뒤 그 루트에서 실행한다. 소스를 수동으로 추가하거나 서버 주소를 코드에 적을 필요가 없다. 데이터와 학습된 가중치는 Git 소스에 포함하지 않으므로 검증된 자료를 별도로 준비한다.

## 서버·환경·경로 설정

```bash
./ai/drone-ai configure
```

기존 Python 실행 파일, 서버 클론의 절대 경로, 데이터 경로, 결과 상위 경로를 입력한다. 서버에서 직접 쓸 때 SSH 서버는 비워 둘 수 있다. 다른 컴퓨터에서 접속할 프로필은 서버·사용자·포트를 입력한다. 인증은 SSH 키 또는 터미널 비밀번호로 처리하고 비밀번호·개인키는 프로필에 넣지 않는다.

기본 프로필은 `ai/.local/profiles/default.json`이다. Git에서 제외하고 새 파일 권한은 0600이다. 이름 또는 절대 경로를 `--profile`로 지정한다. 기존 설정을 바꿀 때만 `configure --replace`를 사용한다.

데이터 경로는 `dataset/`·`evidence/`를 포함한 전처리 인수 폴더다. 현행 전처리 출력처럼 `dataset/`과 같은 위치에 `handoff.json`·품질/봉인 증거가 있는 형식도 짧은 기능시험에서 읽는다. 기존 104run의 `dataset/`·`config.json`·`inventory.json`이 있는 봉인 input도 기능시험에 한해 지원한다. 시험 전용 자료나 기존 snapshot을 현재 본학습용 1,200run 데이터로 간주하지 않는다.

지원 환경은 Python3.10, `torch21`(PyTorch2.1.2) 또는 `mamba25`(PyTorch2.5.1+cu121)다. 선택 프로필의 NumPy·safetensors 버전도 검사한다. 프로필의 CPU·RAM 값은 짧은 GPU 시험용이다. 현재 본학습 비교는 CPU4개·RAM64GiB 계약을 별도로 유지한다. 현재 GPU 승인은 p02 / DIS01 / 단일 노드 / GPU1개다. 서버 직접 입력 기능이 다른 자원 사용을 승인하지는 않는다.

## 기존 환경 준비·진단

```bash
./ai/drone-ai setup --profile default
./ai/drone-ai doctor --profile default
```

같은 읽기 전용 진단이다. Python·필수 패키지·Slurm/tmux·데이터 존재·출력 경로를 검사하고 설치·GPU 연산을 하지 않는다. 환경이 없으면 중지하고 사용자 전용 환경의 최소 설치안을 먼저 검토한다. 공유 Conda/Python/CUDA를 바꾸지 않는다. 전처리 NPZ 학습 경로에서 쓰지 않는 원시 파서 패키지는 선택 항목으로 보고한다.

로컬 프로필의 SSH 정보로 서버 클론 경로와 tmux에 접속할 수도 있다. 서버에서도 사용할 프로필을 설정해야 한다.

```bash
./ai/drone-ai connect --profile default
```

## 짧은 GPU 학습·저장·복원 시험

설정한 결과 상위 경로 아래 아직 없는 새 시도 폴더를 지정한다.

```bash
./ai/drone-ai smoke --profile default --root /absolute/output/smoke-01
./ai/drone-ai watch --root /absolute/output/smoke-01
./ai/drone-ai status --root /absolute/output/smoke-01 --json
```

`smoke`만 tmux 유한 실행기를 시작한다. 기존 pending/running 작업과 공통 잠금을 확인하고 Slurm 작업 하나를 제출한다. 할당 안에서 소유자·노드·GPU1개·Python/PyTorch를 검증한다. 본학습과 겹치거나 미확인 자원이면 계산하지 않는다.

봉인된 실제 정상 train/validation에서 각각 4개 창을 읽고 기존 train 전용 scaler를 적용한다. LSTM·Transformer 각각 32hidden·2층·batch2로 독립 프로세스의 학습→저장→복원을 비교한다. 전체 16 optimizer step이며 작은 캐시 표본의 기능시험이다. 전체 학습 처리율·최대 batch·탐지 성능·본학습 완료를 뜻하지 않는다. 시험 가중치를 본학습 초기값으로 쓰지 않는다.

화면은 상태·단계 처리율·GPU/VRAM 기록을 읽는다. `q` 또는 Ctrl+C는 화면만 닫고 tmux/Slurm 작업은 유지한다. 로컬 SSH 창을 닫아도 서버 작업은 유지된다. 프로그램이 승인된 절차를 수행하며 LLM을 호출하거나 다음 작업을 예약하지 않는다.

| 기록 | 의미 |
|---|---|
| `server-profile.json`, `request.json` | 제출 시 고정한 설정·해시·시험 범위 |
| `control.json`, `controller.log`, `slurm.log` | 제출·실행·실제 종료/반납 확인 |
| `compute/progress.json`, `compute/result.json` | 기능시험 진행·저장/복원 검산 |
| `compute/gpu-live.json`, `compute/gpu-summary.json` | 검증된 할당 GPU 관측값 |

worker의 `SMOKE_COMPUTE_FINISHED_PENDING_RELEASE`는 반납 전 상태다. `control.json`의 `SMOKE_READY_STOPPED`는 계산 통과와 Slurm 종료·반납을 확인했다는 뜻이다. 실패·UNKNOWN·반납 미확인을 성공으로 보지 않는다. 증거를 보존하고 자동 재시도하지 않는다.

Slurm이 `COMPLETED / 0:0`이고 작업이 큐에서 사라졌는데 공유 파일시스템의 `compute/result.json`만 아직 보이지 않으면 `SMOKE_WAITING_RESULT_VISIBILITY`로 최대120초간5초 간격으로 파일만 다시 읽는다. 할당은 이미 반납된 상태이며 GPU 제출·학습 재시도는 하지 않는다. ENOENT 이외의 입출력 오류·잘못된 JSON은 즉시 실패하고,120초가 지나도 파일이 없으면 대기 증거를 남기고 중지한다.

제출 후 controller만 중지됐고 기존 GPU 작업은 정상 종료한 경우 `./ai/drone-ai reconcile --root /absolute/output/smoke-01`로 기존 job 한 개를 일회성 검산할 수 있다. 제출·재시작·GPU 연산은 하지 않으며 원래 `control.json`도 보존한다. 제출 프로필 해시·job 소유자·Slurm 종료/반납·기능시험 결과·저장/복원 증거가 모두 일치해야 별도 `reconciliation.json`을 기록한다. 이후 상태 화면은 검산 결과를 우선 표시한다. 작업이 실행 중이거나 증거가 불충분하면 성공으로 기록하지 않는다.

## 본학습 준비와 별도 승인

현재 비교는 검증된 1,200run 통합 bundle 하나, seed0, LSTM·Transformer 각각 hidden64·2층 /128·2층 /128·4층 /256·4층과 persistence/ridge_forecast, 총10종이다. batch256·FP32·최대60epoch·patience10을 유지한다. 데이터를 자동 수집하거나 이 계약을 조정하지 않는다.

```bash
./ai/drone-ai train --profile default prepare --help
```

`prepare`에는 portable bundle의 `--bundle`, 새 준비 출력 `--root`, 서버의 새 실행 경로 `--remote-root`, 학습·평가 예산 `--wall-hours`·`--evaluation-hours`를 지정한다. 무결성·역할 분리·기존 train 전용 공통 scaler·메모리를 검산하고 `PREPARED_PENDING_TRAINING_APPROVAL`에서 멈춘다. 같은 서버에서 준비하면 `--root`와 `--remote-root`를 같은 새 경로로 지정할 수 있다. 다른 서버로 옮길 때 `deploy`는 프로필 SSH 정보와 기존 키 인증을 사용한다.

본학습은 사용자가 해당 회차를 별도로 승인한 뒤에만 실행한다. 준비·smoke 통과가 승인을 대신하지 않는다. 아래는 승인 후의 명령이며 지금 실행하라는 지시가 아니다.

```bash
./ai/drone-ai train --profile default authorize --root /absolute/output/comparison-01 --approve-one-comparison
./ai/drone-ai train --profile default start --root /absolute/output/comparison-01
./ai/drone-ai watch --root /absolute/output/comparison-01
```

모델·epoch·batch·loss·단계·GPU/VRAM을 공통 화면에서 확인한다. 비교 전용 화면은 `train --profile default status --root /absolute/output/comparison-01`로도 연다.

모델은 `compute/models/`의 구성별 폴더에 가중치·best/last 체크포인트·설정을 따로 저장한다. 기존 모델을 덮어쓰거나 회차마다 자동 누적하지 않는다. 정상 보정·개발 평가·할당 반납까지 끝나야 `REPORT_READY_STOPPED`다. 잠긴 최종 시험 자료, 추가 seed·모델·재학습·재보정·다음 회차는 승인 없이 시작하지 않는다.

정식 비교도 학습·평가 할당의 성공 반납 직후 결과 파일이 아직 보이지 않으면 최대120초간5초 간격으로 파일 가시성만 확인한다. 할당 반납과 파일 대기 상태를 별도로 기록하며, JSON·상태·권한·해시 오류는 즉시 중지한다. 기존 전체 해시 검증은 파일이 보인 뒤 한 번 수행하고 계산 작업을 다시 실행하지 않는다.
