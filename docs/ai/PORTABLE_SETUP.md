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

**학습까지만 승인된 작업**은 준비 명령에 `--training-only --evaluation-hours 0`을 붙인다. 이 모드는 평가 정답을 추출하지 않으며, 보정·개발 추론·평가 작업 제출을 차단한다. 실제 train mmap과 데이터 로더로 구성별 50회 워밍업 뒤 500개 전체 학습 step을 측정하고, 전체 validation 시간은 따로 기록한다. 캐시가 데워진 측정이며 원본 압축 해제 시간은 처리율에 포함하지 않는다. 최대 epoch 기준 예상 학습·validation 시간에 25% 여유와 baseline·저장·입출력 가정 예산 1시간을 더하고, 이미 쓴 검사 시간까지 포함해 지정한 유한 상한을 넘으면 본학습 전에 정지한다. 예산을 자동 연장하지 않는다.

학습 전용 작업은 구성별 메모리·복원 검사와 시간 예산을 통과한 뒤 새 가중치로 학습하고, 모델 봉인 및 실제 Slurm 반납 확인 후 `TRAINING_READY_STOPPED_PENDING_EVALUATION_APPROVAL`에서 멈춘다. 결과는 `training-report.txt`, 측정·견적은 `compute/training-budget.json`에 남는다. 이 상태는 보정·평가 완료를 뜻하지 않으며 후속 단계에는 별도 승인이 필요하다.

사용자가 실패 후 새 시도를 승인했으며 데이터 복원은 이미 끝난 경우, 같은 서버의 새 `prepare`에 `--training-only --reuse-from /absolute/failed-attempt`를 지정할 수 있다. 이전 작업의 실패·할당 반납·소유자·설정·데이터 봉인을 검증하고 실제 학습 입력을 다시 인수한다. `--root`와 `--remote-root`는 같은 새 경로여야 하며, 새 프로필의 `dataset_root`는 이전 작업이 사용한 `upstream` 경로를 가리켜야 한다. 데이터 복사·압축 해제·전처리·기존 가중치 재사용은 하지 않는다. 이전 상태·로그는 보존하며, 새 작업이 참조하는 기존 `upstream`은 삭제하지 않는다. 준비 후 승인·시작 절차는 별도이며 자동 재시도 기능은 아니다.

RAM 판정은 노드 `MemAvailable`과 현재·부모 cgroup 제한을 함께 적용한다. cgroup에 잡힌 파일 캐시 중 보수적으로 회수 가능하다고 계산한 부분만 여유로 인정하며, 실제 제한과 최소1GiB 보호를 유지한다. 학습 인수의 통과·실패 양쪽 모두 `compute/host-memory-admission.json`에 당시 수치와 판정 근거를 남긴다. 진행 중 메모리 보호로 중지하면 해당 진행 파일 옆의 `*-memory-failure.json`을 확인한다.

본학습은 사용자가 해당 회차를 별도로 승인한 뒤에만 실행한다. 준비·smoke 통과가 승인을 대신하지 않는다. 아래는 승인 후의 명령이며 지금 실행하라는 지시가 아니다.

```bash
./ai/drone-ai train --profile default authorize --root /absolute/output/comparison-01 --approve-one-comparison
./ai/drone-ai train --profile default start --root /absolute/output/comparison-01
./ai/drone-ai watch --root /absolute/output/comparison-01
```

모델·epoch·batch·loss·단계·GPU/VRAM을 공통 화면에서 확인한다. 비교 전용 화면은 `train --profile default status --root /absolute/output/comparison-01`로도 연다.

### 저장한 모델의 후속 보정·개발 평가

학습 전용 회차가 실제로 종료·반납된 뒤 사용자가 평가를 승인하면 다음 명령을 사용한다.
프로필의 `dataset_root`는 기존에 복원·검증한 `upstream`, `output_root`는 **새 평가 출력 경로**로 설정한다.
현재 승인 환경은 기존 mamba25, DIS01/p02/GPU1/CPU4/RAM64GiB다.

```bash
./ai/drone-ai evaluate --profile default prepare \
  --source /absolute/completed-training \
  --bundle /absolute/portable-dataset \
  --root /absolute/new-evaluation \
  --approve-post-training --fresh-calibration --calibration-hours 24 --evaluation-hours 12 --queue-hours 24
./ai/drone-ai evaluate --profile default start --root /absolute/new-evaluation
./ai/drone-ai watch --root /absolute/new-evaluation
```

시간 값은 예상 소요시간이 아니라 실패 시 정지하는 유한 상한이다. 시작 뒤 서버 tmux의 실행기가
정상 보정·개발 추론을 끝내고 첫 Slurm 할당의 정상 종료·반납을 확인한 다음 별도 평가 작업을 한 번만
제출한다. 완료·실패 후 멈추며 로컬 컴퓨터·SSH·CLI 화면을 닫아도 서버 작업은 유지된다.
학습을 다시 실행하거나, 기존 회차의 설정·모델·상태 파일을 수정하지 않는다.

`ai/configs/development-evaluation.json`은 현재 평가의 역할·기준·해석 범위다.
정상 validation104run으로 오차 규모를 구하고 정상 calibration104run만으로 경보를 보정한다.
정상 개발128run·이상 개발144run을 평가하며 잠근 최종 시험은 사용하지 않는다.
정답은 추론 봉인·할당 반납 후 evaluator에서 연결한다. 탐지율·탐지 지연·정상 오경보/시간·예측 오차를
함께 보고하며, 성능이 낮다는 이유로 보정·학습을 자동 반복하지 않는다.

새 출력의 `compute/models`, `compute/calibration/<model>`에는 모델 복사본, 정상 보정 잔차와 식별자,
보정 설정, 비행별 예측·점수·판정이 보존된다. `evaluation`에는 정답·개별 결과·집계·해시 기록이 남는다.
기존 학습 폴더와 원래 평가 결과는 계속 보존한다. 다른 기준을 사용하는 후속 재집계는 별도 출력에
기준 파일과 해시를 남겨 수행하며, 자동으로 시작하지 않는다.

현재 전체 비교는 `--fresh-calibration`을 사용한다. 학습된 모델만 복사하고,
10종 모두 정상 validation 오차 규모·정상 calibration 임계값·개발 추론을 새로 계산한다.
추론 batch는256이며, 이전 보정·잔차·임계값·개발 결과를 불러오지 않는다.
이 옵션은 `--continue-from`과 동시에 사용할 수 없다.

사용자가 완료된 정상 보정의 재사용을 명시적으로 승인한 경우에만, 새 평가 `prepare`에
`--continue-from /absolute/stopped-evaluation`을 추가한다. 실제 종료·반납·빈 대기열과
정답 미사용을 검증한 뒤 완료된 정상 보정 증거만 재사용한다. 이전 개발 추론은 복사하지 않으며
10종 모두 같은 새 코드와 batch256으로 처음부터 실행한다. 기존 학습 모델·보정·중지 증거는 보존한다.
전체 보정·개발 추론 전에 10종 모두의 수치 검사를 먼저 끝낸다. 동일한 train scaler와
정상 validation257개 창·식별자를 강제한다. 정책을 지정하지 않은 기존 검사는
`rtol=1e-5`, `atol=1e-6`을 유지한다. 실패 시 멈추며 허용오차를 자동 완화하지 않는다.

수치 차이 원인 확인만 승인된 경우 새 출력의 준비 명령에 `--fresh-calibration --diagnostic-only`를
사용한다. batch1과256, 동일 batch256의 두 번 실행을 비교하고 원소별 예측·정상 정답·식별자·
물리 단위 오차를 `compute/batch-checks/`에 보존한다. 이 모드는 보정·개발 추론·평가를 제출하지
않으며 실제 할당 반납 후 `DIAGNOSTIC_READY_STOPPED`로 끝난다. 진단 완료는 평가 완료가 아니다.

근거를 검토하고 공통 기준을 확정한 뒤에는 새 전체 비교의 `prepare --fresh-calibration`에
`--numerical-policy /absolute/common-policy.json`을 지정할 수 있다. 정책을 입력 봉인에 포함하고
10종에 똑같이 적용한다. 동일 batch 반복 일치, 대칭 상대·절대 오차와 선택적 절대 상한을 검사한다.
모델별 허용치·개발 정답에 따른 조정·자동 완화는 지원하지 않는다. 전 모델이 통과해야 정상
보정부터 새로 시작한다. 작은 예측 차이가 임계값 근처의 경보까지 동일함을 보장하지는 않는다.
신경망은 FP32와 동일 CUDA backend를 사용하고, 두 baseline은 기존 NumPy 구현을 유지한다.

연속 정상 구간의 점수·경보 민감도까지 확인하는 승인 회차는 새 출력의 준비 명령에
`--fresh-calibration --sensitivity-policy /absolute/project/ai/configs/sensitivity-policy.json`을
사용한다. `--numerical-policy`, `--diagnostic-only`, 이전 보정 재사용과는 함께 사용할 수 없다.
준비 과정에서 정상 hover·payload·lowSOC의 run/group 분리와 구간 선택을 봉인한다.
예측값을 읽지 않고 정한 일반 검사 구간과, 이전 실패 시점의 명시적인 회귀 검사 구간을 구분한다.
GPU 제출 전에는 작은 식별자 배열과 NPZ 헤더로 구간 길이·연속성을 확인하며, 실제 신호의 유한성과
모델 출력은 계산 노드에서 다시 검사한다.

진단용 오차 규모 산정288창·보정288창·비교706창을 전 모델에 동일하게 사용한다.
batch는 run별로256개 이하와 마지막 남은 창으로 나누고, batch1과256 비교 및256 반복을 수행한다.
같은 모델의 두 경로에는 같은 진단용 오차 규모·임계값을 고정한다. 정규화 절대 차이1e-4,
각속도 차이1e-5rad/s·팩 전압 차이1e-4V, 임계값 대비 점수와 EWMA 차이1% 이하,
영역별/통합 경보 유지·시작·종료 변화0건을 모두 요구한다. 이는 공통 공학적 예산이며
센서 정밀도 보증값이나 탐지 성능 합격선이 아니다. 기존 혼합 상대/절대 검사 위반 수는 참고 기록한다.

정확한 예측·잔차·점수·경보·표본 식별자는 `compute/sensitivity/<model>/`에 남는다.
진단 보정의0.08시간은 오경보율 인증에 부족하므로 실제 경보 수와 검사 범위를 함께 보고한다.
모든 모델이 통과해야 본 보정과 개발 추론을 시작하며 진단용 보정은 절대 재사용하지 않는다.
실패하면 기준을 바꾸거나 재시도하지 않고 멈춘다. 통과하면 기존 유한 controller가 새 보정·
개발 추론·독립 평가와 실제 Slurm 반납까지 수행하고 종료한다. LLM 감시·자동 호출은 없다.
이 표본 검사는 모든 임계값 판정의 동일성을 보증하지 않는다.

최적화한 오프라인 추론은 예측을 묶어 계산하되 각 예측 발행·실측 도착·판정은 원래 논리 시간순으로
처리한다. 파일 동기 저장은 run 봉인 시 수행하므로 중단된 미봉인 run은 완료로 취급하지 않는다.
배치 처리 시간은 건당 평균 분담 시간으로 기록하며 실시간 단일 요청 지연으로 해석하지 않는다.

모델은 `compute/models/`의 구성별 폴더에 가중치·best/last 체크포인트·설정을 따로 저장한다. 기존 모델을 덮어쓰거나 회차마다 자동 누적하지 않는다. 정상 보정·개발 평가·할당 반납까지 끝나야 `REPORT_READY_STOPPED`다. 잠긴 최종 시험 자료, 추가 seed·모델·재학습·재보정·다음 회차는 승인 없이 시작하지 않는다.

정식 비교도 학습·평가 할당의 성공 반납 직후 결과 파일이 아직 보이지 않으면 최대120초간5초 간격으로 파일 가시성만 확인한다. 할당 반납과 파일 대기 상태를 별도로 기록하며, JSON·상태·권한·해시 오류는 즉시 중지한다. 기존 전체 해시 검증은 파일이 보인 뒤 한 번 수행하고 계산 작업을 다시 실행하지 않는다.

다른 기준의 재집계는 별도로 요청했을 때 다음과 같이 실행한다. `criteria.json`의 지원 필드는
`event_success` (`awareness` 또는 `new_episode`), `max_detection_delay_s` (초 또는 `null`),
`threshold_multiplier` (추진·배터리 순의 양수 두 개)다. 생략한 항목은 원래 기준을 유지한다.
고정 배율은 저장한 점수를 재판정할 뿐 정상 threshold를 다시 적합하지 않는다.
개발 결과를 본 뒤 정한 기준은 탐색적 결과로 구분하며 기존 평가를 대체하지 않는다.

```bash
./ai/drone-ai reassess --profile default -- \
  --evaluation /absolute/completed-evaluation/evaluation \
  --criteria /absolute/criteria.json --output /absolute/new-reassessment
```
