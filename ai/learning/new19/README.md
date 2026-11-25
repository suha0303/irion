# X500 V0.2 AI 모델 8종 실행

비행 중과 전체 런에서 각각 LSTM·Transformer `64×2`, `128×4`를 학습한다. 숫자는 hidden 크기와 스택별 층수다. 모델 ID는 `inflight-lstm-64x2` 또는 `whole-run-transformer-128x4`와 같은 형태다. [설계](../../../docs/ai/NEW_EIGHT_MODEL_DESIGN.md)

```bash
./ai/drone-ai train new19 models
./ai/drone-ai train new19 smoke --output ai/artifacts/new19-smoke
```

합성 CPU 시험은 8개마다 정상 입력의 2 epoch로 순전파, 역전파, 검증, 저장과 복원을 확인한다. 첫 epoch에서 끊고 같은 계획으로 재개한다. 본학습 산출물과 다른 위치에 둔다.

## 새 SITL 원자료 준비

아래 인자는 수집 결과에 있는 실제 경로와 확인된 PX4 ID로 채운다. `--buffered`는 `execution.json`의 모터 시동(`armed`)부터 착륙 후 비무장(`disarmed`)까지 확인하고, MAVLink에서 확인된 이륙 준비·이륙·순항·착륙 하강을 학습 배열에 보존한다. 이미 지상 상태인 앞뒤 프레임과 이륙·착륙 양끝의 미관측 프레임 수는 인수 기록에 남긴다. 비행 도중의 공백은 마스크로 보존한다. 시험에서 사용한 X500 프로필은 4S·12 Ah와 모터 채널을 명시한다.

```bash
./ai/drone-ai train new19 prepare \
  --capture "$CAPTURE" --profile "$PROFILE" --execution "$EXECUTION" \
  --source-system "$FC_SYSID" --source-component "$FC_COMPID" --imu-id "$IMU_ID" \
  --buffered \
  --output ai/artifacts/new19-data/flight-0001
```

`--buffered`는 10,000 ms를 기다려 100 ms 간격으로 common19 프레임을 확정한다. PX4 부팅 시각을 확인할 수 있는 관측은 측정 시각으로 정렬하고, 부팅 시각에 대응시킬 수 없는 메시지에는 수신 시각을 쓴다. 마감 이후 도착한 관측 수와 시계 복구·수신 시각 사용 횟수를 기록한다. 같은 UDP 원문과 수신 시각을 재생하면 `BufferedPreprocessor`가 같은 프레임을 만든다. 저장된 앱 세션은 기록 당시 전처리 버전으로 재생한다. 과거 시험의 수신 시각 전용 변환은 `--buffered` 없이 재현한다.

`run.npz`는 `values[T,19]`, `time_us[T]`, `segment[T]`, `observed[T,4]`를 보존한다. 값의 물리 단위는 팩 V/A가 포함된 common19다. `observed`는 각속도 3축과 전압이 학습·평가 정답으로 사용 가능한지 표시한다. 짧은 공백은 기존 공통 인과 샘플러 정책으로 보충하고, 보충된 타깃은 정답에서 제외한다. 길이가 긴 공백은 비행 중 창을 끊고, 전체 런 입력의 연속성이 깨지면 결과는 `unavailable`이다. `receipt.json`에는 원문·프로필·실행 구간·배열의 해시와 품질 집계를 기록한다.

학습 매니페스트의 schema는 `drone-ai.new19.dataset/1`이다. `features`는 실제 common19 순서이며 `runs`의 각 행은 `run_id`, `role`, `path`, `sha256`, `raw_sha256`, `group_id`, `retry_group`, `pair_id`를 포함한다. `path`는 매니페스트 위치의 하위 파일 경로다. `pair_id`는 짝이 없으면 `null`이다. 같은 원본·재시도·짝·그룹은 한 데이터 역할에 둔다. `normal_train`, `normal_validation`, `normal_calibration`, `normal_test`, `anomaly_test`를 별도로 기록한다. 학습기는 정상 train과 validation 배열만 열며 정상 적격성 및 역할 지정은 수집 인수 자료에 근거해야 한다.

## 학습과 예측

비행 중 모델의 정상 학습 창은 연속 구간별 첫 유효 예측 시작점부터 1초 격자로 뽑는다. LR Range Test 및 본학습에 같은 격자를 쓰며, 정답이 부족한 슬롯은 건너뛴다. 입력 100프레임과 미래 10프레임은 10 Hz다. 정상 검증, 보정, 최종 탐지 및 실시간 재생에는 100 ms 시작점 간격을 적용한다.

```bash
./ai/drone-ai train new19 lr-range --model inflight-lstm-64x2 \
  --manifest "$MANIFEST" --output ai/artifacts/new19-lr1

./ai/drone-ai train new19 fit --model inflight-lstm-64x2 \
  --manifest "$MANIFEST" --output ai/artifacts/new19-fit1 \
  --lr "$SELECTED_LR" --epochs 20 --until-epoch 2

./ai/drone-ai train new19 fit --model inflight-lstm-64x2 \
  --manifest "$MANIFEST" --output ai/artifacts/new19-fit1 \
  --lr "$SELECTED_LR" --epochs 20 --resume

./ai/drone-ai train new19 calibrate --bundle ai/artifacts/new19-fit1 \
  --manifest "$MANIFEST" --output ai/artifacts/new19-calibration1

./ai/drone-ai train new19 predict --bundle ai/artifacts/new19-fit1 --run "$PREPARED_RUN" \
  --calibration ai/artifacts/new19-calibration1/calibration.json \
  --output ai/artifacts/new19-replay1
```

`lr-range`는 모델마다 새 초기화로 실행한다. EMA 학습 손실 곡선에서 정한 LR을 `fit --lr`로 지정한다. `--resume`의 계획·소스·입력 해시가 다르면 복원을 거절한다. 내보낸 `model.safetensors`는 검증 손실 기준 최적 가중치다. `predict`에서 보정 파일을 빼면 예측과 잔차만 저장한다. 절대 단위는 비행 중 각속도 `rad/s`와 팩 `V`, 전체 런은 셀 환산 `V`다.

기존 `FlightSession`의 C14 미래 예측은 100 ms마다 연속 입력을 받고 `t+100 ms` 전 생성 완료만 등록한다. 명령의 `predict`는 저장 자료의 논리 재생이다. 앱의 10초 경로는 확정한 과거 프레임을 각 프레임의 실제 발행 시각과 함께 별도로 보관한다. 현 진입점의 실행은 로컬 CPU용이다.

## 실기체 파인튜닝 기본 절차

기체 또는 확인된 제어기별로 정상 학습·정상 보정·독립 평가 로그를 분리한다. 비행 중 후보는 로컬 CPU에서 원본 모델의 출력층부터 적응하고, 고정한 후보의 정상 잔차로 점수별 99백분위 판정 기준을 보정한다. 3초 연속 초과 조건을 적용한 채 동일 예측으로 오경보와 이상 검출을 비교한다. 세부 기본값, 2026-11-21 실기체·SITL 검증 결과 및 앱 연동 상태는 [로컬 실기체 체크리스트](../../../docs/ai/LOCAL_REAL_FLIGHT_CHECKLIST.md#기본-파인튜닝-절차)에 기록한다.

비행 중 모델 번들을 연결한 앱에서 **로컬 AI · 기체 관리 → 로그 준비**를 열고 같은 기체 프로파일로 구간을 저장한다. 각 구간의 용도와 확인된 건강 상태의 근거를 입력한다. **파인튜닝 · 보정**에서 학습·보정·평가 구간과 실제 수집 날짜를 각각 선택한 뒤 **후보 만들고 독립 평가**를 누른다. 담당자가 입력한 수집 날짜와 원문 해시를 기준으로 역할 간 겹침을 검사한다. **작업 결과**에서 오차와 경보를 비교하며, 후보 모델과 보정 기준은 작업별 검증 대기 산출물로 보관한다.

**로그 분석**에서는 준비된 한 파일을 V0.2 모델 한 종 또는 여러 종으로 분석한다. **파인튜닝 · 보정**에서는 한 종, 비행 중 4종, 종료 후 4종, 전체 8종을 고를 수 있다. 종료 후 모델은 초기 10초를 이력으로 사용하고 후속 실측 전압으로 마지막 선형 층만 조정한다. 정상 보정 로그별 구간 점수 최댓값의 경험적 99백분위를 사용한다. 결과에서 모델과 로그를 골라 예측·실측·잔차를 비교한다. 후보는 작업별 검증 대기 산출물이며 운영 모델과 분리한다.

## 앱에 PX4 MAVLink 직접 연결

학습한 **비행 중 예측용** 번들 네 개와 각 번들의 보정 결과가 들어 있는 경로를 지정한다.

```bash
ARION_PORT=8880 ARION_AI_ROOT="$PWD" \
  ARION_V02_INFLIGHT_ROOT="$PWD/ai/artifacts/new19-best-models-20261121" \
  ./irion-upload/UI/start.sh
```

앱에서 UDP 세션을 열고 등록된 PX4 기체 프로파일을 선택한다. MAVLink 시스템·컴포넌트 ID와 HIGHRES_IMU ID를 설정한 뒤 `비행 중 모델 4종 시작`을 누른다. 네 모델은 하나의 10초 버퍼가 확정한 같은 프레임으로 각자 예측하고 해당 모델의 보정 기준으로 판정한다. 그래프의 `모델` 선택은 표시할 결과를 바꾸며 네 모델의 계산은 계속된다. PX4 SITL의 프로젝트 수집 설정은 인스턴스 `i`에 UDP `14700+i`, 시스템 ID `1+i`를 사용한다. 시험은 `simulation/capstone-sim run --configuration collection --mavlink-only`로 실행한다. 기체 송신 주기가 준비된 링크를 사용한다.

한 모델을 실행하려면 `ARION_V02_INFLIGHT_ROOT` 대신 `ARION_NEW19_BUNDLE`과 `ARION_NEW19_CALIBRATION`을 지정하고 앱에서 `PX4 MAVLink · 10초 버퍼 모델 사용`을 켠다.

앱의 UDP 수신기가 받은 바이트와 수신 시각을 원문 파일 `.mavlink.bin`에 보존한다. 10초 버퍼의 100 ms 프레임으로 예측하고, 유효한 실측과 비교해 판정한다. 새 세션의 분석 기록은 첫 유효 프레임에서 시작해 마지막 유효 프레임으로 끝난다. 양끝의 결측 프레임을 제외하고 비행 도중의 공백은 무효로 표시한다. 실행 중 확정한 실제 발행 시각은 세션 기록에 보존한다.

같은 번들 경로에 종료 후 모델 4종도 있으면 **비행 종료 · 결과 분석** 뒤 확정된 관측 프레임으로 비행 후 판정을 자동 계산한다. 메인 모니터링에서 비행 중 결과와 종료 후 점수·임계값을 함께 볼 수 있다. 업데이트 전에 저장한 PX4 기록은 **이 비행의 종료 후 판정 계산**으로 결과를 한 번 저장할 수 있다. 개발 검증에서는 세션의 `replay-buffered` API로 저장 원문과 실시간 처리의 재현성을 확인한다.

## 전체 런 모델 4종을 연결한 앱

`ARION_V02_WHOLE_ROOT`에는 네 `whole-run-*` 디렉터리가 들어 있는 경로를 지정한다. 각 디렉터리는 `fit/bundle.json`, `fit/model.safetensors`, `calibration/calibration.json`을 포함한다.

```bash
ARION_PORT=8879 ARION_AI_ROOT="$PWD" \
  ARION_DATA_DIR="$PWD/ai/.local/new19-whole-live-20261120" \
  ARION_V02_WHOLE_ROOT="$PWD/ai/reports/new19-production-20261120/attempt-01/finished" \
  ./irion-upload/UI/start.sh
```

이 모드에서 앱은 등록된 PX4 X500 UDP 세션에 네 전체 런 모델을 함께 연결한다. 비행 중에는 관측을 표시한다. `비행 종료 · 결과 분석`을 누르면 원문 기록을 마감하고, 10초 버퍼를 정리한 뒤 네 모델의 점수·보정 기준·판정을 표시한다. 원시 UDP와 분석 배열 및 모델별 예측은 세션 데이터 경로에 보존한다.

새 UDP 세션은 비행 관측을 `.mavlink.bin`에만 기록한다. 세션 JSONL에는 설정과 재접속 위치를 저장한다. 완료된 비행을 열면 UDP 원문에서 100 ms 간격의 화면 재생 프레임을 메모리에 만든다. 이전에 저장된 세션은 기존 형식으로 열 수 있다.


## 이전 모델 실행

이전 16종은 V0.2 번들 환경 변수를 설정하지 않은 별도 앱에서 실행한다. 별도 포트와 데이터 디렉터리를 지정하면 V0.2의 작업과 구분할 수 있다.

```bash
env -u ARION_V02_INFLIGHT_ROOT -u ARION_V02_WHOLE_ROOT \
  -u ARION_NEW19_INFLIGHT_ROOT -u ARION_NEW19_WHOLE_ROOT \
  ARION_PORT=8878 ARION_AI_ROOT="$PWD" ARION_DATA_DIR="$PWD/ai/.local/legacy-ui" \
  ./irion-upload/UI/start.sh
```
