# 비행 종료 후 run 판정: LSTM / Transformer 8종

기존 `learning.history19`의 비행 중 미래 예측과 별개인 신규 실험이다.
2026-11-08 사용자 지시에 따라 비행 중 예측과 같은 8종을 사용한다.
LSTM·Transformer 각각 h64/l2, h64/l4, h128/l2, h128/l4이며 heads4·SGD를 유지한다.
층 수는 local encoder와 run encoder 각각의 수다. CLI 기본값은 기존 h64/l2다.
동일한 계층 구조·입력·전압 목표·loss·점수 정의를 사용하며 전체 parameter 수는 다르다.

```
완료된 run의 임무 구간 [T,19], 10Hz
  ├─ 첫 100시점으로 초기 상태 인코딩
  └─ 이후 측정 전압을 가린 19개 관측
       → 100시점씩 local encoder (선택한 hidden64/128, 2/4층)
       → run encoder (같은 hidden, 같은 층 수)
       → 같은 pointwise head + 해당 시점 부하 관측
       → 건강한 상태의 예상 전압 [T]
측정 전압 − 예상 전압 → 단기 응답 / 장기 경향 잔차 → run 점수
```

두 encoder를 LSTM 또는 Transformer로 바꾼다. 모든 모델은 정상 run만 학습하며
정상 validation만 모델 선택에 사용한다. 이 실험의 출력은 전압 반응 이상이며,
실제 열화 원인·SOH·RUL을 확정하는 값이 아니다. 다른 부품 이상도 점수를 올릴 수 있다.

2026-11-07 사용자 승인으로 run 입력 범위는 **실제 mission 시작 이상, landing 시작 미만**이다.
생산자의 `evaluation/execution.json`에 기록된 두 `sim_time`을 PX4 boot us로 변환한다.
이착륙과 그 품질은 전체 canonical 증거에 보존하고, 임무 밖이라는 이유로 따로 집계한다.
유효한 값이 있는 곳을 찾아 경계를 옮기지 않는다. 경계 누락·중복·역전, 임무 내부의
결측·gap·segment 변경, 선언한 임무 양끝 누락은 unavailable이다. 결측을 보간하지 않는다.
첫100시점 초기 상태와 마지막 부분 block도 이 명시된 임무 구간을 기준으로 한다.

19개 신호의 순서는 `capstone_ai.mapping.FEATURES`와 같다. 추가 60초 통계·누적량·
실제 SOC/Q/R·열화 단계·미래 target은 입력하지 않는다. 전압은 4로, 전류는 고정
명판 12Ah로 나눈 뒤 **정상 train의 고유 전체 frame만으로** 계산한 mean/std를 적용한다.
run마다 전압 평균을 다시 빼지 않는다. 관측 전압은 첫 100시점의 초기 상태 추정과
학습 target/점수 계산에만 사용한다. 이후 reference 경로의 전압 channel은 가린다.
이는 직접 전압 복사를 차단하는 설계다. 초기 상태의 열화 흡수, 부하·잔량 혼동,
전류와 전압의 물리적 결합 때문에 탐지력이 충분하지 않을 가능성은 실제 자료로 평가해야 한다.

10초에 한 점을 취하지 않는다. 10Hz 100시점을 encoder가 읽고 선택한 hidden 크기의 block
벡터로 만든다. 10초 block은 겹치지 않는다. 마지막 1~99시점도 mask와 실제 길이로
보존한다. run 길이 차이는 padding/mask로 처리한다. 전압 head는 각 시점의 부하와
local encoder의 시점별 상태도 받으므로 block 평균만으로 짧은 응답을 지우지 않는다.
run encoder는 이전/현재 block만
참조하지만 각 block 안의 관측은 모두 이용한다. **비행 종료 후 모델이며 live forecast가 아니다.**

첫 10초는 초기 상태 구간으로 미리 정했으며 점수를 내지 않는다. 남는 시점이 없거나,
gap·clock reset·누락이 있는 run은 `unavailable`로 처리한다. 현재 버전은 단일 연속
segment만 지원하며 reset 전후를 붙이거나 결측을 0으로 채우지 않는다. 독립 평가에서는
이런 run도 분모·coverage·미탐 수에 남겨야 한다. 정상 판정으로 간주하거나 제외하면 안 된다.

전압 잔차를 train scaler로 셀 환산 V로 돌린 뒤, block 안의 평균 제거 RMS를 단기 성분,
block 평균의 frame 가중 RMS를 장기 성분으로 둔다. 두 성분의 제곱합은 전체 잔차 MSE다.
`scoring.Calibration`은 정상 validation으로 성분 scale, 별도 정상 calibration으로
max 성분 점수의 순위 임계값을 만든다. test/이상 자료의 임계값 적합은 거절한다.
alpha .05일 때 유한 순위 임계값에는 최소19개 독립 calibration group이 필요하다.
이는 최소 계산 조건이며 충분한 정상 범위·오경보 성능을 보장하지 않는다. alpha .05와
오경보 목표5%는 제안 상태이고 독립 평가 기준은 아직 잠그지 않았다.

## 이번 실행 범위

FP32, SGD만 사용한다. 기존 층간 LN/Post-LN, no dropout, seed0,
초기화, gradient clip1, warmup/cosine 정책을 재사용한다. CPU 기능시험은
합성 정상 train4개 / validation2개, run batch2, lr .01, 4epochs, 모델당8 optimizer step이다.
test·이상 데이터를 보거나 lr를 고르는 과정이 없다. 본학습 batch는 run 전체의 길이별
메모리 검사가 필요하므로 미정이며 기존 온라인 모델의 batch256창을 256run으로
그대로 바꾸지 않았다. 신규 run 모델의 GPU 실행기·본학습·조기 종료·LR 비교 회차는
아직 연결하지 않았다. CPU fit은 최대16run/총20,000frame의 개발 시험으로 제한한다.

```bash
./ai/drone-ai train run-health --help
./ai/drone-ai train run-health smoke --model lstm --output ai/artifacts/run-health-lstm-full
./ai/drone-ai train run-health smoke --model transformer --output ai/artifacts/run-health-transformer-full
./ai/drone-ai train run-health smoke --model transformer --hidden 128 --layers 4 --output ai/artifacts/run-health-transformer-h128-l4-full

# 같은 4epoch 계획을 2epoch에서 저장하고 별도 프로세스에서 재개
./ai/drone-ai train run-health smoke --model lstm --stop-after-epoch 2 --output ai/artifacts/run-health-lstm-part
./ai/drone-ai train run-health smoke --model lstm --resume ai/artifacts/run-health-lstm-part/last.pt --output ai/artifacts/run-health-lstm-resumed

PYTHONPATH=ai:ai/src:Data/generation ai/.venv/bin/python -m pytest ai/tests/test_run_health19.py -q
```

매번 새 출력 경로가 필요하다. CPU checkpoint와 best model의 safetensors, scaler, 모델 명세,
고정 계획, 입력 fingerprint, 코드 hash, 손실과 integrity를 저장한다. 복원은 데이터·코드·
scaler·계획·환경이 다르면 거절한다. GPU는 이 CLI에서 실행하지 않는다.

## 새 전체 run 데이터 인수

`data.load_manifest`는 새 schema `drone-ai.run-health19.dataset/1`만 받는다.
각 NPZ는 `values[T,19]`와 `time_us[T]` 두 배열만 가지며 물리 common19의 pack V/A다.
추가 truth/C 배열, symlink, hash 불일치, 불연속100ms 시각을 거절한다.
manifest의 feature_order는 위 FEATURES 전체 목록, `units`는
`physical_common19_pack_V_A`, step_us=100000, vehicle_profile=`x500_reference`,
battery_policy=`cmu-vah01-rpt16-only/1`이다. runs 항목은 다음을 지정한다.
추가로 최상위 `interval_policy`는 `mission_only_actual_execution/1`이어야 하며,
각 run의 `mission_interval`에 같은 policy, `start_us`, `end_us_exclusive`를 기록한다.
NPZ는 이 범위의10Hz 격자 전체를 포함해야 한다. 기존 전체 airborne 입력을 자동 인수하지 않는다.

```json
{
  "run_id": "actual-producer-run-id",
  "role": "normal_train",
  "lineage": ["raw-flight:ACTUAL", "retry-family:ACTUAL", "paired-case:ACTUAL", "group:ACTUAL"],
  "health": "confirmed_healthy",
  "origin": "px4_sitl",
  "path": "runs/actual-run.npz",
  "sha256": "ACTUAL_FILE_SHA256"
}
```

role은 normal_train/normal_validation/normal_calibration/normal_test/anomaly_test다.
raw 비행·retry·paired case·group 중 하나라도 split을 넘으면 거절한다. 모든 계보 ID와
건강 적격성의 원자료 대조는 데이터 인수 단계에서 별도로 확인해야 한다. 현재 구현은
제공된 metadata를 검사하며 그 사실 자체를 인증하지 않는다.

기존 생산자·raw/C9 계약·온라인 dataset을 바꾸거나 자동 변환하지 않는다. 새 VAH01
rpt_00 정상 train/validation/calibration 및 독립 정상/rpt_16 이상 test의 인수는 미실행이다.
`validate --manifest PATH`는 schema 검사이며 SITL 적격성·탐지 성능 통과가 아니다.
실제 자료에서 모델/scale/임계값/평가 규칙을 먼저 잠근 뒤 새 독립 test를 열어야 한다.
열화 run 검출률80%는 목표이고, 이번 합성 소프트웨어 시험으로 달성 여부를 판단하지 않는다.
# P27 예비 인수 보완 (2026-11-07)

평가 role은 `normal_test`/`anomaly_test`이고 `cohort`를 preliminary/development/final 중
하나로 명시한다. 같은 cohort의 pair/group은 정상·이상 사이에 공유할 수 있으나 raw/retry
공유나 개발·최종 간 계보 중복은 거절한다. 학습/validation/calibration은 기존 role이 partition이다.
여러 role 파일을 따로 검사한 결과만으로 전체 독립성을 주장하면 안 된다.

`iter_manifests([...], expected_producer=FINAL_BINDING)`는 모든 소형 metadata를 먼저
교차 검산한 뒤 한 run씩 decode한다. CLI도 이 경로를 사용한다. `--manifest`를 여러 번
지정해 전체 cohort/role 파일을 함께 전달한다. 실제 데이터에는 최종 producer 봉인과
per-run source_evidence가 필요하고 과거 producer는 자동으로 수용하지 않는다.

```bash
./ai/drone-ai train run-health validate \
  --manifest NORMAL_ROLE_JSON --manifest ANOMALY_ROLE_JSON \
  --producer-binding FINAL_BINDING_JSON
```

run 하나는 최대10,000frame을 header 단계에서 제한하고 hash를1MiB chunk로 계산한다.
호출자는 반환한 run을 처리 후 버려야 한다. 전체 ndarray list를 만드는 `load_manifest`는
개발용 최대16run convenience다. scaler.fit/batch/CPU fit은 여전히 작은 개발 입력을
모으는 경로이며 대규모 본학습 실행기로 검증한 것이 아니다.
NPZ에 없는 실패·불가용 run도 Data의 전체 case/attempt 품질 장부에는 남겨야 한다.
reader 오류를 낸 run만 빼고 남은 파일을 재검산한 결과를 전체 통과라고 보고하면 안 된다.

세부 필드/단위/품질 형식과 미검증은
`ai/reports/p27-precollection-20261107/attempt-01/REPORT.md`를 따른다.
