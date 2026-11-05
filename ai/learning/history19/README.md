# History19 SGD

새19입력 미래 예측 구현이다. 기존 학습 모델·데이터셋·동결 패키지를 읽거나 변환하지 않는다.
기존 구현은 그대로 보존한다. run 전체 열화 판정은 별도 모델의 범위다.

입력은 `model(history)`의 **[B,100,19]** 하나다. 문맥 인자, C 배열, context token,
60초 요약과 누적량 계산이 없다. 출력은 기존과 같은 [B,10,4], 각속도p/q/r와 팩 전압이다.
10Hz·100시점 history(첫~끝9.9초)·1초 미래·1초 발행 간격을 유지한다.

새 전처리 엔진은 기존 native mapping/단위/좌표·인과 필터·quality/segment/watermark 규칙을
공유하고, 문맥을 계산하는 코드는 포함하지 않는다. history100과 비행/품질 조건이 갖춰지면
창을 발행한다. 60초 대기는 없다. 초기부터 정상인 fixture의 첫 origin은10초다.
IIR 필터 상태는 유지되므로 전체 정보 지지 구간이 정확히10초로 제한됐다는 뜻은 아니다.

## 그대로 유지한 학습 조건

- LSTM/Transformer: h64·l2·heads4, h64·l4·heads4, h128·l2·heads4, h128·l4·heads4.
  각 구조를 SGD로만 실행하는8종이다. LSTM 층 사이LN, Transformer Post-LN,
  dropout 없음, 기존 gate별 초기화·유효 forget bias+1·Xavier/orthogonal 정책을 유지한다.
- seed0·batch256·FP32·AMP 없음·최대60epochs·patience10·global gradient norm clip1.
- vanilla SGD: momentum0·weight_decay0·Nesterov 없음. 후보0.001/0.01/0.03을
  각3epochs, 정상 train 최대8192/validation 최대4096창에서만 비교한다.
  후보별 동일 초기 가중치, 최종 validation loss 최솟값, 동률은 선언 순서다.
  전체8종×3후보=24번의 유한 probe 후 main은 다시 초기화한다.
- 기존 warmup5%·시작비율0.1·cosine 최저비율0.01·restart 없음·warmup 이후 조기종료.
  target loss는 기존 각속도3개 평균0.5 + 전압 평균0.5다.
- 본 학습 전 batch256의50warmup/500step 메모리·속도 검사,512창의4epoch와2+2epoch
  별도 프로세스 복원 검사, 동일장치 best checkpoint/export 비교(atol1e-6)를 유지한다.
- CUDA는 기존 할당 내부에서만 실행한다. 기존 runtime checker·프로필·공유 GPU lock·
  p02/DIS01/node1/task1/GPU1·단일작업·CUDA_VISIBLE_DEVICES 보존 규칙을 사용한다.
  설치/SSH/Slurm 제출/추가 할당/자동재시도/자동재개/보정/평가는 이 CLI가 시작하지 않는다.
  학습 프로그램 종료 뒤 할당 반납·reconcile은 해당 할당의 외부 실행기가 확인한다.

설정은 `ai/configs/history19-sgd.json`에 있다. 기존 normalization-study와 비교해 입력과
SGD 선택 이외의 조건 변경을 거절한다. 실제 GPU 검사·데이터 인수·본 학습은 미실행이다.

## 새로운 데이터셋 계약

manifest schema는 `drone-ai.history19.dataset/1`이다. 옛 ctx6 데이터·scaler는 거절한다.
아래 경로는 manifest 기준 상대경로이고 각 항목에 실제 SHA256을 제공한다.

```json
{
  "schema": "drone-ai.history19.dataset/1",
  "feature_profile": "common19_ctx0",
  "feature_order": ["기존 registry 순서의19개 이름"],
  "target_order": ["body_rate_p", "body_rate_q", "body_rate_r", "pack_voltage"],
  "history": 100, "future": 10, "step_us": 100000, "emit_every_us": 1000000,
  "scaler_scope": "normal_train_unique_valid_frames",
  "scaler_fit_run_ids": ["새 train run ID"],
  "runs": [
    {"run_id":"새 train run ID","role":"normal_train","raw_flight_id":"독립 비행 ID",
     "paired_case_id":"train pair","group_id":"train group","health_eligibility":"confirmed_healthy"},
    {"run_id":"새 validation run ID","role":"normal_validation","raw_flight_id":"다른 독립 비행 ID",
     "paired_case_id":"validation pair","group_id":"validation group","health_eligibility":"confirmed_healthy"}
  ],
  "files": {
    "normal_train":{"path":"normal_train.npz","sha256":"생산 후 실제 hash"},
    "normal_validation":{"path":"normal_validation.npz","sha256":"생산 후 실제 hash"},
    "normal_train_frames":{"path":"normal_train_frames.npz","sha256":"생산 후 실제 hash"},
    "scaler":{"path":"scaler.npz","sha256":"생산 후 실제 hash"}
  }
}
```

이 예시는 구조 설명이며 실제 run/통과 증거가 아니다. feature_order는
`capstone_ai.mapping.FEATURES`와 정확히 일치해야 한다.

- 역할별 NPZ 멤버는 **X, Y, run_id, clock_id, segment_id, origin_us**만 허용한다.
  X=[N,100,19], Y=[N,10,4], 식별 배열=[N]이다. C·truth·미래 특징 추가는 거절한다.
  X는 과거 관측만, Y는 같은 clock/segment의 origin+0.1~1.0초 도착 실측이다.
- 기존 물리 변환을 유지한다: 팩 전압/4, 팩 전류/명시된 정격Ah. 대상 전압도/4.
  기본 새 전처리 설정은 현재 VTC6_4S4P12Ah source 설정에 ctx0만 적용한 것이다.
  실제 SOC·건강값은 입력이 아니다. scaler 이전 값으로 NPZ를 전달한다.
- `normal_train_frames.npz`는 정규화 검산용 **values[M,19], run_id, clock_id,
  segment_id, time_us**다. 생산자가 선택한 모든 valid airborne train frame을 한 번씩
  담는다(초기 history warmup 포함). 값은 X와 같은 물리 변환 후 값이다.
  중복/역순 frame과 train 외 run을 거절한다. 이 파일이 추가 모델 입력이 되지는 않는다.
- scaler는 **x_mean[19], x_scale[19], y_mean[4], y_scale[4]**만 허용한다.
  train frame에서 mean/std(ddof0,floor1e-5)를 재계산해 atol=rtol=1e-10으로 검산한다.
  target scaler는 X의 각속도/전압 열을 사용한다. Y·validation에서 fit하지 않는다.
  원본 frame의 물리/관측 quality 적격성은 새 데이터 생산 단계에서도 검증해야 한다.
- run/raw flight/paired/group/retry 계보가 train과 validation 사이에 겹치면 중단한다.
  결측 target을0으로 채우지 않는다. `arrived_samples`는 미도착/비정상 target을 제외
  사유와 함께 반환한다. 입력 창 생성은 미래 target 도착 여부와 분리되어 있다.
- 정상 train/validation만 인수한다. calibration/dev/final test는 이 학습 manifest에 없다.
  창 수·run 수는 새 데이터셋이 결정한다. 기존1200run을 자동 선택하지 않는다.

## 실행

코드 확인·입력 형식 설명(서버 접속 없음):

```bash
./ai/drone-ai train history19 --help
./ai/drone-ai train history19 validate --manifest /absolute/new-dataset/manifest.json
./ai/drone-ai train history19 preprocess --run /absolute/new-raw-run --output /absolute/new-output
```

validate는 실제 새 입력이 제공된 뒤 실행한다. train/validation X/Y는 chunk512 float64
정규화 후 float32 mmap으로 적재한다. scaler 검산용 train frame 메모리까지 호스트 계획에
포함한다. 원본 데이터·scaler는 수정하지 않는다. 새 출력만 허용한다.

향후 명시 승인과 Slurm 할당이 갖춰진 뒤, 같은 계산 노드 셸에서 실행할 명령:

```bash
./ai/drone-ai train --profile YOUR_PROFILE history19 fit --manifest /absolute/new-dataset/manifest.json --output /absolute/new-attempt --approve-training
```

이 명령은 새 할당을 제출하지 않는다. 기존 runtime checker가 확인하는
`drone-ai-fit` 이름의 p02/DIS01 GPU1개 작업 안에서 실행해야 한다. 외부 실행기는 기존
pending/running 작업과 공유 잠금을 확인하고 해당 작업만 제출·반납해야 한다.
위 명령의 문서화는 이번 세션에서의 본 학습 실행을 뜻하지 않는다.

## 로컬 검증

```bash
PYTHONPATH=ai:ai/src:Data/generation ai/.venv/bin/python -B -m pytest -q ai/tests/test_history19.py
```

합성 관측/NPZ만 사용한다. 모델8종 shape/gradient/초기화, 60초 대기 제거, watermark
분할 공급 동등성, 미래 변경 인과성, 입력/누출 거절, train scaler 검산, 각 모델의 작은
CPU 학습→validation→저장→별도 프로세스 복원/재개→export를 검증한다.
실제 SITL live/C9 재연결, GPU, 실제 새 데이터, 탐지 성능은 별도 미검증 항목이다.
# P27 예비 인수 보완 (2026-11-07)

예비45run은 `normal_test`/`anomaly_test`, `cohort=preliminary`로 보관한다.
학습용 loader는 독립 normal_train/normal_validation 및 기존 정상 train-only scaler를
요구하므로 예비 자료를 중복 role로 넣어 통과시키지 않는다. 신규 read-only 검산은
`./ai/drone-ai train history19 inspect-run --manifest ONLINE_RUN_JSON --producer-binding FINAL_BINDING_JSON`이다.
per-run schema는 `drone-ai.history19.run/1`이며 기존 X/Y/identity NPZ를 읽고 scaler를 요구하지 않는다.
실제 자료에는 외부에서 최종 동결한 expected producer와 per-run selection/transition 증거가 필요하다.

온라인 NPZ의 단위 표시는 `nameplate_transformed_common19_cellV_Crate`다.
`arrived_samples(engine, mission)`에 실제 execution mission~landing 경계와 clock을 전달하면
history100과 미래 target10 전체가 임무 안에 있는 창만 내보내며 제외 사유를 반환한다.
원래 canonical frames/windows는 수정하지 않는다. mission 인자를 생략한 경로는 전체
airborne 발행 결과이며 새 mission-only export라고 부르면 안 된다.
`load(..., expected_producer=...)`의 정상 train 고유 frame 통계 검산은8192행 chunk를 사용한다.
이를 새 scaler 적합으로 사용하지 않는다. GPU fit runner에 expected producer를 전달하는
본학습 연결은 별도 승인 시 필요하며 이번 준비에서 실제 학습을 실행하지 않았다.

cohort·metadata·품질 보고 요구는
`ai/reports/p27-precollection-20261107/attempt-01/intake-requirements.json` 및 REPORT.md를 따른다.
