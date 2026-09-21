# 공통 인수인계 계약 — DRONE-RAW 2.0 / 평가·추적 확장 2.1

이 계약은 v4.2 시뮬레이션과 AI 사이의 구현 기준이다. 원시 관측·watermark schema는 2.0을 유지하고, evaluation event는 선택적 필드를 추가한 2.1을 사용한다. AI의 forecast/decision 출력은 의미가 바뀌므로 별도 3.0 schema다. 변경이 필요하면 스키마 버전, 예제, 생산자·소비자 시험과 변환 도구를 같은 변경 단위에서 갱신한다. 두 담당자는 계약을 공동으로 관리한다.

## C0. 현재 적용 범위

이번 계약의 필수 생산자·소비자는 **X500 기준 기체 한 구성의 시뮬레이션과 AI**다. 여러 정상 환경·독립 비행·건강/열화 조건은 같은 X500 구성 안에서 생성한다. 파운데이션 모델과 다양한 실기체 커스터마이징은 장기 목표이며, 회사 기체·다기체 사전학습·미세조정·다른 topology 지원은 현재 인수 조건이 아니다. 원시 관측/평가/예측 schema 버전은 이번 범위 정리로 변경하지 않는다.

AI의 현재 기본 평가는 `x500_run_holdout`이다. 같은 X500 profile의 독립 run을 raw/paired/retry 계보로 분할하고, 같은 로그를 자르거나 이어받은 상태를 독립 시험으로 섞지 않는다. 같은 자산의 실제 연속 이력은 `same_asset_future`로 시간 순서를 보존한다. 새 기체 일반화 시험은 필수가 아니다.

기존 프로파일·모듈 경계만 유지하고 사용하지 않을 범용 adapter·미세조정 framework를 새로 만들지 않는다. 현재 범위 밖 기능·실기체 검증은 `not_in_scope`로 구분하며 X500 필수 실행의 미완료 항목으로 합산하지 않는다.

## C1. 책임과 데이터 흐름

시뮬레이션은 PX4 실행, 물리 계산, 센서 모사, 원시 관측 기록, 실험 조건과 정답 기록을 담당한다.
AI는 원본 파싱, 단위·좌표 변환, 시간 동기화, 품질 판정, 특징/미래 target 생성, 정상 예측 학습, 보정, forecast 발행·target 대기·경보·비행 간 집계·평가를 담당한다. 기본 학습 과제는 forecast다.
배터리 실험 자료에서 물리 파라미터를 추정하는 작업은 시뮬레이션 모델 개발에 속한다.
시뮬레이션 산출물의 의미는 센서 관측과 제어기 상태 수준으로 유지한다. AI의 재표본화 주기와 특징 수는 AI 프로파일에서 정한다.

관측 저장소와 평가 정답 저장소를 분리하고, 학습 특징은 관측 스키마에 선언된 필드에서 생성한다. 평가 도구는 저장된 예측에 정답을 결합한다.

## C2. 실행 산출물

```text
runs/<run_id>/
  run.json                        # 실행 정체성·버전·종료 상태
  observations/
    metadata.json                 # 시간·좌표·센서·명령 의미
    streams.jsonl                 # 원시 관측 이벤트
    watermarks.jsonl              # 수신 완료 진행점
    diagnostics.jsonl             # 누락·과부하·기록 상태
  reference/
    resolved_vehicle.yaml         # 명판·형상·알려진 부품 정보
    source_lock.json              # 실제 실행한 버전·commit·패키지
  evaluation/
    scenario.json                 # 정상 적격성·paired_case_id·환경 설정
    truth.jsonl                   # 물리 내부 상태
    events.jsonl                  # 실제 적용된 고장·열화 사건
  native_logs/                    # 확보한 ULog 등
  integrity.json                  # 종료 후 파일별 SHA-256과 크기
```

관측 metadata에는 운용 시 알 수 있는 기체·센서 설명을 넣는다. 실제 열화 용량, 고장 강도, 완전한 내부 SOC 같은 실험 정답은 evaluation에 저장한다. 실제 운용 중 얻는 추정 SOC는 별도의 `estimated` 관측으로 정의할 수 있다.

## C3. 원시 이벤트

다음 JSON은 하나의 배터리 센서 샘플 예제다. 숫자와 run ID는 계약 시험용 예시다.

```json
{
  "schema": "drone.raw.event/2.0",
  "run_id": "example-run-001",
  "source": "px4.uorb",
  "topic": "battery_status",
  "instance": 0,
  "seq": 42,
  "sample_time_us": 12000000,
  "available_time_us": 12005000,
  "clock_id": "run-monotonic",
  "fields": {
    "voltage_v": 15.2,
    "current_a": 12.4,
    "temperature_c": null
  },
  "field_quality": {
    "voltage_v": "valid",
    "current_a": "valid",
    "temperature_c": "unavailable"
  }
}
```

필수 의미:

| 필드 | 구현할 의미 |
|---|---|
| `run_id` | 실행마다 유일한 ID. 재시도는 새 ID와 `parent_attempt_id`로 연결 |
| `source/topic/instance` | 원천·메시지·센서 인스턴스 식별 |
| `seq` | 스트림별 증가 순번. 누락 판정에 사용 |
| `sample_time_us` | 값이 의미하는 측정 시각, 정수 마이크로초 |
| `available_time_us` | 해당 관측을 소비자가 사용할 수 있었던 시각, 같은 정규화 시간축 |
| `clock_id` | 시간축 식별. 리셋 시 새 clock/segment로 구분 |
| `fields` | 원본 필드명과 단위가 설명된 값. 미확보 수치는 JSON null |
| `field_quality` | valid / unavailable / invalid / stale / clipped |

기록 자료에서 수신 시각을 복원할 수 있는 경우 `capture_replay`로 표시한다. BIN처럼 그 정보가 없는 경우 `source_replay`로 표시하고, 원본 순서·측정 시각을 이용한 재생임을 결과에 기록한다. 후자의 end-to-end 통신 지연은 별도 측정 항목으로 둔다.

JSON의 수치는 유한값 또는 null로 직렬화한다. 동일 topic의 배열 길이, 값 단위, 의미, 필드 추가 규칙은 JSON Schema와 topic 사전에 명시한다.

## C4. metadata와 명령 의미

다음 구조를 실제 센서와 기체로 완성한다.

```yaml
schema: drone.observation.metadata/2.0
run_id: example-run-001
clock:
  id: run-monotonic
  unit: us
  mode: capture_replay
  epoch: run_start
frames:
  body: FRD
  navigation: NED
  quaternion: wxyz_body_to_navigation
vehicle:
  profile_id: x500_reference
  rotor_count: 4
  motor_order: [front_right, front_left, rear_right, rear_left]
  nominal_series_cells: 4
  nominal_capacity_ah: 6.0
actuation:
  semantic: esc_request
  encoding: normalized_0_1
  stage: post_output_mapping
  mapping_id: example-output-map-v1
sources:
  battery_status:
    instance: 0
    units: {voltage_v: V, current_a: A, temperature_c: degC}
    current_sign: discharge_positive
    expected_period_us: 100000
```

`esc_request`는 이 프로젝트가 정의한 출력 단계의 명령이다. 실제 RPM, 추력 설정, PWM, DShot 명령은 각각 의미·단위를 가진 별도 필드로 정의한다. source profile은 원천 명령을 공통 입력으로 옮길 때 적용한 변환을 기록한다.

단위·좌표의 기본은 SI, body FRD, navigation NED다. 원본이 다른 좌표·단위일 때 metadata에 원본을 기록하고 AI 변환기가 명시적으로 변환한다. 변환된 값을 다시 변환하는 상황은 `conversion_history`와 시험으로 검출한다.

## C5. 진행점과 결측

기록자는 관측을 flush한 뒤 해당 범위의 watermark를 발행한다. watermark는 `available_time_us <= through_available_time_us`인 이벤트 기록이 완료되었다는 뜻이다.

```json
{
  "schema": "drone.raw.watermark/2.0",
  "run_id": "example-run-001",
  "clock_id": "run-monotonic",
  "through_available_time_us": 12010000,
  "recorded_events": 418,
  "closed": false
}
```

실제 스트림 누락은 diagnostics에 누락 순번·시각 범위와 원인을 남긴다. AI는 사용 가능한 자료와 품질 기준으로 결정한다. 기록 종료 시 최종 watermark, 종료 원인, 파일 해시를 확정한다. 부분 기록도 종료 상태와 함께 읽을 수 있게 구현한다.

## C6. 정답 사건과 그룹

```json
{
  "schema": "drone.evaluation.event/2.1",
  "run_id": "example-run-001",
  "event_id": "fault-001",
  "kind": "motor_friction_change",
  "onset_kind": "in_flight_step",
  "physical_onset_time_us": 60001000,
  "target": "front_right",
  "requested_time_us": 60000000,
  "applied_time_us": 60001000,
  "end_time_us": 120000000,
  "profile_revision": 2,
  "parameters": {"friction_multiplier": 1.4}
}
```

고장 명령의 요청 시각과 실제 물리 계산에 적용된 시각을 분리한다. 계속되는 사건의 종료 시각은 null로 저장하고 실행 종료 시각을 평가 구간에 사용한다.
정상/이상 쌍은 `paired_case_id`, 같은 가상 기체는 `vehicle_instance_id`, 같은 배터리는 `battery_instance_id`를 가진다. 모터/프로펠러/ESC assembly instance와 교체 계보도 기록한다. AI가 독립적인 분할을 만들 수 있도록 이 관계를 dataset manifest로 전달하되 feature tensor에는 넣지 않는다. same_asset_future는 별도 시간 순 분할 protocol로 선언한다.
관측만으로 판정할 수 있는 운용 상태와, 시나리오가 알고 있는 정상 적격성은 각각 출처를 기록한다.

## C7. 변경과 호환성

스키마는 JSON Schema로 구현하고, 의미 변경은 major, 호환 가능한 확장은 minor로 기록한다. CLI는 읽은 schema와 지원 버전을 출력하고 지원 범위에 맞는 변환을 실행한다.
동일한 의미의 샘플을 두 구현에서 해석한 결과를 golden fixture로 비교한다. 공통 계약의 콘텐츠 해시와 producer/consumer 버전을 각 실행에 기록한다.
사용자가 후속 새 기체·센서 지원을 별도로 요청한 경우에만 관련 양쪽 구현·예제·source map·계약 시험을 함께 갱신한다. 이번에는 X500에 필요한 필드와 의미를 검증한다. 프로파일은 실제 사용하는 신호 목록과 품질 기준을 선언한다.

## C8. 인수인계 fixture

Codex는 최소 다음 fixture를 실제 생성한다.

| fixture | 확인할 사항 |
|---|---|
| 정상 관측 90초 | 전체 핵심 신호, 초기 워밍업, 정상 창 생성 |
| 수신 지연·순서 역전 | available time에 맞는 인과 처리 |
| 센서 누락·긴 공백 | 데이터 품질 상태와 segment 분리 |
| clock reset·instance 변경 | 상태 초기화와 새 구간 |
| 부분 종료·마지막 불완전 행 | 이미 완성된 이벤트의 복구 |
| 물리 이상 + 별도 정답 | 관측 예측 후 사건별 평가 결합 |

fixture의 `origin`은 `synthetic_contract_fixture`로 표시한다. 실제 PX4 비행 fixture는 `px4_sitl`로 표시한다. 각각의 시험 결과를 따로 보고한다.

## C9. 연결 완료 기준

시뮬레이션이 생성한 실제 PX4 실행 자료 한 개를 AI가 파싱한다. AI가 자기 전처리로 프레임과 창을 만들고, 같은 이벤트를 파일 재생과 실시간 수신으로 처리한 결과를 허용오차 안에서 일치시킨다.
두 파트의 개발 순서는 각자의 작은 fixture에서 시작하고 이 연결 시험으로 수렴한다. 공통 계약이 수정될 때마다 이 연결 시험을 다시 실행한다.

## C10. 파일·실시간 인수인계의 첫 실행 경로

첫 통합 경로는 실행 디렉터리의 `observations/streams.jsonl`과 `observations/watermarks.jsonl`을 함께 읽는 `drone_raw_directory_tail`로 구현한다. 시뮬레이션은 기록 중인 실행 디렉터리를 공개하고, AI는 같은 파일을 진행점까지 따라 읽는다. 이 경로를 file replay와 live 비교 시험의 공통 출발점으로 사용한다.

```yaml
schema: drone.source.profile/2.0
id: sitl_directory_tail
kind: drone_raw_directory_tail
run_directory: runs/EXAMPLE
streams_file: observations/streams.jsonl
watermarks_file: observations/watermarks.jsonl
diagnostics_file: observations/diagnostics.jsonl
metadata_file: observations/metadata.json
poll_interval_ms: 20
mapping_profile: configs/sources/px4.yaml
```

`run_directory`는 CLI 실행 디렉터리를 기준으로 해석하고, 내부 파일 경로는 해당 run 디렉터리를 기준으로 해석한다. `mapping_profile`은 AI가 실제 PX4 release와 센서 의미를 확인해 생성한다. 두 CLI가 같은 실제 경로를 resolve했는지 연결 시험에서 출력한다.

생산자는 newline까지 완성한 JSON 이벤트를 기록하고 flush한 뒤 그 이벤트를 포함하는 watermark를 확정한다. 소비자는 완성된 행을 읽고, 각 source에서 확보한 이벤트를 `available_time_us`와 결정적인 tie-break 규칙으로 정렬하여 해당 watermark까지 처리한다. 마지막 미완성 행은 이어지는 bytes를 기다리는 버퍼로 관리한다. `recorded_events`와 실제 누적 행 수를 대조하여 파일 가시성 지연과 자료 누락을 구분한다.

watcher의 wall-clock 대기는 파일이 준비되는 시점을 기다리는 용도다. 관측의 시간 의미는 이벤트에 기록된 sample/available time으로 계산한다. 실제 통신 지연을 평가하는 실행에서는 수신단의 timestamp와 clock 변환 근거를 함께 수집한다.

시뮬레이션 종료 시 마지막 관측 flush, `closed: true` watermark, 최종 run 상태와 integrity 파일을 순서대로 확정한다. AI는 최종 watermark까지 처리한 뒤 결과와 자체 종료 상태를 저장한다. 중단 복구는 run/clock ID, 파일 identity, byte offset, 처리한 sequence를 비교해 이어가거나 새로운 segment를 시작한다.

```bash
# 시뮬레이션이 반환한 실제 run 경로를 source profile에 반영한다.
capstone-sim run --vehicle x500_reference --scenario hover_normal --seed 17
# 별도 프로세스에서 같은 run을 따라 읽는다.
capstone-ai live --bundle artifacts/BUNDLE --source configs/sources/sitl_tail.yaml
# 종료된 동일 run을 같은 bundle로 다시 처리한다.
capstone-ai replay --bundle artifacts/BUNDLE --input runs/EXAMPLE
```

별도 socket·DDS·MAVLink transport를 채택할 때는 동일한 이벤트·clock·watermark 의미를 제공하는 adapter와 연결 시험을 추가한다. transport 선택과 경로 설정은 실제 배포 환경에 맞춰 resolve한다.

## C11. 관측 인수인계의 완결성

producer는 `observations/metadata.json`의 `semantic_bindings`에 각 관측 역할을 실제 source/topic/instance/field로 연결한다. binding에는 원본 단위·frame·명령 단계·예상 주기·가용성 근거를 담는다. consumer는 binding과 자신의 source profile을 대조해 capability를 판정한다. 이 표는 의미 사전이며, 실제 native topic 이름과 기록 주기는 선택한 PX4 소스와 실행 기록에서 확정한다.

| 관측 역할 | producer가 연결할 의미 | consumer가 확인할 사항 |
|---|---|---|
| motor_request | 위치별 실제 사용 출력 요청 | 채널 순서, 출력 함수, 정규화 단계 |
| body_rate | FRD 실제 각속도 | 단위, timestamp, estimator/센서 경로 |
| body_rate_target | 제어기가 생성한 각속도 목표 | 모드별 가용성과 측정값 대비 시각 |
| attitude | body→navigation 자세 | quaternion 순서, 좌표, 정상화 |
| specific_force | body 특정힘 | 축, 단위, 중력 처리 의미, instance |
| climb_rate | 실제 수직 속도 | 위쪽 양수 변환과 원본 단위 |
| climb_rate_target | 실제 제어기 수직 속도 목표 | 비행 모드·가용성·시간 의미 |
| pack_voltage_current | 팩 전압·DC 전류 | 배터리 ID, 방전 부호, 유효값 |
| operating_state | arm·mode·landed 등의 실제 상태 | 운용 gate를 구성할 필드와 갱신 시각 |

가용한 native 로그와 live export의 관측 역할을 각각 보고한다. 현재 X500의 기본은 `common19_ctx6`다. 실제 목표값이 부족하면 `common15_ctx6`를 선택하고, `ctx0` 비교를 수행한 경우 별도 bundle로 관리한다. 모든 대체/비교 profile을 필수로 학습·배포하지 않는다. 기본 AI 입력에는 개별 RPM·개별 모터 전류·모터/배터리 온도·개별 셀 전압·BAT.Res가 없으며, 확장은 별도 요청/센서 검증으로 관리한다. 팩 전압/직렬 셀 수는 평균 환산값이며 개별 셀 센서가 아니다.

`run.json`은 `run_id`, `parent_attempt_id`, 시작·종료 시각과 clock 정의, 실행 status, 종료 원인, source lock, observation metadata 경로, 확보한 native 로그 목록을 포함한다. 실제 시작·종료가 확인된 값으로 상태를 갱신하고 부분 실행의 산출물도 해당 상태로 인수인계한다.

최종 인수인계 보고서에는 필수 역할별 실제 생산·소비 시험 결과, 정상/이상 자료의 origin, 누락·지연·gap 요약, 실제 읽은 run 경로와 producer/consumer 버전을 기록한다.

## C12. 완료 보고의 증거 연결

양쪽 구현은 요구사항별로 구현 파일, 실행 명령, 실행 시점의 commit과 환경, exit code, 표준 출력·오류 기록, 생성 artifact와 hash, 판정 결과를 연결한다. 완료 보고 직전에 해당 범위의 검증을 다시 실행하고 그 결과로 상태를 확정한다.

상태는 `implemented`, `verified`, `needs_environment`, `needs_data`, `failed`처럼 실제 증거를 설명하는 값으로 기록한다. 수치 시험, 실제 SITL 연결, AI smoke, 연구 학습, 독립 평가, 실제 기체 검증을 각각 구분한다. 현재 범위 안에서 추가 환경이나 자료가 필요한 항목에는 필요한 조건과 다음 실행 명령을 함께 제시한다. 다른 기체 적용·미세조정·실기체 검증 등 후속 항목은 `not_in_scope`로 별도 표시하며 현재 필수 요구의 실패/미완료와 구분한다.

검증 결과 파일과 요구사항 대응표를 인수인계에 포함하고, 생산자·소비자 변경 후에는 C9의 실제 연결 시험을 다시 실행한다.



## C13. 건강 적격성·기존 열화·asset 계보

정상 학습 후보는 모든 주요 부품이 건강한 것으로 확인된 `confirmed_healthy`다. `uncertain`과 `abnormal`은 기본 fit/validation/calibration에서 제외한다. 근거와 normal_range_revision, 임무/환경 범위·실행 결과를 scenario/dataset manifest에 저장한다. 비행 성공 자체를 건강 근거의 전부로 삼지 않는다.

```json
{
  "schema": "drone.evaluation.event/2.1",
  "run_id": "preexisting-example-001",
  "event_id": "battery-resistance-existing",
  "kind": "battery_resistance_increase",
  "target": "battery_pack",
  "onset_kind": "pre_existing",
  "physical_onset_time_us": null,
  "requested_time_us": 0,
  "applied_time_us": 0,
  "observation_start_time_us": 0,
  "end_time_us": null,
  "parameters": {"r0_multiplier": 1.4},
  "profile_revision": 1,
  "origin": "synthetic_contract_fixture"
}
```

위 1.4는 schema 예시이며 검증된 검출 한계가 아니다. `applied_time_us=0`은 실행 초기의 설정 적용이다. `physical_onset_time_us=null`은 실제 노화가 시작된 시각을 알 수 없다는 뜻이다. 첫 유효 판정 시각은 AI가 입력/target·quality·warmup으로 계산하며 simulator truth로 정하지 않는다.

2.0 event를 읽는 adapter는 실제 근거가 있을 때만 onset_kind를 변환한다. 정보가 없으면 `unknown`으로 남기며 무조건 ‘비행 중 발생’ 또는 ‘처음부터 열화’로 추정하지 않는다. 2.1 선택 필드를 모르는 기존 도구와의 호환은 실제 schema 시험으로 확인한다. writer는 제공할 event version을 선언한다.

운용에서 실제 확보한 battery/propulsion asset ID·장착 위치·정비/교체 이력은 reference의 별도 asset registry로 전달할 수 있다. 이는 trend 연결용이다. 실제 물리 건강 파라미터·고장종류·건강 정도·가상 수명 단계·수치 초기 SOC는 evaluation에만 둔다. 식별자·경로명·scenario 이름에 담긴 health 정보를 neural feature로 인코딩하지 않는다.

## C14. 미래 예측의 정보·시각 계약

첫 기준은 10Hz, history 100시점, horizon 10시점, 1초 발행 간격이다. 시점별 common19/common15와 ctx6 문맥을 구분한다. ctx6는 최근 60초 통계와 segment 누적량이므로 ‘10초만 관측’이 아니다. 추가 장기 문맥이 없는 ctx0는 별도 profile로 평가한다. 인과 필터의 state가 담는 이전 정보까지 엄밀히 10초로 제한한 것은 아니므로 필터 정책도 기록한다.

forecast origin=t에서 sample/available이 모두 t 이하인 입력으로 미래 t+0.1~t+1.0초의 측정 각속도 3개와 팩 전압을 예측한다. 미래 실측 전류·명령·target은 forward 입력이 아니다. train label Y는 같은 정상 비행의 이후 값으로 생성한다.

forecast는 발행 즉시 불변 저장하고, 기본 live에서는 생성 완료가 첫 target 시각보다 빨라야 한다. 이후 실측이 모두 도착한 뒤 score/decision을 저장한다. 발행 시각·origin·target 시각·마지막 target available·실제 decision 시각을 분리한다. score를 origin으로 소급하거나 미래 실측을 알고 forecast를 재생성하지 않는다.

pre_existing 사건은 관측 시작/첫 유효 판정으로부터의 인지시간을 보고한다. 실제 최초 노화 onset을 모르면 onset 지연은 null이다. 정상 범위 점수는 건강 인증이 아니며 영역별 점수도 고장 원인 확정·SOH·RUL을 뜻하지 않는다. 런타임 예측과 evaluation 정답 결합은 파일/모듈 경계로 분리한다.

## C15. v4 추가 인수인계 fixture

| fixture | 검증할 계약 |
|---|---|
| 정상 임무/부하/잔량 다양성 | 건강 범위와 정상 환경 편차 구분, 노출시간 기록 |
| 배터리 저항 증가 pre_existing | 초기 적용과 물리 노화 onset 구분, 중간 급변 없이 관측 제공 |
| 용량만 감소 pre_existing | Q와 R 독립 설정, 충분한 방전 구간과 초기 조건 |
| 추진 손실 pre_existing | 건강한 배터리, 명령 보상·전력·운동의 연결 |
| 시점이 있는 step/ramp | 요청/실제 적용/열화 경계 시각과 지연 |
| 미래 target 대기·누락·재시작 | 미리 저장한 forecast만 사용, null/late/coverage·중복 방지 |
| truth 내용 교체 | 선택 split 고정 시 feature·forecast 불변 |
| asset 교체/비교 조건 부족 | 잘못된 장기 추적 연결 차단 |

문서 fixture, 실제 PX4 SITL, 공개자료 보정, 실제 기체 검증을 각각 다른 증거 수준으로 보고한다. 단순 severity sweep은 실제 비행 횟수별 열화율 검증이 아니다. AI 점수로 PX4 안전 제어를 우회하지 않는다.
