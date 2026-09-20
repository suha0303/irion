# Codex 구현 명세 02 — 비행 로그 전처리·정상학습 AI 제품

문서 버전: 4.2 · 작성일: 2026-10-13  
작업 결과: 원시 로그에서 학습·보정·실시간 추론·비교 평가까지 실행되는 AI 소프트웨어  
짝 문서: `01_SIMULATION_CODEX_SPEC.md`  
공통 자료: `00_SKILL_LINKS.md`, `00_SHARED_DATA_CONTRACT.md`, `00_SOURCE_NOTES.md`, `AGENTS.md`

## 0. 현재 범위와 장기 목표

<!-- CURRENT-SCOPE-BEGIN -->
**현재 단계의 개발·학습·평가 대상은 X500 기준 기체 한 구성으로 한정한다.** 하나의 정상 비행만 사용한다는 뜻은 아니다. 같은 X500 기준 구성에서 건강한 모터·배터리·프로펠러·ESC·센서로 다양한 허용 기동·바람·탑재량·잔량·온도 조건의 정상 비행을 생성한다.

**이번 산출물은 X500 전용 시계열 예측·이상탐지 모델과 실행 가능한 전체 흐름이다.** 시뮬레이션 → 원시 기록 → 전처리 → LSTM·Transformer 미래 예측 학습 → 경보 → 독립 평가를 연결한다. 배터리·추진계통의 기존 열화 상태와 비행 중 이상을 구분해 평가하며, 같은 X500/부품의 비행 간 집계는 기존 설계 범위에서 유지한다.

**다양한 실기체에 커스터마이징 가능한 파운데이션 모델은 장기 목표다.** 다기체 데이터 수집·공통 사전학습·가중치 이전·기체별 미세조정·회사 기체 적용·기종 간 일반화 검증은 현재 필수 구현 및 완료 조건에서 제외한다. 다른 기체·모터 수를 지원하는 범용 모델이나 확장 프레임워크를 미리 구현하지 않는다.

X500의 기체·센서 설정, 전처리, 모델 코드는 기존의 단순한 경계로 분리한다. 후속 확장을 위한 빈 adapter·추상 계층·미세조정 도구를 추가하지 않는다. 새 기체 지원은 사용자의 별도 요청 후 범위와 검증 기준을 정한다.

현재의 필수 실행 검증은 X500 PX4 SITL 기준이다. 실측 X500 검증과 다른 실기체 검증은 자료·장비가 필요한 별도 단계로 표시하며, 시뮬레이션 결과를 실기체 성능이나 파운데이션 모델 완성으로 보고하지 않는다. 현재 범위 밖 작업은 `not_in_scope`로 분리하고 미완료 필수 항목에 합산하지 않는다.
<!-- CURRENT-SCOPE-END -->

## 1. Codex 작업 요청

이 문서를 실제 파서·전처리·학습·추론·평가 코드, 설정, CLI, 시험, 사용 문서로 구현한다. 사용자가 파일을 넣고 명령을 실행하면 각 단계의 산출물과 상태가 나오는 제품을 완성한다.

**전처리 전체는 이 AI 파트가 구현한다.** 시뮬레이션은 원시 관측을 전달하고, 이 파트가 단위·좌표 변환, 시간 정렬, 결측 처리, 특징·창 생성, 정규화와 데이터 분할을 책임진다.

PX4는 전체 프로젝트의 비행제어기로 유지한다. AI의 프레임워크·라이브러리·폴더·특징·모델 크기·입력 주기는 현재 목적과 실제 시험 결과에 맞게 선택한다. 이 문서의 숫자와 구조는 재현 가능한 첫 실험을 위한 출발 설정이다.

### 1.1 유지하는 제품 목적과 v4의 학습 과제

건강한 모터·배터리·프로펠러·ESC·센서를 갖춘 X500 기준 기체의 다양한 정상 운용에서 **명령·전기 부하·기체 반응의 시간적 관계**를 학습한다. 기본 연구 과제는 `task: forecast`이며, LSTM 예측기와 Transformer 예측기를 같은 조건에서 비교한다 [A11].

| 단계 | 배우거나 계산하는 것 | 최종 해석 |
|---|---|---|
| 정상 예측 학습 | 과거 X와 문맥 C에서 이후 실제 센서값 Y를 예측 | 건강한 학습 분포의 조건부 반응 패턴 |
| 실시간 이상 판정 | 미리 저장한 예측과 이후 도착한 실측의 오차 | 정상 반응과 다른 징후, 위치·지속·근거 |
| 비행별 누적 평가 | 동일 부품·유사 조건의 점수와 반응 지표를 비교 | 반복·악화 여부와 열화 의심 |
| 별도 후속 과제 | 건강상태/고장종류/수명 정답을 이용한 검증 | 검증한 범위의 SOH·원인 분류·RUL |

1차 출력은 `propulsion_score`와 `battery_score`다. 전자는 추진·제어 반응, 후자는 부하에 대한 배터리 반응의 이상 정도를 뜻한다. 어느 점수가 높다는 사실만으로 특정 모터·ESC·배터리가 근본 원인이라고 확정하지 않는다. 영역별 근거를 제공하되 복합/미구분을 허용한다.

열화가 수백 회에 걸쳐 쌓였더라도 이미 변한 부품의 반응은 짧은 구간에 나타날 수 있다. 그러나 **관측 가능한 차이가 있어야 검출할 수 있으며, 짧은 창·예측오차만으로 모든 열화를 검출한다고 보장하지 않는다.** 짧은 배터리 응답으로 건강을 추정한 연구도 건강상태 기준과 시험 조건을 사용한다 [A14].

예측기의 수학적 결과는 건강한 자료에서 학습한 통계적 예측이지, 열화된 입력을 주어도 자동으로 생성되는 ‘완벽한 건강 기체의 반사실적 정답’이 아니다. 열화된 추세를 그대로 잘 예측하여 오차가 작을 수 있다. 따라서 예측 정확도와 별도로 **처음부터 열화된 상태의 독립 시험**을 필수로 수행한다. 오차가 작다는 이유만으로 건강을 인증하지 않는다.

실제 X500 측정 자료는 현재 확보되지 않은 상태다. X500 시뮬레이션 검출 결과와 공개 배터리 시험의 근거를 구분한다. 실제 X500 또는 회사 기체의 실측 성능 검증은 후속 자료/장비 확보 단계이며 이번 필수 완료 조건이 아니다. PX4 안전 제어는 AI 경보와 독립적으로 유지한다. 이번 단계에서 ‘그 밖의 모든 이상’의 포괄적 검출이나 실기체 안전 인증을 주장하지 않는다.

### 1.2 최우선 적용 스킬

<!-- PRIORITY-SKILLS-BEGIN -->
**모든 설계·구현·수정·검증·완료 보고에서 아래 네 원문을 최우선 개발 지침으로 적용한다.**

| 스킬·지침 | 원문 링크 |
|---|---|
| attention-span | https://github.com/alexgreensh/attention-span |
| andrej-karpathy-skills | https://github.com/multica-ai/andrej-karpathy-skills |
| anti-slop | https://github.com/dmmulroy/anti-slop |
| verification-before-completion | https://github.com/obra/superpowers/blob/main/skills/verification-before-completion/SKILL.md |

각 작업 시작 시 네 링크의 원문과 원문이 안내하는 적용 파일을 읽고, 현재 작업에 적용한다. 원문이 명시한 언어·도구·작업 범위에 따라 적용 대상을 정하고, 실제 확인한 경로·리비전 또는 조회 시점과 적용 결과를 기록한다.
<!-- PRIORITY-SKILLS-END -->

현재 프로젝트의 작업 지침을 이 원문 링크와 제품 목표에 맞춰 갱신한다. 전면 재설계가 필요한 부분은 요청한 결과와 연결해 교체하고, 각 구현 단계의 실제 검증 결과를 기록한다.

## 2. 출발 구성과 구조

### 2.1 기술 선택

| 역할 | 초기 선택 | 선택 근거 |
|---|---|---|
| 언어 | 지원되는 Python 3.x | 로그·시계열·학습 도구와 배포 편의 |
| 학습 | PyTorch | LSTM·Transformer를 같은 실행기로 비교 [A5–A7] |
| 파싱 | pyulog·DRONE-RAW reader | X500 PX4 입력을 우선 구현. 기존 pymavlink DataFlash는 후속 참고/선택 경로 [A3,A4] |
| 수치·자료 | NumPy, SciPy, Arrow/Parquet | 시계열 처리와 재사용 가능한 중간 자료 |
| 타입·검증 | typed dataclass/Pydantic, JSON Schema | source·feature·bundle 계약 검사 |
| 배포 모델 | safetensors + JSON/NPZ 설정 | 가중치와 해석 가능한 metadata 분리 |
| 실행 | CPU smoke, 가용 CUDA 학습 | 현재 하드웨어에 맞춘 기능·성능 검증 |

실제 설치·import·작은 학습을 성공시킨 패키지 조합을 exact version/hash로 lock한다. 이후 변경은 환경·정확도·재현 시험을 함께 기록한다.

### 2.2 권장 구조

```text
ai/
  pyproject.toml
  src/
    sources/                     # X500 raw stream·ULog·live; BIN은 후속/기존 선택 경로
    mapping/                     # 단위·좌표·명령 의미
    preprocessing/               # clock·filter·quality·frame·window
    datasets/                    # 분할·저장·로딩
    models/                      # LSTM·Transformer·baseline
    training/                    # 공통 trainer·checkpoint
    inference/                   # score·calibration·alarm·bundle
    evaluation/                  # 사건·오경보·지연·보고서
    cli.py
  configs/                       # source·feature·experiment·deployment
  tests/                         # unit·parser·causality·integration
  docs/                          # 사용법·data dictionary·model card
contracts/                       # 시뮬레이션과 공동 관리
```

모듈 크기는 실제 구현량에 맞춰 조정한다. 파일 replay와 live 처리는 같은 전처리·추론 객체를 사용한다. AI 설치 환경에서 원시 파일과 bundle만으로 처리할 수 있도록 simulator 의존성을 경계 adapter에 분리한다.

## 3. 원본 자료 감사와 source profile

### 3.1 지원 입력

현재 필수 입력은 X500 실행에서 생성한 DRONE-RAW 2.0 JSONL과 PX4 ULog다. live는 공통 계약 C10의 directory-tail 경로를 구현하고 같은 자료의 replay와 대조한다. 별도의 transport 여러 종류를 새로 만드는 요구가 아니다.
ArduPilot DataFlash BIN·회사 로그 파서는 후속/선택 범위다. 기존 구현과 기존 감사 결과는 보존할 수 있지만, 새 파서 개발·회사 BIN 분석·기체 이전 완료를 X500 구현의 선행조건으로 삼지 않는다. 아래 DataFlash 단위 설명과 대응표는 해당 경로를 사용할 때만 적용하는 참고다.

원자료에 대해 file hash, parser version, firmware metadata, 메시지·인스턴스, 필드, 단위, 실제 기록 간격, 결측·상수값·시각 공백·reset을 보고한다. 실제 필드와 의미로 source profile을 확정한다.

정상과 이상 라벨의 출처, 비행 이벤트, 센서의 실제 장착 여부를 분리한다. 과거 문서에 적힌 필드 목록은 확인 후보로 사용하고, 현재 파일의 실제 스키마로 최종 판정한다.

### 3.2 단위와 parser scaling

DataFlash parser가 적용한 스케일과 원본 format 정의를 필드별로 기록한다. decoder 출력 단위에서 SI로 한 번 변환하는 테스트를 만든다.
예를 들어 raw centidegree 값이 decoder에서 degree로 변환된 경우, 이후 단계는 degree→rad 변환을 수행한다. 자료의 숫자 크기만으로 단위를 확정하는 대신 firmware 스키마·FMT/FMTU/UNIT/MULT와 decoder 동작을 확인한다 [A2,A3].

로그에 의미 정보가 부족하면 `profile_incomplete`와 필요한 항목을 보고하고 audit 결과를 제공한다. 확인된 필드만 사용하는 capability profile을 생성할 수 있게 한다.

### 3.3 X500 PX4 mapping과 후속 ArduPilot 참고

현재는 아래 PX4 후보를 선택한 X500 firmware의 소스·실제 로그와 대조해 mapping을 완성한다 [A1,A4]. ArduPilot 열은 후속 참고이며 현재 동등 동작·이전 성능 검증을 요구하지 않는다 [A2,A3].

| 공통 의미 | X500 PX4 후보 — 현재 | ArduPilot 후보 — 후속 참고 |
|---|---|---|
| 모터별 출력 요청 | 실제 post-output-map exporter/출력 topic | RCOU.Cn + SERVO/MOTOR 함수 배치 |
| 실제 각속도 | vehicle_angular_velocity.xyz | RATE.R/P/Y |
| 목표 각속도 | vehicle_rates_setpoint | RATE.RDes/PDes/YDes |
| 자세 | vehicle_attitude.q | ATT.Roll/Pitch/Yaw |
| body 특정힘 | vehicle_acceleration.xyz 또는 의미가 맞는 센서 | IMU.AccX/Y/Z, 선택 instance |
| 실제 수직 속도 | vehicle_local_position.vz | CTUN.CRt 등 |
| 제어기 수직 속도 목표 | vehicle_local_position_setpoint.vz | CTUN.DCRt 등 |
| 배터리 전압·전류 | battery_status.voltage_v/current_a | BAT.Volt/Curr |
| arm·mode·운용 상태 | vehicle_status 등 | ARM/MODE/EV 등 |

PX4 actuator control, 출력 driver 값, 실제 ESC 요청의 처리 단계를 구분한다. PWM은 출력 범위로 정규화하고, 그 의미와 response mapping을 profile에 남긴다. 현재 검증은 선택한 X500 PX4 명령 경로를 대상으로 한다. 후속 ArduPilot 지원 시에는 정규화만으로 명령의 물리 반응이 같아진다고 가정하지 않고 분포 차이를 별도로 평가한다.
후속 DataFlash 경로를 실제로 사용할 때 CTUN 수직 속도와 RATE 각속도의 단위는 실제 decoder 기준으로 확정한다. 같은 이름의 메시지라도 firmware별 의미가 달라질 수 있으므로 source profile에 firmware 적용 범위를 기록한다.

## 4. 관측 capability와 특징 프로파일

### 4.1 공통 기준 프로파일

첫 기준은 아래 `common19_ctx6`로 구현한다. 네 모터가 FR·FL·RR·RL 위치 순서로 정리된 quad를 예로 든다. 실제 행렬 크기는 feature registry에서 계산한다.

| 이름 | 개수 | canonical 의미 |
|---|---:|---|
| motor_request_* | 4 | 모터 위치별 정규화된 출력 요청 |
| body_rate_p/q/r | 3 | body FRD 실제 각속도 rad/s |
| body_rate_error_p/q/r | 3 | 목표−실제 각속도 rad/s |
| roll/pitch | 2 | 자세 rad |
| specific_force_x/y/z | 3 | body FRD 특정힘 m/s² |
| climb_rate | 1 | 상승 양수 m/s |
| climb_rate_error | 1 | 목표−실제 상승률 m/s |
| pack_voltage | 1 | 팩 전압 V |
| pack_current | 1 | 방전 양수 A |
| 합계 | 19 | 첫 quad 기준 |

현재 기본은 X500의 `common19_ctx6`다. 실제 X500 source audit에서 목표값이 없어 필요한 경우에만 rate error 세 개와 climb error를 뺀 `common15_ctx6`를 대체 profile로 사용한다. 서로 다른 profile을 사용한 경우 입력 사전·bundle·평가 결과를 구분한다. 두 profile을 모두 학습·배포하는 것이 필수는 아니다.
기본 입력에서 개별 ESC RPM·개별 모터 전류·모터 온도·배터리 온도·개별 셀 전압·`BAT.Res`는 제외한다. `IMU.T`·`BARO.Temp`·MCU 온도를 배터리 온도 대신 넣지 않는다. PX4/BIN 경보·fault flag는 기본 연속 학습 입력으로 사용하지 않는다. arm·mode 등의 관측 상태는 운용 gate로 사용한다.

해당 센서를 실제로 확보한 후 별도 요청·검증이 있는 경우에만 확장 profile을 추가한다. 시뮬레이터가 계산하는 rotor RPM·온도·실제 SOC·열화 파라미터를 공통 입력에 몰래 추가하지 않는다. 원본으로 계산 가능한 특징과 추가 센서가 필요한 특징을 registry에서 구분한다.

`common19_ctx6`는 시점별 19개 관측 변환과 창 끝의 6개 문맥을 별도 tensor로 제공한다. `common15_ctx6`는 실제 목표값이 부족한 X500 자료의 대체 경로이며 둘을 조용히 혼용하지 않는다. 문맥 없는 단기 비교를 선택하면 `common19_ctx0`/`common15_ctx0`라는 별도 profile과 bundle로 실행한다. `ctx0`는 선택 비교이며 기본 완료의 선행조건이 아니다. 첫 운영 기준은 `common19_ctx6`를 유지한다.

10Hz는 현재 확보한 공통 관측의 시작 sampling profile이다. 고주파 진동·베어링 결함을 이 데이터로 정밀 진단한다고 보장하지 않는다. 원신호에서 이미 소실된 대역은 업샘플링으로 복원할 수 없고, 진동 연구는 별도의 고속 관측/사전 요약 capability로 검증한다 [A16].

이번 지원 topology는 X500 quad-X의 모터 4개다. 다른 모터 수·새 기체를 같은 가중치로 처리하는 기능은 구현하지 않는다. model card에는 `supported_vehicle_profile: x500_reference`와 실제 profile revision·시험 범위를 기록한다. 서로 다른 feature profile을 비교한 경우 이름과 정보량을 함께 보고한다.

### 4.2 모터 위치·좌표

모터 번호는 source의 출력 함수와 기체 geometry를 연결해 위치·회전 방향 기준으로 매핑한다. forward/right/down, north/east/down과 quaternion convention을 하나의 좌표 모듈로 구현한다.
roll/pitch는 실제 attitude 표현에서 계산하고, 각도 경계와 quaternion 정규화를 시험한다. 가속도는 중력 포함 여부와 특정힘 의미를 source profile에 기록한다. 센서가 사용하는 시간·좌표 의미가 맞는지 stationary·hover·회전 fixture로 확인한다.

### 4.3 배터리 명판과 메타데이터

직렬 셀 수, 정격 용량, 정격 셀 전압은 기체·배터리 profile에서 제공한다. 실제 열화 용량과 simulator truth는 evaluation 자료로 관리한다.
모델 입력에 사용할 메타데이터는 실제 deployment에서도 확보 가능한 항목을 feature contract에 명시한다. 해당 정보가 없는 입력은 요구 정보 또는 그 정보 없이 학습한 다른 profile을 안내한다.

## 5. 시간 정렬과 인과 처리

### 5.1 하나의 전처리 엔진

```python
class Preprocessor:
    def ingest(self, event: RawEvent) -> None: ...
    def advance(self, watermark_us: int) -> list[CanonicalFrame]: ...
    def snapshot(self) -> PreprocessState: ...
    def restore(self, state: PreprocessState) -> None: ...
    def reset(self, reason: str) -> None: ...
```

file replay와 live adapter는 같은 event·watermark 인터페이스를 사용한다. 오프라인 학습용 자료도 이 엔진에서 생성한다.
판정 시각 t에서는 `sample_time <= t`와 `available_time <= t`를 만족하는 관측을 사용한다. watermark가 보장하는 수신 완료 범위까지 처리한다. 이미 발행된 예측은 그때 사용 가능했던 관측에 대한 기록으로 보존한다.

BIN처럼 availability를 복원하기 어려운 자료는 `source_replay`로 표시한다. 이 자료의 지연 지표는 source time 기준 탐지 지연으로 보고하고, 실제 전송·컴퓨팅 지연 측정과 나눈다.

### 5.2 10Hz 출발 grid

첫 sampling profile은 정수 마이크로초 기준 100000us 간격이다. clock origin과 grid anchor를 metadata로 저장한다. 다른 sampling profile은 지원하는 원신호 기록률과 anti-alias 시험으로 정한다.
동일 timestamp는 source/instance/seq의 결정적 규칙으로 처리한다. 순서 역전 이벤트는 profile에 정의한 허용 lateness와 sample age에 따라 처리하고 진단을 남긴다.

### 5.3 인과 필터

명령값은 zero-order hold로 처리한다. 연속 센서는 원신호 cadence·대역폭에 맞는 causal low-pass를 선택하고 필터 계수와 군지연을 기록한다.
불규칙 sample용 출발 구현으로 두 개의 동일한 1차 low-pass를 연속 연결한 상태 모델을 사용할 수 있다. 구간 동안 입력 x가 일정할 때 정확한 상태 업데이트는 다음과 같다.

```text
a = exp(-dt/tau)
y1_next = a*y1 + (1-a)*x
y2_next = a*y2 + (dt/tau)*a*y1 + [1-a-(dt/tau)*a]*x
```

직전 측정값을 구간 입력으로 사용하고 새 sample 시각에서 입력을 교체한다. grid 시각의 출력 조회는 필터 snapshot을 진행해 계산한다. 늦게 들어온 sample은 이후 판정에 영향을 주는 방식으로 처리하고, 실제 적용/폐기 이유를 diagnostic에 남긴다.
높은 차수 필터가 필요한 경우 SOS 등의 안정 구현을 선택한다. downsampling 전에 감쇠 특성을 확인하고 aliasing·위상 지연·step 응답을 시험한다. 정확한 cutoff와 적용 대상은 data audit 결과로 정한다.

### 5.4 결측·quality·segment

source cadence에서 sample age 한계를 정한다. 초기 후보는 빠른 제어·IMU 0.2s, 배터리 0.35s지만 실제 기록률에 맞게 설정한다. 데이터 품질을 `valid`, `stale`, `unavailable`, `invalid`, `clipped`로 전달한다.
첫 신경망 profile은 필요한 특징이 유효한 창에서 점수를 계산한다. 결측을 다루는 모델을 추가하면 mask와 missingness 학습·평가를 같은 feature profile에 포함한다.
긴 공백, clock reset, 주센서 교체, run 전환은 새로운 segment를 만든다. 전처리·문맥·경보 상태의 reset 범위와 warmup을 명시한다.

운용 gate는 arm·airborne·mode 등 실제 관측 가능한 신호로 구성한다. 지상·비행·데이터 부족 상태를 출력한다. 실제 fault와 함께 발생한 failsafe·착륙은 evaluation에서 사건과 coverage를 함께 기록한다.

### 5.5 canonical frame

```json
{
  "schema": "drone.canonical.frame/2.0",
  "run_id": "example-run-001",
  "segment_id": "segment-001",
  "time_us": 62000000,
  "feature_profile": "common19_ctx6",
  "values": {"pack_voltage": 15.1, "pack_current": 12.0},
  "quality": {"pack_voltage": "valid", "pack_current": "valid"},
  "operating_state": "airborne",
  "source_clock_mode": "capture_replay"
}
```

이 예제는 배터리 필드 일부만 보여준다. 실제 선택한 profile의 모든 필드를 schema로 검사한다. 원신호부터 canonical 값까지 source lineage와 변환 이력을 추적할 수 있게 한다.

## 6. 모델 특징·문맥·창

### 6.1 관측 기반 변환

canonical SI 자료를 보존하고, 모델 입력 변환을 feature registry로 정의한다. 시작 profile은 팩 전압을 직렬 셀 수로 나눈 **셀당 평균 환산 전압**과, 전류를 명판 정격 용량으로 나눈 **정격 용량 기준 C-rate**로 변환한다. 평균 환산 전압은 개별 셀 센서값이 아니며 셀 불균형을 알려주지 않는다. 실제 열화 용량으로 나누어 열화 효과를 상쇄하거나 정답을 유출하지 않는다.
가용한 관측에서 소비 전력, 기동 수준, 명령 불균형, 변화율을 계산할 수 있게 하되, 각 실제 실험에서는 사용한 특징 목록과 파라미터를 명시한다. LSTM과 Transformer는 같은 목록을 사용한다.

### 6.2 6개 문맥 특징

첫 기준 문맥은 다음처럼 정의한다. 필드명 `cell_voltage`는 모두 팩 전압/직렬 셀 수의 평균 환산값이다. 이 문맥들은 과거 관측의 요약이며 실제 SOC·SOH가 아니다.

| 문맥 | 계산 |
|---|---|
| mean_cell_voltage_60s | 최근 60s 팩 전압/Ns 환산값의 평균 |
| cell_voltage_slope_60s | 최근 60s 시각과 전압의 least-squares 기울기 |
| mean_c_rate_60s | 최근 60s C-rate 평균 |
| std_c_rate_60s | 최근 60s C-rate 표준편차 |
| observed_charge_fraction | 관측된 순 전류 적분 Ah / 정격 Ah |
| observed_energy_fraction | 관측 전력 적분 Wh / 정격 Wh |

누적량은 segment 시작부터 확보한 유효 관측의 signed integral로 정의한다. discharge-only profile의 센서 noise가 만든 작은 역전류는 명시한 센서 정책으로 처리한다. 실제 충전 운용은 signed current의 source 의미를 유지한다.
이 값은 기록 구간에서 관측한 사용량이며, 배터리의 정확한 남은 SOC와는 다른 특징으로 이름을 붙인다. 공백을 만났을 때 누적 coverage와 reset 정책을 기록한다.

전류·전력 적분은 과거 값 hold와 실제 dt를 사용하고 최대 유효 gap을 적용한다. 동일한 자료에 대한 해석식·직접 계산과 대조한다.
최근 60초에 필요한 valid coverage가 확보된 뒤 해당 profile의 점수를 발행한다. 조기 판정이 필요하면 짧은 문맥 profile을 따로 학습·보정한다.

### 6.3 입력 창·예측 구간·출력 시각

초기 기준은 10Hz, history 100시점, 미래 horizon 10시점, forecast 발행 간격 1초다. 연속 100시점의 첫·마지막 간격은 9.9초이며 ‘10초급 history’로 표기한다. 미래 target은 origin 다음 0.1초부터 1.0초까지다. 이 값은 검증 전 출발값이지 최적값이나 검출 가능성 보장이 아니다.

```text
Δ = 0.1 s, T = 100, H = 10
origin = t                     # 입력 정보 cutoff와 grid 시각
X_t = X[t-99Δ : t]              # [T,D], 끝점 포함
C_t = context(as_of=t)          # [C], 기본 C=6
Y_t = targets[t+Δ : t+10Δ]      # [H,K], X와 겹치지 않음
Y_hat_t = model(X_t, C_t)
```

학습 정답은 같은 정상 비행에서 실제로 이어진 센서값이다. 고장종류 라벨이 없어도 예측용 정답 Y는 존재한다. 단, 정상 학습 적격성·데이터 분할·평가 사건은 manifest로 관리한다.

#### 6.3.1 입력과 target을 구분한다

`target_profile: response4_v1`의 첫 target 순서는 `[body_rate_p, body_rate_q, body_rate_r, pack_voltage]`다. 측정 각속도 3개가 추진 반응 group, 팩 전압 1개가 배터리 반응 group이다. 학습 시에는 registry의 관측 변환 및 train scaler를 target에도 같은 방식으로 적용하고, 출력은 역변환하여 rad/s·V로 저장한다.

입력에 전류·모터 명령·자세 등이 있다고 모두 예측 target이 되는 것은 아니다. 현재 전류는 부하를 설명하는 입력이며 `response4_v1`에서는 전류를 예측하지 않는다. 특정힘·상승률·전류 등을 출력에 더하는 실험은 target profile·loss·오차 scale·경보 보정을 함께 바꾸어 별도 비교한다. 4개 target만으로 모든 추진 성능 저하를 관측할 수 있다고 가정하지 않는다.

#### 6.3.2 실제로 보는 시간과 지연

`ctx6`는 최근 60초 요약과 segment 시작부터의 전류/전력 적분을 사용한다. 따라서 기본 실험은 **‘10초 history + 60초 통계 + segment 누적량’**이며, 과거 정보 전체가 10초로 제한된 모델이 아니다. 60초 문맥 coverage를 채워야 첫 forecast가 가능하다. 추가 장기 문맥의 효과는 `ctx0` 비교에서 따로 평가한다. `ctx0`에도 인과 필터의 내부 상태는 남으므로 엄밀한 ‘원시 정보가 10초로만 제한된 실험’과 동일하지 않다. 그런 제한 실험은 필터 상태/유효 support·초기화·warmup을 별도로 정의한다.

forecast를 t에서 발행한 뒤 t+1초의 마지막 실측까지 도착해야 해당 블록 점수를 계산한다. 예를 들어 문맥이 t=60초에 준비되면 첫 블록 점수는 대략 t=61초 이후에 나온다. 필터·수신·계산 지연과 경보 지속 조건은 여기에 별도로 더해진다. ‘1초 후를 예측한다’는 말은 ‘고장을 1초 전에 경고한다’는 뜻이 아니다.

window는 같은 run·segment·feature profile·split에서 생성한다. history, 필요한 context/누적 상태, target의 전체 계보를 저장한다. 초기 기준의 출력 블록은 발행 간격과 horizon이 같아 연속 origin 간 target 시각이 겹치지 않는다. 더 자주 발행하는 확장은 한 실측에 대한 여러 예측의 선택·집계 규칙을 미리 정한다. 사후에 가장 잘 맞는 예측을 선택하지 않는다.

### 6.4 정규화

정상 train split에서 per-feature scaler를 fit하고 dataset·scaler hash를 저장한다. val/calibration/test/live에는 해당 학습 scaler를 적용한다.
Scaler fit 시 중첩창으로 특정 frame을 여러 번 세는 효과를 제어한다. 기본은 고유 frame 단위 또는 문서화한 flight weighting이다. 거의 상수인 특징은 scale floor와 진단으로 처리한다.
X500 정상 기준을 다시 보정하는 경우 별도 calibration revision으로 저장하고 기존 기준과 비교한다. 비행별/윈도별 평균·표준편차로 매번 다시 정규화하여 장기 전압 강하나 명령 보상 차이를 없애지 않는다. 그러한 변환은 원 신호 보존과 영향 평가를 갖춘 별도 실험으로만 사용한다.

운영 중 scaler·정상 오차 기준·threshold·가중치를 무감독으로 계속 적응시키는 기능은 기본값에서 끈다. 이미 열화된 상태를 새 정상으로 흡수할 수 있기 때문이다. 업데이트는 확인된 건강 구간과 명시적인 revision으로 수행하고 이전 기준과 병렬 비교한다.

## 7. 데이터 분할과 정상학습 연구

### 7.1 그룹부터 분리

윈도 생성 전에 원시 비행/run·부분 파일·재시도·동일 paired case의 관계를 모아 독립 그룹부터 분할한다. 한 정상 원시 비행의 일부를 train, 다른 부분을 test에 무작위 배치하지 않는다. 같은 시뮬레이션의 재생·잘라낸 로그·다른 포맷 변환본도 같은 그룹이다 [A12,A13].

현재 기본은 **같은 X500 기준 구성의 독립 비행 평가**다. 다수의 기체·배터리 개체를 확보해야만 학습·평가할 수 있는 요구를 두지 않는다.

| protocol | 분할 | 주장 가능한 범위 |
|---|---|---|
| `x500_run_holdout` 현재 기본 | 같은 X500 기준 구성에서 독립 초기화/seed의 run을 생성하고 raw/run/paired/retry 계보로 그룹 분할 | 같은 X500 구성·선언한 운용 범위에서 보지 않은 비행의 성능 |
| `same_asset_future` 같은 X500의 누적 평가 | 동일 부품의 확인된 건강한 과거 비행으로 fit/calibration하고 이후 비행을 시간 순으로 잠금 | 등록된 같은 개체의 향후 변화; 새 기체 일반화 아님 |

같은 `x500_reference` profile 또는 같은 명판 사양을 공유한다는 이유로 모든 독립 run을 하나의 그룹으로 묶지 않는다. 기본 평가에서 같은 기체·부품 정체성이 split에 걸쳐 존재하는 것은 허용하지만 새 개체로의 일반화를 주장하지 않는다. 동일 원시 기록, 반복 replay, 재시도, 정상/이상 paired case, 이어받은 simulator 상태를 독립 표본처럼 나누지 않는다.

실제 연속 사용·정비/열화 이력이 있는 같은 자산은 `same_asset_future`로 시간 순서를 유지하고 미래 기록을 이전 fit에 넣지 않는다. 과거 비행을 쓰는 실험과 새 개체를 쓰는 실험은 별개의 질문이다. `unseen_asset`이나 기종 전체 holdout은 후속 선택 평가이며 이번 필수 완료 조건이 아니다.

첫 비율 후보는 train 60%, validation 15%, calibration 15%, test 10%다. 정상/이상 paired case가 final test에 배정되면 모두 해당 평가 그룹에 유지한다. 그룹 수와 노출시간을 먼저 보고하고 비율 자체를 성공 기준으로 쓰지 않는다. 독립 run 그룹이 부족하면 `insufficient_independent_groups`를 보고하고 같은 로그를 잘라 평가용 그룹을 만들지 않는다. 후속 공개 셀/RPT·실기체 자료를 사용하는 경우 그 원자료와 자산 계보도 보존한다.

context·필터·누적량·target까지 split 경계를 넘지 않게 생성한다. 장기 평가의 여러 비행을 연결하는 것은 사후 trend 계층이며 단기 window를 비행 경계 너머로 연결하는 것이 아니다. 관측/환경 조건과 평가 질문을 잠근 뒤 모델·임계값을 선택한다.

### 7.2 정상 데이터의 역할

| 자료 | 역할 |
|---|---|
| 정상 train | scaler·모델 가중치 fit |
| 정상 validation | architecture·early stopping·오차 기준 선택 |
| 정상 calibration | 경보 임계값·지속 조건 보정 |
| 독립 정상 test | 새로운 정상 조건의 오경보 |
| 독립 이상 test | 사건 탐지·지연·coverage 평가 |

정상 적격성은 `health_eligibility: confirmed_healthy | uncertain | abnormal`과 근거로 관리한다. 기본 train/validation/calibration에는 `confirmed_healthy`만 들어간다. 정상은 단순히 ‘비행 성공·추락 없음’이 아니며 모터·배터리뿐 아니라 프로펠러·ESC·센서·제어 설정도 연구 목적상 건강한 기준이어야 한다. 후속 회사 BIN을 확보하더라도 전체가 정상이라고 가정하지 않으며, 현재 학습 자료에는 자동 편입하지 않는다.

| 정상 자료의 다양성 | 포함·관리 기준 |
|---|---|
| 비행 동작 | 같은 X500의 호버·상승/하강·전진/회전·가감속·목표 운용 임무 |
| 부하·환경 | 허용된 바람·탑재량·온도·기동의 조합, 조건별 노출시간 |
| 방전 진행 | 안전한 운용 범위의 높은 잔량부터 낮은 잔량까지 |
| 건강 편차 | 확인된 제작 편차/건강 범위만 포함, 검출 대상 열화와 분리 |
| 기체 설정 | X500 기준 형상·모터/배터리 규격·펌웨어/제어 설정과 revision을 고정. 선언한 탑재량·건강 편차만 조건으로 변화 |

이 표의 온도·바람·실제 초기 SOC는 시뮬레이션 조건/evaluation metadata일 수 있다. 다양성을 만들었다는 이유로 실제 기체에서 없는 센서값을 모델 입력으로 추가하지 않는다. 온도 등 미관측 교란요인 때문에 열화와 구별하기 어려운 조건은 진단 한계로 평가한다 [A15].

이상 범위까지 정상 저항/효율 분포를 넓혀 섞으면 검출할 차이를 정상으로 학습할 수 있다. 정상과 열화 경계를 수치 범위·근거로 사전 정의하고, 정상의 불리한 환경 조건과 열화가 겹치는 구간도 시험한다. 정상 제작 편차와 열화 세기는 독립적으로 설계한다.

모델 구조·가중치·scaler는 정상 train/validation에서, 경보 기준은 별도 정상 calibration에서 확정한다. 고장 종류/강도 정답은 기본 모델 선택·fit·calibration에 넣지 않는다. 고장 개발 세트를 보고 설계를 바꾸었다면 그 세트는 `anomaly_dev`로 표시하고, 새로 잠근 독립 test에서 최종 평가한다. 기존 test를 재사용하면서 손대지 않은 독립 성능이라고 보고하지 않는다.
loader의 feature 경로는 관측과 deployment에서 알려진 metadata의 allowlist로 구성하고, evaluation label은 evaluator가 별도 입력으로 읽는다. 정상 적격성으로 학습 run을 선택하는 과정과 고장/열화 정답을 tensor에 넣는 과정을 구분한다. manifest의 선택 결과를 고정한 상태에서 truth를 바꾸어도 feature·scaler·forecast가 같음을 시험한다.

### 7.3 저장 형식

원본 → canonical Parquet → window index → 학습 batch의 단계를 구현한다. 중간 산출물에는 source hash, parser·mapping·filter version, feature order, 시간 범위, split group, quality summary를 기록한다.
대규모 데이터는 메모리 제한과 batch 읽기를 지원하고, 완료된 shard와 manifest를 원자적으로 확정한다. 재시작 시 hash와 파이프라인 version을 확인해 이어서 생성한다.
학습 manifest는 실제 사용한 run·시간 범위·window 수·제외 사유·노출 시간을 포함한다.

## 8. LSTM 미래 예측기

### 8.1 출발 architecture

```text
history [B,T,D]
  → LSTM encoder (2 layers, hidden 128, dropout 0.1)
  → final hidden → latent 64
context [B,C] → concatenate with latent → MLP → fused latent 64
fused latent repeated H times + future lead-time embedding
  → LSTM decoder (2 layers, hidden 128)
  → propulsion head (3) + battery head (1)
  → predicted targets [B,H,4]
```

`C=0` profile은 context 입력 없이 같은 역할의 encoder/decoder를 사용한다. 모든 shape·target 순서는 registry/config에서 결정한다. 출력 head는 target profile에 지정한 미래 센서값만 예측한다.

### 8.2 decoder 입력과 학습

decoder에는 latent와 horizon 위치 표현만 전달한다. 미래 실측 target·미래 실측 전류·미래 모터 명령을 teacher forcing 입력으로 주지 않는다. loss 계산용 Y는 별도 label tensor이며 forward 입력이 아니다. 마지막 관측 이후에 생기는 명령 변화는 예측 불확실성의 원인이며 13절에서 조건별 오탐을 평가한다.

dropout은 train에서만 적용하고 eval에서 끈다. v3의 denoising noise=0.02를 새 forecast 기본값에 그대로 복사하지 않는다. forecast의 입력 noise 기본값은 0이며 noise ablation은 별도 실험으로 실행한다. 모델 크기 선택은 정상 validation의 예측 손실·계산 비용을 사용하되, 그 선택이 이상 검출 최적임을 의미하지 않는다고 보고한다.

## 9. Transformer 미래 예측기

### 9.1 출발 architecture

```text
history [B,T,D] → linear embedding 128 + positional encoding
  → Transformer encoder (2 layers, 4 heads, FFN 256, dropout 0.1)
  → memory containing only history through origin
context [B,C] → embedding added as a separate context token when C>0
H learned/positional forecast query tokens (no future measurements)
  → Transformer decoder (2 layers, 4 heads, FFN 256)
       cross-attends only to history/context memory
  → propulsion head (3) + battery head (1)
  → predicted targets [B,H,4]
```

첫 구현은 전체 horizon을 한 번에 출력한다. query끼리의 attention에는 미래 정답이 없으므로 미래 target 누출이 아니다. 과거 memory 내부의 양방향 attention도 origin까지 확보한 값만 포함하면 온라인 인과성을 깨지 않는다. causal mask를 무조건 사용해야 한다는 주장 대신 실제 정보의 sample/available cutoff와 target 입력 차단을 검사한다.

각 층 초기화, layer norm, GELU, tensor shape, padding/mask 정책은 선택한 PyTorch release로 확인한다 [A5,A6]. 상태 없는 window 추론을 기본으로 하며 window 밖 hidden cache를 쓰는 확장은 실제 기억 범위를 별도 기록한다.

### 9.2 공정한 비교

LSTM과 Transformer는 같은 feature/target/scaler/split/loss/발행 cadence/calibration/evaluation을 사용한다. 파라미터 수·연산량 추정·peak memory·추론 지연도 함께 보고한다. 정상 validation 튜닝 횟수와 선택 규칙을 맞춘다.

예측기별로 정상 보정한 경보를 같은 정상 오탐 목표·같은 시험·실제 지연/coverage 기준으로 평가한다.

## 10. 공통 학습기와 baseline

### 10.1 forecast loss

train scaler로 변환한 target에 대해 group별 평균을 계산한다.

```text
L_prop = mean over batch, horizon, p/q/r of squared prediction error
L_batt = mean over batch, horizon, pack_voltage of squared prediction error
L = 0.50 * L_prop + 0.50 * L_batt
```

가중치는 초기 실험 설정이다. context와 현재 입력 명령은 조건 입력이며, loss는 지정한 미래 target에 대해서만 계산한다. target 추가/삭제 시 group 멤버·손실 가중치·scoring contract를 명시적으로 갱신한다.

기본 학습은 history와 모든 target이 유효한 window를 사용한다. 결측 확장은 mask·유효 분모·완전 결측 group 정책을 선언하고, 결측 target을 정상값으로 채워 정답처럼 사용하지 않는다.

### 10.2 시작 training 설정

```yaml
experiment: common19_forecast_reference_v4
vehicle_profile: x500_reference
split_protocol: x500_run_holdout
task: forecast
feature_profile: common19_ctx6
target_profile: response4_v1
sampling_hz: 10
window_samples: 100
horizon_samples: 10
target_names: [body_rate_p, body_rate_q, body_rate_r, pack_voltage]
context_seconds: 60
infer_every_seconds: 1
forecast_policy:
  future_observed_inputs: false
  teacher_forcing: false
  score_after: all_targets_available
  require_issued_before_first_target: true
training:
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.00001
  batch_size: 128
  max_epochs: 100
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  noise_std_scaled: 0.0
  seeds: [17, 29, 43]
  device: auto
loss_weights:
  propulsion: 0.5
  battery: 0.5
```

실제 메모리·데이터에 맞춰 resolve한 최종 config를 저장한다. history/context/horizon의 효과는 서로 분리하여 비교하고, 60초 문맥을 켠 결과를 ‘10초만 관측한 결과’로 표시하지 않는다. 학습 진행률·정상 validation 손실·best epoch·wall time을 기록한다.

### 10.3 checkpoint와 재현성

best/last checkpoint에 optimizer·scheduler·RNG·epoch·data order·config·source hash를 포함한다. 실행 중단 후 resume하여 이어서 학습하는 시험을 구현한다.
Python/NumPy/PyTorch seed, 실제 device·dtype·library version·deterministic 설정을 기록한다. GPU 연산의 재현 한계와 실제 반복 편차는 [A7]과 측정 결과로 설명한다.
finite loss·gradient 검사, 이상 종료 사유, OOM 시 batch 조정과 재시작 이력을 제공한다. 변경된 batch·정밀도는 새 resolved config로 남긴다.

### 10.4 간단한 forecast baseline

`persistence`는 마지막 관측 target을 H개 미래 시점에 반복한다. `ridge_forecast`는 동일 history/context를 고정된 벡터로 펼쳐 미래 target을 예측하는 정상학습 회귀 기준선이다. 전처리·split·scaler·경보 보정·evaluator를 두 신경망과 공유한다. ridge 규제 강도는 정상 validation으로 선택한다.

지속값 예측이 잘 맞는다는 사실은 건강의 증거가 아니다. 성능 저하가 일정하게 유지되는 구간에서 persistence와 신경망이 모두 잘 맞아 이상을 놓치는 반례를 보고한다. 낮은 정상 MSE만으로 신경망의 채택을 확정하지 않는다.

## 11. 이상 점수와 경보 보정

### 11.1 확정된 forecast와 실측의 오차

origin별로 불변 저장한 예측을 이후 실측과 target 시각·이름으로 매칭한다. 기본은 10개 target 시점이 모두 유효하고 도착한 뒤 한 번만 점수를 낸다. origin 뒤의 관측을 넣어 과거 forecast를 다시 계산하거나 실측을 본 뒤 교체하지 않는다.

정상 validation에서 특징 j·예측 lead h별 squared forecast error의 q95와 양의 floor로 `reference_error_scale[j,h]`를 정한다. horizon이 길어질수록 오차가 달라질 수 있으므로 lead를 보존한다. 학습/validation 밖 자료로 이 scale을 다시 fit하지 않는다.

```text
e[j,h] = (actual_scaled[j,h] - forecast_scaled[j,h])^2
z[j,h] = e[j,h] / max(q95_normal_validation[j,h], positive_floor[j,h])
propulsion_score = mean(z[j,h] for j in [p,q,r], all h)
battery_score = mean(z[j,h] for j in [pack_voltage], all h)
```

기본 배터리 점수에는 예측하지 않는 전류의 오차나 context 복원오차를 넣지 않는다. propulsion group과 battery group을 표시하는 것은 근거를 나누는 것이지 원인을 분리해 증명하는 것이 아니다.

실제 단위의 signed residual `actual - forecast`, 절대/제곱 오차, 항목별 기여도, 예측/실측 시계열, 입력 부하·품질도 저장한다. 음의 전압 residual은 예측보다 낮은 전압이라는 근거지만 단독으로 열화를 확정하지 않는다. 정상에서도 미래의 부하·기동·온도 등 조건에 따라 오차가 커질 수 있다 [A15].

### 11.2 상태기계

출발 경보 설정은 다음과 같다.

```yaml
alarm:
  score_ewma_alpha: 0.3
  enter_consecutive_decisions: 3
  exit_consecutive_decisions: 10
  exit_threshold_ratio: 0.7
  target_false_alarms_per_hour: 1.0
  calibration_quantiles: [0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999]
```

1초 단위 점수에 적용하는 기준이다. sampling/추론 주기를 바꾸면 시간 기반 지속 조건으로 환산하고 새 calibration을 실행한다.
`warming_up`, `awaiting_targets`, `ready`, `anomalous`, `insufficient_data`, `out_of_profile`, `forecast_late` 상태를 구현한다. `ready`는 계산 가능 상태이지 건강 인증이 아니다. score가 없으면 null과 원인을 출력하며 0 또는 정상으로 대체하지 않는다.

`out_of_profile`은 실제 관측/알려진 metadata에서 확인한 capability·운용 범위 불일치를 뜻한다. 미래 고장 정답이나 미관측 실제 온도를 읽어 판단하지 않는다. 알려지지 않은 환경 변화가 언제나 탐지될 것이라고 보장하지 않는다. 데이터 공백/late에서 경보를 자동 해제하거나 정상 지속시간을 누적하지 않고 상태를 unavailable로 분리한다. 유효 점수의 enter/exit 카운트는 gap을 가로질러 연속으로 세지 않는다. 세부 reset 정책은 시험으로 고정한다. 전압·전류·명령이 정상 학습 범위 밖이라는 이유만으로 out_of_profile 처리해 물리 이상을 평가에서 제외하지 않는다. 실제 센서 invalid/capability 불일치와 정상분포 이탈을 구분한다.

### 11.3 정상 calibration

각 q 후보에서 두 영역의 high threshold를 정상 calibration 분포로 정하고, 전체 시간 순서에 상태기계를 실행한다. 두 영역 경보의 합집합 episode를 계산하여 목표 오경보/시간을 만족하는 가장 민감한 후보를 선택한다.
유한 threshold 후보가 목표를 만족하지 못하면 실제 최선 결과와 `target_unmet` 상태를 기록한다. 노출시간이 짧으면 `provisional`과 필요한 추가 정상 시간의 근거를 표시한다.

0회 오경보도 관측 시간과 함께 보고한다. 독립 Poisson 가정하의 95% 상한 약 `2.996 / exposure_hours`는 가정이 붙은 참고치로 계산하고, 비행 단위 bootstrap 등 상관을 반영하는 불확실성 추정도 제공한다.
최종 test 결과는 calibration된 bundle 그대로 계산한다. 새로운 보정은 새 revision과 별도 결과로 보관한다.

## 12. bundle과 실시간 추론

### 12.1 독립 실행 bundle

```text
bundle/
  model.safetensors
  model.json
  feature_profile.json
  source_compatibility.json
  preprocessing.json
  forecast_contract.json         # task·origin/target 시간·target 순서·발행/매칭 규칙
  scaler.npz
  calibration.json
  training_manifest.json
  model_card.md
  integrity.json
```

bundle은 task, 필요한 입력, target profile/순서/역변환, 실제 기억 범위, 기체 metadata, 좌표·명령 의미, sample rate, warmup, 필터 상태 정책, 발행/매칭 규칙, model shape, 학습·calibration domain, 파일 hash를 포함한다. bundle의 task·target·출력 schema가 현재 forecast 계약과 일치하는지 검사하고, 불일치한 bundle은 로드하지 않는다.
로드 시 실제 source와 feature contract를 검사하고 지원 여부를 설명한다. 일치하는 adapter/profile을 선택하거나 추가 mapping이 필요한 항목을 출력한다.

### 12.2 실시간 엔진과 미래값 대기

```text
source → Preprocessor → history/context window
       → Model → immutable ForecastWriter → PendingForecastStore
source → Preprocessor → actual target frames
       → target-time matcher → Scorer → Alarm → DecisionWriter
```

입력 cutoff t에서는 `sample_time_us <= t`이면서 `available_time_us <= t`인 관측만 사용한다. 생성 완료 시각 `issued_time_us`와 origin t를 별도로 기록한다. 기본 live 유효성은 `issued_time_us < first_target_time_us`다. 늦게 계산된 블록은 `forecast_late`로 남겨 coverage/지연에 반영하고 사전 예측 성능으로 세지 않는다. 유효성 기준은 해당 target의 실제 가용성 뒤에 발행한 예측을 차단하는 것보다 보수적인 첫 기준이다.

score의 `decision_time_us`는 모든 target의 sample/available 시각 및 scorer 완료보다 이르지 않아야 한다. score를 forecast origin으로 소급 기록하지 않는다. 상대 시간 비교는 같은 정규화 run clock을 사용하고 wall time 추론 지연도 별도 기록한다. BIN의 `source_replay`에서 가정한 availability와 실제 live 지연을 혼동하지 않는다.

기본 horizon=발행 간격=1초인 경우 origin 순서로 완성된 블록을 한 번씩 채점한다. 결측·run 종료로 완성되지 않는 블록은 `unscored` 사유와 대상 시간을 보존한다. 이미 판정한 실제값/forecast를 뒤늦은 데이터로 수정하지 않는다. 정정 replay는 새 revision으로 실행한다.

checkpoint에는 전처리 filter·문맥/누적 상태·경보 state·pending forecast·마지막 처리 seq·bundle hash를 저장한다. 재시작 시 동일 run/clock/bundle과 pending identity를 검사해 복원하거나 새 segment로 warmup한다. 중복 arrival·재생에도 forecast와 점수가 중복 발행되지 않아야 한다. queue·backpressure·처리율·메모리·지연을 관측한다.

### 12.3 forecast record와 decision record

아래는 문법·시각 계약을 설명하는 예시이며 실제 모델 실행 결과가 아니다. forecast 발행과 실측 도착 후의 판정을 구분하는 출력 schema 3.0을 사용한다. 원시 관측 schema는 2.0을 유지한다.

```json
{
  "schema": "drone.forecast/3.0",
  "task": "forecast",
  "forecast_id": "run-001:120000000:bundle-001",
  "run_id": "run-001",
  "segment_id": "segment-001",
  "origin_time_us": 120000000,
  "last_input_sample_time_us": 120000000,
  "input_available_cutoff_us": 120000000,
  "issued_time_us": 120002000,
  "source_clock_mode": "capture_replay",
  "bundle_id": "bundle-001",
  "feature_profile": "common19_ctx6",
  "target_profile": "response4_v1",
  "target_names": ["body_rate_p", "body_rate_q", "body_rate_r", "pack_voltage"],
  "target_units": ["rad/s", "rad/s", "rad/s", "V"],
  "target_times_us": [120100000, 120200000, 120300000, 120400000, 120500000, 120600000, 120700000, 120800000, 120900000, 121000000],
  "predicted_values": [[0.01,0.00,0.00,15.20],[0.01,0.00,0.00,15.20],[0.01,0.00,0.00,15.19],[0.01,0.00,0.00,15.19],[0.01,0.00,0.00,15.19],[0.01,0.00,0.00,15.18],[0.01,0.00,0.00,15.18],[0.01,0.00,0.00,15.18],[0.01,0.00,0.00,15.17],[0.01,0.00,0.00,15.17]],
  "status": "awaiting_targets",
  "inference_wall_ms": 2.0
}
```

```json
{
  "schema": "drone.prediction/3.0",
  "task": "forecast_residual_detection",
  "forecast_id": "run-001:120000000:bundle-001",
  "run_id": "run-001",
  "segment_id": "segment-001",
  "origin_time_us": 120000000,
  "target_start_time_us": 120100000,
  "target_end_time_us": 121000000,
  "last_target_available_time_us": 121005000,
  "decision_time_us": 121010000,
  "source_clock_mode": "capture_replay",
  "bundle_id": "bundle-001",
  "feature_profile": "common19_ctx6",
  "target_profile": "response4_v1",
  "status": "ready",
  "scores": {"propulsion": 1.2, "battery": 0.8},
  "alarms": {"propulsion": false, "battery": false},
  "suspected_domains": [],
  "diagnosis_status": "not_established",
  "health_estimate": {"soh": null, "rul": null},
  "quality": {"matched_target_count": 10, "required_target_count": 10}
}
```

실제 record에는 항목별 오차/기여도와 해당 forecast/실측 artifact 참조를 포함한다. schema allowlist에 없는 진실값·열화율·모터 고장 번호를 예측인 것처럼 작성하지 않는다. 경보가 있어도 `diagnosis_status`는 추가 진단 근거가 없으면 `not_established`를 유지한다.

원자료·bundle·profile·calibration revision으로 재현한다. 초기 제품은 모니터링·기록·경보 전달용이며 제어기 명령을 바꾸지 않는다. 별도 안전 검증 없이 AI의 정상 판정을 PX4 failsafe 우회나 비행 허가 근거로 사용하지 않는다.

## 13. 평가와 비교 보고서

### 13.1 평가 순서

독립 test 관측에 대해 예측을 생성하고 파일 hash를 확정한다. evaluator가 별도로 읽은 사건·정상 적격성 manifest를 그 예측에 결합한다.
시각마다 실제 계산된 score와 alarm 상태를 이용한다. 평가 구간과 warmup·불충분 자료·simulator 실패의 취급을 protocol에 명시한다.

### 13.2 사건과 episode: 급성 발생과 기존 열화

alarm episode는 상태기계가 경보에 들어간 실제 decision 시각부터 종료 조건 충족 시각까지다. 영역별 및 합집합 episode를 모두 제공한다.

`in_flight_step`/`in_flight_ramp`는 실제 파라미터 적용 시각을 근거로 탐지 지연을 계산한다. 램프 시작과 사전 정의한 열화 경계 도달 시각을 구분한다. **`pre_existing`은 기록 시작 전부터 존재하는 상태**이며, 물리적인 최초 열화 발생 시각은 알 수 없을 수 있다. 설정을 t=0에서 적용했다고 수백 회 누적 열화가 t=0에 발생한 것은 아니다.

| 사건 형식 | 보고할 시간 지표 |
|---|---|
| 비행 중 새 이상 | `alarm_decision_time - applied_time`, 필요 시 경계 도달 기준도 병기 |
| 처음부터 열화 | 모니터링 대상 운용 시작→첫 경보의 인지시간, 첫 유효 판정 기회→첫 경보의 추가 인지시간 |
| 알려진 onset이 없는 실측 열화 | 최초 관측·최초 경보·검증 범위, onset 탐지 지연은 null |

`first_evaluable_decision_time`은 정상 여부가 아니라 history/context/target·품질·운용 gate로 계산한다. warmup·late·관측 소실로 판정 기회가 없었던 사건을 분모에서 숨기지 않는다. pre-existing 사건은 최초 유효 판정부터 이미 진행 중인 이상으로 평가하며 새로운 파라미터 변경 이벤트가 있어야만 매칭하도록 구현하지 않는다.

기본 매칭은 사건의 평가 구간 안에서 최초로 경보 상태가 확인된 실제 decision을 사용한다. onset 전부터 켜져 있던 경보는 `preexisting_alarm`으로 별도 표시하고 onset 이전 시각을 음의 탐지 지연으로 인정하지 않는다. 신규 경보 기준 recall과 기존 경보 포함 인지 여부를 나누어 보고한다. 다수 사건/경보의 중복 규칙과 ramp 경계는 protocol에 잠근다.

### 13.3 핵심 지표

| 지표 | 계산 기준 |
|---|---|
| 정상 오경보/시간 | 독립 정상 자료의 경보 episode 수 / 정상 유효 노출시간 |
| 사건 탐지율 | 정의한 matching 구간에서 탐지된 사건 / 전체 해당 사건 |
| 탐지/인지 지연 | 비행 중 이상은 실제 적용 기준, 기존 열화는 관측/첫 유효 판정 기준으로 분리 |
| coverage | 유효한 실제 점수로 감시한 시간 / 운용 대상 시간. warmup·late·결측·pending을 각각 보고 |
| 불충분 자료 사건 | 관측 부족으로 판정 기회가 적었던 사건 수·원인 |
| 계산 비용 | p50/p95/p99 추론 시간, 처리율, 메모리, 하드웨어 |
| 원시 시점 성능 | 실제 시점별 예측으로 계산한 PR/F1 등 보조 지표 |

전체 사건 기준 recall과 충분한 관측이 있었던 사건의 조건부 recall을 함께 보고한다. 실제 고장 때문에 관측이 사라진 상황과 수치 시뮬레이터 실패를 구분한다.
이상 크기·비행 형태·바람·SOC·기체 instance·발생 형식별 결과를 분해한다. truth의 실제 SOC/온도 등은 이 사후 분석에만 쓰고, deployment에서 모르는 값을 threshold 선택에 사용하지 않는다. 실제 고장 샘플과 등가 물리 파라미터 변화의 평가 범위를 설명한다.
평가 방법은 [A9]을 참고하고 실제 예측 시점과 episode를 그대로 집계한다.

### 13.4 통계·비교

세 seed 등의 반복 실험 결과를 평균·표준편차와 비행/그룹 단위 bootstrap 구간으로 제시한다. 중첩된 창을 독립 표본처럼 취급하는 대신 비행·기체 그룹으로 불확실성을 계산한다.
LSTM·Transformer·baseline의 입력·split·노출시간·calibration 목표·계산 budget을 한 표에 보여준다. 더 복잡한 모델의 실질적인 개선과 비용을 같이 해석한다.

### 13.5 생성할 보고서

`data_audit`, `source_mapping`, `dataset_card`, `training_report`, `calibration_report`, `comparison_report`, `model_card`, `deployment_check`를 Markdown과 machine-readable JSON으로 제공한다.
모든 그래프·표는 실제 산출물에서 생성하고 사용 데이터·run·bundle hash를 연결한다. software smoke 결과와 X500 실제 SITL 연구 학습·독립 평가 결과를 따로 표시한다. 미실행 실기체/회사 기체 평가는 후속 범위이며 현재 성과에 포함하지 않는다.

### 13.6 필수 시험 행렬과 검출 주장

모든 현재 시험군은 선택한 X500 기준 구성에서 수행한다. 여러 기종/새 부품 카탈로그를 추가하는 시험 행렬이 아니다.

| 시험군 | 조건 | 확인할 질문 |
|---|---|---|
| N1 정상 다양성 | 건강한 전체 기체, 여러 기동·바람·탑재량·잔량·온도 | 정상 조건 변화가 오탐을 만드는가? |
| N2 어려운 정상 | 저잔량·큰 정상 부하·허용 범위의 저온 등 | 단순히 낮은 전압/큰 전류를 고장으로 판단하는가? |
| F1 기존 저항 증가 | 건강한 모터 + 비행 시작부터 열화된 배터리 R0 | 급변 없이 부하 반응의 차이를 검출하는가? |
| F2 기존 용량 감소 | 건강한 모터 + 시작부터 Q_actual만 감소 | 짧은 창과 긴 방전 관측의 차이는 무엇인가? |
| F3 기존 추진 저하 | 건강한 배터리 + 시작부터 모터 손실/추진 특성 변화 | 제어 보상 후에도 관측 차이를 검출하는가? |
| F4 비행 중 이상 | 명시된 step/ramp와 실제 적용 시각 | 발생 이후 경보·필터/수신/계산 지연은 얼마인가? |
| F5 복합 이상 | 배터리와 추진 이상을 함께 적용 | 단일 원인 확정 없이 복합 징후를 표현하는가? |
| O1 정보 부족/중첩 | 약한 열화·부하 변화 부족·건강 편차와 반응 중첩 | 놓친 사례와 식별 한계를 정직하게 보고하는가? |
| Q1 관측 품질 | 지연·누락·reset·invalid | 센서/통신 품질 문제와 물리 이상을 분리하는가? |

F1/F2/F3에서 해당 파라미터는 비행 중 일정하게 두어 ‘급변 찾기’만으로 성공하지 못하게 한다. 급성 이상 성공과 기존 열화 성공은 별도 지표다. `F2`가 짧은 구간에서 분리되지 않으면 이를 보고하고 충분한 방전 구간/건강상태 기준을 사용하는 별도 capacity task를 후속 검토 항목으로 남긴다. 새로운 용량/SOH 모델의 구현은 이번 필수 완료 조건이 아니다. 검출이 어려운 사례를 숨기거나 threshold를 test에 맞춰 조정하지 않는다.

정상에서 바람/기동/저온을 넓게 다루더라도 미지 환경과 열화를 반드시 구분하는 것은 아니다. 같은 관측을 만드는 서로 다른 원인은 현재 입력만으로 유일하게 식별할 수 없다. 특정 모터 베어링·셀 불균형 등은 이번 공통 신호 시험의 보장 범위가 아니다.

### 13.7 비행 간 열화 의심 추적

같은 X500/부품의 여러 비행을 단기 경보와 별도 집계 계층으로 구현한다. 다기체 관리 시스템이나 실제 수백 회의 노화 시험을 새로 요구하지 않는다. 동일 기체 ID만으로 배터리나 모터가 동일하다고 가정하지 않는다. 운용 시 확보한 `battery_asset_id`, 모터/프로펠러/ESC assembly ID, 장착 위치, 교체·정비·펌웨어 변경 이력을 사용한다. asset ID는 연결용이며 신경망 입력 feature가 아니다. ID가 없으면 비행별 집계만 제공하고 특정 부품의 장기 추적은 `identity_unavailable`로 표시한다.

비행마다 유효 노출시간·조건 coverage·영역별 score 중앙값/상위 분위수·경보 지속비율·signed residual과 대표 부하 반응을 저장한다. 같은 bundle/scaler/calibration revision과 비교 가능한 운용 조건의 구간을 비교한다. 정상 baseline이 없는 조건을 억지로 등가 비교하지 않는다.

단순히 전체 비행 평균 score가 커졌다는 이유로 열화로 확정하지 않는다. 임무·잔량 분포·온도·바람·탑재량·센서 보정·제어기 설정이 달라졌을 수 있다. 사용할 수 있는 조건은 관측/운용 기록으로 매칭하고 미관측 조건은 불확실성으로 남긴다. 부하·전압 등의 건강 영향 지표로 지나치게 조건화해 열화 차이를 제거하는지도 비교한다.

출력은 `stable_evidence`, `repeated_deviation`, `worsening_evidence`, `insufficient_comparable_data` 등 근거 수준이며 SOH 백분율/RUL로 치환하지 않는다. 점수가 열화 세기에 단조 증가한다고 가정하지 않는다. threshold·trend 판단 조건은 건강한 비교 비행으로 사전 보정하고 실제 열화 이력이 있을 때 별도 검증한다.

```json
{
  "schema": "drone.health.trend/1.0",
  "asset_scope": "battery",
  "asset_id": "battery-asset-001",
  "bundle_id": "bundle-001",
  "calibration_revision": "cal-001",
  "run_ids": ["run-001", "run-002"],
  "comparison_status": "insufficient_comparable_data",
  "trend_value": null,
  "soh": null,
  "rul": null,
  "reason": "example_only_no_measured_trend"
}
```

시뮬레이션의 여러 고정 열화 수준은 기본적으로 ‘열화 강도 sweep’이다. 이를 비행 횟수에 따른 실제 노화 속도나 수명 곡선으로 부르지 않는다. 수백 회의 가상 비행을 생성했다는 사실만으로 물리적인 수명 모델이 검증되지 않는다.

### 13.8 forecast만으로 부족할 때의 별도 비교

pre-existing 시험에서 예측기가 열화된 추세를 그대로 따라가면 그 한계를 결과로 남긴다. 다음 비교는 사용자가 후속 실험을 선택한 경우에만 별도 task/profile로 수행한다. 기본 예측기가 놓친 결과는 그대로 보고하며, 별도 모델 추가를 이번 완료의 선행조건으로 삼지 않는다.

`conditional_response`는 새로 측정한 전류/명령과 과거 상태에서 **그 시점의** 정상 전압/운동 반응을 추정하는 진단이다. 현재 target 자체를 입력에 넣는 항등 경로를 차단한다. 같은 시점 전류를 사용했다면 미래를 미리 맞혔다고 보고하지 않는다. 예측기와 데이터 시점·label 생성·calibration을 분리한다.

관측 기반 전류 변화 대비 전압 변화, 명령 보상과 기체 반응 등의 간단한 지표도 evidence 후보로 비교할 수 있다. `-ΔV/ΔI`는 충분한 전류 변화·정렬·조건이 있을 때의 유효 반응 지표이지 자동으로 순수 R0·SOH가 아니다. 분극·SOC·온도·필터 영향과 분모가 작은 구간을 처리한다. 이는 `BAT.Res`를 기본 입력에 추가하는 변경이 아니다.

미래 실측 전류를 forecast에 넣어 검출 결과를 좋게 만든 뒤 원래 모델의 성공이라고 보고하지 않는다. 이 확장은 기본 forecast 산출물을 대체하지 않으며 실제 비교가 필요할 때 채택한다.

## 14. 장기 목표와 후속 확장 — 현재 범위 제외

장기적으로는 다양한 실기체에 커스터마이징 가능한 파운데이션 모델을 목표로 한다. **이번 결과물은 X500 전용 시계열 예측·이상탐지 모델**이며, 다기체 공통 기반 모델이 이미 완성되었다고 표현하지 않는다.

| 후속 항목 | 이번 단계의 처리 |
|---|---|
| 여러 기체의 데이터 수집·공통 사전학습 | 구현·데이터 확보·성능 입증을 요구하지 않음 |
| 회사 실기체 적용·기체별 보정/미세조정·가중치 이전 | 현재 CLI·필수 체크리스트에서 제외 |
| 다른 모터 수/센서 구성 지원·새 기종 holdout | 범용 architecture·adapter를 미리 만들지 않음 |
| 파운데이션 모델 평가·처음부터 학습 대비 적응 효율 | 별도 요청 후 데이터/지표/범위를 정의 |

이번에는 X500 profile, 실제 source mapping, 전처리, 학습 가중치, 정상 calibration, 독립 평가 결과를 구분해 저장하는 기존 구조만 유지한다. 빈 확장 클래스·다기체 모델 registry·범용 fine-tuning API는 추가하지 않는다. 후속 기능이 없어도 현재 X500 요구사항을 충족하면 이번 단계의 완료 판단이 가능하다.

같은 X500에서도 배터리/모터 교체·정비·설정 변경이 있으면 asset ID와 baseline revision 연결을 갱신한다. 이는 기존 추적 자료의 정확성을 지키는 규칙이지 회사 기체 이전 기능을 이번에 개발하라는 요구가 아니다.

## 15. 실제 제공할 CLI

아래 명령의 필수 대상은 X500다. `runs/EXAMPLE`과 ULog 파일명은 실제 생성 경로로 바꾼다. 회사 기체 이전·미세조정 CLI는 이번에 추가하지 않는다.

```bash
capstone-ai doctor --report artifacts/ai-environment.json
capstone-ai audit --input runs/EXAMPLE/native_logs/flight.ulg --out artifacts/audit
capstone-ai source validate --profile configs/sources/px4.yaml
capstone-ai preprocess --input runs/EXAMPLE --source configs/sources/px4.yaml --out data/canonical
capstone-ai dataset build --config configs/datasets/reference.yaml
capstone-ai train --model lstm --task forecast --experiment configs/experiments/reference.yaml
capstone-ai train --model transformer --task forecast --experiment configs/experiments/reference.yaml
capstone-ai train --model ridge_forecast --task forecast --experiment configs/experiments/reference.yaml
capstone-ai baseline --model persistence --experiment configs/experiments/reference.yaml
capstone-ai calibrate --bundle artifacts/BUNDLE --dataset data/calibration
capstone-ai replay --bundle artifacts/BUNDLE --input runs/EXAMPLE
capstone-ai live --bundle artifacts/BUNDLE --source configs/sources/live.yaml
capstone-ai evaluate --predictions artifacts/predictions --truth data/evaluation
capstone-ai compare --experiment artifacts/EXPERIMENT
capstone-ai trend --predictions artifacts/predictions --assets data/asset_registry.json --out artifacts/trends
capstone-ai package --bundle artifacts/BUNDLE --out artifacts/deployment
capstone-ai verify --suite unit
capstone-ai verify --suite integration
capstone-ai verify --suite end-to-end
```

CLI마다 입력 검사, 진행률, JSON 결과, artifact 경로, 종료 코드를 제공한다. `--help`와 README에는 실제 예제와 이어서 수행할 작업을 설명한다.
작은 `make-demo`로 parser·전처리·두 forecast 모델 forward/backward·미래값 대기/매칭·bundle/replay·평가 흐름을 시험한다. demo는 `synthetic_contract_fixture`이고 본 연구 데이터는 실제 `px4_sitl` 실행으로 구분한다.
학습 자료가 준비되면 실제 정상 비행 자료로 두 모델을 학습하고 가중치·보고서를 생성한다. 자원·자료가 필요한 단계는 현재 실행 상태와 다음 명령을 기록한다.

## 16. 자동 시험과 합격 기준

Python lint·타입 검사, tensor shape 검사와 serialization schema 검사를 설정하고, 변경 시 관련 단위·통합 시험과 함께 실행한다. 검사 도구의 버전과 결과를 기록한다.

### 16.1 데이터·전처리

| 시험 | 확인할 결과 |
|---|---|
| X500 PX4 ULog decoding | 실제 ULog 필드 의미·단위→SI 변환이 기대값과 일치 |
| X500 ULog/raw 단위 fixture | 같은 관측의 ULog와 DRONE-RAW에서 canonical 의미·단위 일치 |
| 모터 위치 mapping | 출력 번호 변화에도 geometry 기준 배열 일치 |
| 좌표·specific force | 정지·hover·회전의 부호와 중력 의미 일치 |
| source capability | 가용 신호에 맞는 profile와 필요한 정보 안내 |
| sequence·watermark | 누락·순서 역전·지연 처리의 결정성 |
| 필터 | step·주파수·불규칙 dt의 응답과 군지연 |
| 창·문맥 | 경계·워밍업·누적량·slope의 직접 계산 일치 |
| segment reset | clock·instance·공백에서 상태 전이 일치 |
| 품질 | 유효 노출시간·데이터 부족 상태 일치 |

### 16.2 인과성·재현성

미래 관측을 크게 바꾼 두 입력에서 이전 시각의 canonical frame·feature·prediction이 같음을 시험한다.
동일 이벤트를 파일 일괄 재생, 작은 chunk 재생, 실시간 속도 재생으로 처리하고 결과를 비교한다. availability 순서와 source clock mode를 같은 조건으로 맞춘다.
학습 run 선택과 split manifest를 고정하고 물리 truth/고장종류/세기만 바꾼 fixture에서 feature/scaler/forecast의 일치를 확인한다. 정상 적격성 변경에 따른 학습 대상 변경은 별도 membership 시험으로 다룬다. split group 관계와 history/context/target의 전체 사용 시간 구간을 검사한다.

### 16.3 실제 모델과 배포

두 모델의 shape·finite output·gradient·짧은 학습·best checkpoint·resume·save/load 일치를 시험한다. CPU에서 작은 전체 경로를 실행하고 사용 가능한 GPU에서 실제 학습·성능을 확인한다.
배포 bundle을 깨끗한 AI 환경에 로드하여 replay와 live adapter를 실행한다. 입력 incompatibility, 가중치 hash 오류, 불충분 자료를 상태와 원인으로 반환한다.

### 16.4 경보·평가

threshold entry/exit, EWMA reset, alarm 지속·중복·합집합, 정상 노출시간, 사건 matching, 실제 onset delay, coverage를 손으로 검산 가능한 fixture로 확인한다.
0회 오경보·짧은 calibration·목표 미달·전체 결측·사건 직전 시작한 alarm 등 경계 사례를 포함한다.

### 16.5 실제 연결

시뮬레이션 담당이 생성한 실제 PX4 run을 읽어 audit·전처리·window 생성·replay까지 수행한다. research campaign이 준비되면 정상 split으로 두 모델을 학습하고 독립 정상·이상 test에서 비교 보고서를 생성한다.

### 16.6 v4 추가 회귀 시험

| 시험 | 기대 결과 |
|---|---|
| target 경계 | 100 history와 10 future, 첫 target=t+0.1s, 마지막=t+1.0s |
| 시간 fixture | t에 발행한 예측은 실측이 도착하기 전에는 score 없음, 점수 시각 소급 없음 |
| 미래 입력 변경 | t 이후 전류·명령·target을 바꿔도 t의 forecast byte/허용오차 불변 |
| teacher forcing 차단 | forward 입력에 Y 및 미래 관측 tensor 없음 |
| late forecast | 첫 target 시각 이후 발행은 late로 기록되고 live 사전 예측에서 제외 |
| target 중복/미도착 | 한 번 채점, 종료/결측은 null+coverage 감소, 0점 정상 처리 금지 |
| pending 복원 | 같은 clock/bundle 재시작 시 중복 없이 동일 forecast/decision 복원 |
| profile 기억 범위 | ctx6 warmup과 segment 적분, ctx0와 구분, 미래값 없는 문맥 |
| sensor allowlist | RPM·온도·개별 셀·BAT.Res·물리 truth가 기본 feature로 들어가지 않음 |
| 기존 열화 | pre_existing 상태를 t=0 급성 사건으로 오해하지 않고 관측 기준 인지시간 계산 |
| 상태와 진단 | score 정상범위가 SOH=100%·healthy 확정으로 변환되지 않음 |
| 장기 비교 | asset 교체·bundle 변경·비교 조건 부족 시 trend 합산 차단/사유 기록 |
| 단순 예측 반례 | 일정한 열화 추세에서 persistence 오차가 작아도 건강 판정하지 않음 |
| 독립성 | 원시 비행·paired case·context/target 계보에 protocol 위반 없음 |

위 시험 중 문서의 예시 산술/스키마 검사는 구현 시험과 구분한다. 실제 PyTorch 학습·실제 SITL 연결·물리 이상 검출은 구현 저장소에서 별도로 실행해야 한다.

## 17. 완료 체크리스트와 진행 기록

아래 필수 완료 기준은 X500 구성과 실제 선택한 입력 profile만 대상으로 한다. 14절 후속 항목, 실기체 시험, 선택 profile/진단 모델을 필수 미완료로 계산하지 않는다.

- [ ] X500의 DRONE-RAW/ULog parser와 선택 profile의 단위·좌표·4-모터 mapping이 구현된다.
- [ ] 전처리가 AI 파트 안에서 file/live 공통 코드로 실행된다.
- [ ] capability·결측·segment·인과성 시험이 통과한다.
- [ ] X500 독립 run split·scaler·모델·calibration 계보와 paired/retry 누출 방지가 기록된다.
- [ ] LSTM/Transformer forecast와 두 간단한 forecast baseline의 실행·저장·복원이 확인된다.
- [ ] forecast 발행→future target 대기→실제 decision 시각의 경보가 실행된다.
- [ ] 단기 감지·60초/segment 문맥·비행 간 추적의 정보 범위가 구분된다.
- [ ] 시작부터 열화된 경우와 비행 중 이상이 분리 평가된다.
- [ ] sensor/metadata allowlist와 건강 적격성·독립 시험이 검증된다.
- [ ] asset별 추적은 비교 가능한 근거를 표시하고 SOH/RUL로 오인되지 않는다.
- [ ] bundle replay·live와 실제 상태·score·alarm 출력이 동작한다.
- [ ] 독립 test의 오경보·탐지·지연·coverage·계산 비용이 보고된다.
- [ ] 작은 smoke와 실제 연구 실행의 결과가 각각 표시된다.
- [ ] 사용 명령·환경·결과 파일·남은 검증 범위가 제공된다.

진행 순서는 `X500 자료 감사 → 계약 fixture → 전처리 → X500 dataset → 모델 → 정상 calibration → replay/live → 독립 평가·같은 X500 비행 간 집계 → 결과 패키지`다. 각 단계에서 실행과 수정을 반복한다.

상태 파일에는 코드 구현, 소프트웨어 시험, 학습 실행, calibration, 독립 평가, 실제 기체 검증의 상태를 나누어 적는다. 최종 보고는 실제 실행 결과와 artifact 경로, 재현 명령, 추가 자료가 필요한 검증을 정리한다.
참고 자료는 `00_SOURCE_NOTES.md`의 [A1–A7,A9–A16]이며, 실제 사용한 release·parser 동작·추가 논문은 구현 저장소에 함께 기록한다.

---

<!-- SHARED-CONTRACT-BEGIN -->
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
<!-- SHARED-CONTRACT-END -->

## 문서 단독 사용 시 참고 URL

[A1] PX4 메시지 문서: https://docs.px4.io/main/en/msg_docs/

[A2] ArduPilot 로그 메시지: https://ardupilot.org/copter/docs/logmessages.html

[A3] DataFlash parser: https://github.com/ArduPilot/pymavlink/blob/master/DFReader.py

[A4] ULog parser: https://github.com/PX4/pyulog

[A5] PyTorch LSTM: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html

[A6] PyTorch TransformerEncoder: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html

[A7] PyTorch 재현성: https://docs.pytorch.org/docs/stable/notes/randomness.html

[A9] 시계열 이상 탐지 평가 검토: https://arxiv.org/abs/2109.05257

[A10] Dynamics randomization 연구: https://arxiv.org/abs/1710.06537

[A11] LSTM 예측오차 기반 이상탐지 (Hundman et al., 2018): https://arxiv.org/abs/1802.04431

[A12] 데이터 누출·전처리 fit 경계 (scikit-learn): https://scikit-learn.org/stable/common_pitfalls.html

[A13] 그룹/시계열 분할 (scikit-learn): https://scikit-learn.org/stable/modules/cross_validation.html

[A14] 짧은 배터리 응답과 건강상태 기준을 사용하는 실험 연구: https://www.nature.com/articles/s41467-024-54454-0

[A15] PX4 배터리 부하 보상·전류 적분의 의미: https://docs.px4.io/main/en/config/battery

[A16] PX4 고속 IMU 기록·필터/진동 분석: https://docs.px4.io/main/en/config_mc/filter_tuning

이 문서의 architecture·방정식 조합·초기 숫자·완료 기준은 프로젝트 설계다. 실제 구현은 선택한 버전의 API·소스와 수치 시험을 대조하여 완성한다. 작업 지침은 앞부분에 기재한 네 원문 링크를 기준으로 적용한다.
