# Codex 구현 명세 01 — PX4 기반 물리 시뮬레이션 제품

문서 버전: 4.2 · 작성일: 2026-10-13  
작업 결과: 설치·빌드·비행·기록·검증까지 실행되는 시뮬레이션 소프트웨어  
짝 문서: `02_AI_PREPROCESSING_CODEX_SPEC.md`  
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

이 문서를 실제 코드·설정·설치 도구·CLI·시험·사용 문서로 구현한다. 현재 저장소와 실행 환경을 확인한 뒤, 필요한 부분을 재설계하고 끝까지 동작하는 실행 흐름을 완성한다.

**기술적으로 유지할 대상은 PX4다.** PX4 내부 코드, 시뮬레이션 연결부, 메시지와 설정은 목적에 맞게 수정한다. 나머지 코드·시뮬레이터·버전·폴더·의존성은 적합성과 검증 결과에 따라 새로 선택한다. 현재 저장소는 조사·재사용 후보로 취급한다.

사용자가 요청한 제품 목적은 다음과 같다.

- 현재 대상인 X500 기준 기체 한 구성에서 실제 비행에서 관측할 법한 일관된 시계열을 생성한다.
- 모터·프로펠러·배터리·바람·기체·PX4가 서로 영향을 주게 구현한다.
- 모든 주요 부품이 건강한 정상 비행을 다양한 허용 운용 조건에서 생성하고, 별도 이상 시나리오로 AI 평가 자료를 만든다.
- AI의 정상 미래예측과 기존 열화 검증을 위해 급성 고장뿐 아니라 **처음부터 열화된 상태**와 충분한 방전 관측을 제공한다.
- X500의 기체·부품·센서 설정을 코드와 분리한다. 다른 회사 기체의 추가·검증은 이번 구현에 포함하지 않는다.
- 원시 관측과 실험 정답을 각각 전달한다. 동기화·특징 생성·학습용 전처리는 AI 파트가 수행한다.

실제 X500 측정 자료는 현재 확보되지 않은 상태다. 공개 사양·공개 시험 자료·명시적인 공학 가정으로 기준 모델을 만들고, 정확도를 주장할 범위를 실제 증거에 맞춰 표시한다.

### 1.1 최우선 적용 스킬

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

### 1.2 재선정 가능한 출발 구성

아래는 초기 구현을 시작할 구체적 선택이다. 더 적합한 구성이 확인되면 선택 이유와 회귀 시험을 갖추어 변경한다.

| 역할 | 출발 구성 | 채택 근거와 변경 판단 |
|---|---|---|
| OS | Ubuntu 24.04 계열 개발 환경 | PX4·시뮬레이터 설치 지원과 실제 빌드 결과로 선택 |
| 제어기 | 지원되는 안정 PX4 release | 실제 사용할 release와 patch commit을 lock에 기록 |
| 실행 기반 | 현대 Gazebo, Harmonic을 첫 후보로 시험 | PX4 공식 연결, 강체·센서·환경 구현 활용 [S1–S3] |
| 실시간 물리 | 독립 C++ 계산 모듈 + 얇은 시뮬레이터 adapter | 수치 시험과 엔진 이전을 분리하기 쉬움 |
| 실행·수집 | Python CLI와 프로세스 관리자 | 반복 비행·원시 기록·실패 복구의 구현 비용 |
| 모터 | 평균 ESC + 전기·회전·열 모델 | 10Hz급 관측의 원인이 되는 물리 관계 재현 |
| 배터리 | Thévenin 1RC + 열 모델 | 부하 응답·회복을 적은 파라미터로 표현 [S5] |
| 바람 | 평균 + 시간 상관을 가진 확률 과정 | 정상 부하·제어 변동을 재현 [S3,S9] |

설치 가능한 정확한 버전으로 작은 비행을 성공시킨 뒤 재현용 lock을 만든다. 버전 lock은 검증된 실행을 재현하는 기록이다. 업데이트 시 새 lock과 비교 시험 결과를 생성한다.
더 무거운 모델·플러그인·엔진은 필요한 관측 정확도, 계산 비용, 연동 품질을 비교하여 채택한다. 필수 의존성에는 현재 구현에서 실제 사용하는 기능을 명시한다.

## 2. 첫 단계 산출물과 전체 구조

### 2.1 저장소 조사

실제 경로, Git 상태, 기존 코드와 미완료 변경, 실행 환경, 가용 CPU/GPU/RAM/디스크를 조사한다. 기존 자료와 교체할 파일의 복구 가능한 이력을 확보한다.

`docs/BASELINE_AUDIT.md`에 아래를 기록한다.

| 조사 항목 | 기록할 결과 |
|---|---|
| 실제 checkout | 저장소 주소, commit, 변경 상태 |
| 실행 환경 | OS·컴파일러·Python·GPU·패키지 |
| 기존 모델 | 명령→회전수→추력, 전류·전압 계산 경로 |
| 실제 신호 | 생산 코드, 단위, 기록 주기, 명령/센서 구분 |
| 재사용 후보 | 사용 목적과 재시험 항목 |
| 새 구현 범위 | 교체·삭제·이동·PX4 patch의 목적 |

공식 문서와 실제 선택한 release 소스를 근거로 `docs/ARCHITECTURE.md`에 선택을 확정한다. 불명확한 API는 작은 조사 프로그램으로 확인한다.

### 2.2 권장 구조

다음 역할이 드러나는 간단한 구조로 시작한다. 실제 코드량과 책임에 맞게 폴더·모듈을 조정한다.

```text
sim/
  pyproject.toml
  CMakeLists.txt
  src/                         # CLI·실행기·수집·공개자료 보정
  physics/                     # 독립 C++ 물리 core와 단위 시험
  adapters/                    # 선택한 simulator와 PX4 연결
  profiles/                    # 기체·부품·센서·환경·시나리오
  patches/                     # PX4 등 필요한 재현 가능한 변경
  tools/                       # 설치·빌드·자료 취득·검증
  tests/                       # 계약·연결·실행 시험
  docs/                        # 실제 사용법·근거·검증 결과
contracts/                     # 공통 스키마·예제·호환 시험
ai/                            # AI 담당 구현
```

시뮬레이터는 기체 강체 운동을 적분한다. 물리 core는 추진계·배터리·열 상태와 힘을 계산한다. 센서 adapter는 관측을 만들고 PX4에 전달한다. 원시 수집기는 그 관측과 제어기 상태를 기록한다.

계약과 공통 fixture는 두 파트가 함께 갱신한다. 각각의 CLI는 자기 역할의 의존성으로 실행되게 만든다. 관측 예제만으로 AI 개발을 시작할 수 있도록 작은 fixture를 조기에 제공한다.

## 3. 기체·부품 프로파일

### 3.1 프로파일이 표현할 것

기체 질량·관성·모터 배치·프로펠러·ESC·모터·배터리·센서·공기역학 정보를 구조화한다. 이번 지원 기체는 X500 V2 공개 사양에 근거한 quad-X 한 구성이다 [S10]. 기체 형상과 부품 버전, 배터리 구성, 탑재량을 명시한다.

배터리는 CMU 기반 연구용 가상 팩과 실제 제품 사양 기반 팩을 혼동하지 않고 후보의 근거를 구분한다. 이번 캠페인에는 근거를 채운 기준 팩 하나를 선택해 X500 구성과 함께 고정한다. 두 종류를 모두 구현·학습해야 하는 요구가 아니다. 제조사 명판만으로 확인되지 않은 응답은 가정과 검증 범위로 남긴다.

프로파일 예제는 다음과 같다. 이 수치는 스키마를 설명하는 초기 예제이며, 실제 reference profile은 근거·검증·가정 범위를 채워 제공한다.

```yaml
schema: drone.vehicle/2.0
id: x500_reference
frame: FRD
geometry:
  mass_kg: 1.8
  inertia_kg_m2: [[0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.05]]
  center_of_mass_m: [0.0, 0.0, 0.0]
rotors:
  - {id: front_right, position_m: [0.18, 0.18, 0.0], thrust_axis: [0.0, 0.0, -1.0], spin: 1}
  - {id: front_left, position_m: [0.18, -0.18, 0.0], thrust_axis: [0.0, 0.0, -1.0], spin: -1}
  - {id: rear_right, position_m: [-0.18, 0.18, 0.0], thrust_axis: [0.0, 0.0, -1.0], spin: -1}
  - {id: rear_left, position_m: [-0.18, -0.18, 0.0], thrust_axis: [0.0, 0.0, -1.0], spin: 1}
propulsion_profile: x500_average_reference
battery_profile: cmu_reference_4s2p
sensor_profile: common_telemetry_reference
provenance_file: x500_reference_evidence.json
```

프로펠러 직경·회전 방향·축·모터 위치·PX4 allocation은 같은 geometry에서 생성한다. `spin`은 thrust axis를 양의 축으로 본 회전 부호이며, CW/CCW 명칭과의 관계를 도해 또는 수치 예제로 설명한다.

이번 motor topology는 X500 quad-X의 4개 모터다. 배열과 위치를 geometry에 기록하고 선택한 구성과 일치하는지 검사한다. 다른 모터 수·새 형식의 allocation·공기역학 지원은 후속 범위이며 이번 인수 조건이 아니다.

### 3.2 파라미터 근거

모든 중요한 물리 파라미터에 값, 단위, 출처 유형, 원문 위치, 적용 조건, 불확실성 범위를 연결한다.

출처 유형은 `manufacturer`, `public_measurement`, `upstream_default`, `fitted`, `engineering_assumption`으로 구분한다. 추정값은 어떤 시험 결과를 바꿀 수 있는지 sensitivity 보고서에 포함한다.

기체 질량은 프레임·모터·배터리·제어기·탑재량 합계와 일치시킨다. 관성은 장착 위치와 좌표 변환을 포함해 계산하고, 양의 정부호와 질량중심 기준을 검사한다.

## 4. 단위·좌표·시간

### 4.1 단위와 변환

물리 계산은 SI로 통일한다. 회전속도 rad/s, 전류 A, 저항 ohm, 용량 Ah, 내부 온도 K를 사용한다. 외부 온도 degC는 경계에서 변환한다.

| 변환 | 정의 |
|---|---|
| ENU → NED | `[east, north, up] → [north, east, -up]` |
| FLU → FRD | `[forward, left, up] → [forward, -left, -up]` |
| RPM → rad/s | `rpm * 2*pi/60` |
| 배터리 전류 | 방전 방향을 양수로 정의 |
| 올라가는 속도 | `climb_mps = -velocity_ned_z` |

자세 변환은 회전행렬의 기준 좌표계를 명시하여 합성한다. quaternion 순서와 body→navigation 방향을 metadata에 저장한다. hover·yaw·한 축 가속 예제로 방향을 검증한다.

### 4.2 시간과 수치 계산

시뮬레이션 clock을 물리·센서·관측·정답의 기준으로 사용한다. wall clock은 성능 측정과 정지 감시에 별도로 사용한다.
시뮬레이션 pause 동안 물리 상태와 확률 과정의 시각을 유지한다. reset은 상태·랜덤 생성기·stream identity를 명시적으로 초기화한다.

초기 수치 설정의 후보는 다음과 같다. 실제 모델 안정성과 dt 수렴 시험으로 최종값을 확정한다.

```yaml
integration:
  world_step_s: 0.001
  power_max_substep_s: 0.00025
  wind_update_s: 0.01
  bus_voltage_tolerance_v: 0.000001
  bus_max_iterations: 64
logging:
  truth_period_s: 0.01
  battery_publish_period_s: 0.1
```

각 센서·제어 신호의 원래 측정 주기로 기록하고, AI가 자기 sampling profile로 처리한다. physics step과 AI 10Hz 기준을 서로 다른 설정으로 다룬다.
적분 단계의 시작·중간·종료 상태 정의를 문서화하고, 하위 모델들이 같은 단계의 상태를 사용하게 한다.

## 5. 통합 전동계 core

### 5.1 API와 상태 관리

Gazebo/PX4 헤더 없이 단독 시험할 수 있는 물리 라이브러리를 구현한다. 최소 API는 다음 역할을 제공한다.

```cpp
struct StepInput {
  double dt_s;
  std::vector<double> esc_request;
  bool armed;
  Vec3 body_velocity_mps;
  Vec3 body_angular_velocity_radps;
  Vec3 wind_body_mps;
  double air_temperature_k;
  double air_density_kg_m3;
};

class Powertrain {
 public:
  StepOutput step(const StepInput& input);
  Snapshot snapshot() const;
  void restore(const Snapshot& snapshot);
  void reset(const InitialState& state);
  ApplyResult apply(const ParameterChange& change);
};
```

전압 후보를 시험하는 `evaluate_at_voltage`는 입력 상태에서 출력만 계산하는 함수로 작성한다. 실제 상태 갱신은 수치 해가 결정된 단계에서 수행한다. 랜덤 센서 오차도 센서 샘플 생성 시점에 한 번 진행한다.

StepOutput에는 실제 rotor omega, 토크·추력·DC 전류, 단자전압, 열 상태, 기체 force/moment, solver 상태, 에너지 잔차를 담는다. 이 중 실제 기체에서 측정할 수 있는 항목은 센서 계층을 거쳐 관측으로 전달한다.

### 5.2 평균 모터·ESC 출발 모델

첫 구현은 다음 평균 관계를 기준으로 삼는다. NASA ProgPy의 전기·회전 모델은 방정식 구조와 단위 비교에 사용한다 [S4]. 여기의 파라미터는 목표 모터 자료로 추정한다.

```text
R(T) = R_ref * [1 + alpha_R * (T - T_ref)]
e    = K_e * omega
d_req = output_map(esc_request, operating_mode)
I_req = max((d_req * V_bus - e) / R(T), 0)
tau_em = K_t * I_motor
tau_friction = b * omega + tau_c * tanh(omega / omega_epsilon)
J * domega/dt = tau_em - Q_prop - tau_friction
```

`I_motor`는 평균 등가 모터 전류다. motor phase current, ESC DC 입력 전류, PWM duty를 각각 정의한다. SI 평균 모델에서 `K_t`와 `K_e`의 일관성을 확인한다. KV에서 변환한 상수는 데이터 보정 전 prior로 표시한다.

초기 motoring branch에서는 실제 전류를 current limit과 DC bus power limit에 맞춰 정한다. 요청 가능한 effective duty의 범위와 idle/disarmed 동작을 설정한다.

```text
V_motor = e + R * I_motor
P_motor_in = V_motor * I_motor
I_dc = P_motor_in / (eta_esc * V_bus) + I_dc_idle
```

DC 전류 한계는 `P_motor_in / eta_esc <= V_bus * (I_dc_max - I_dc_idle)`를 만족하도록 전류를 구한다. ESC 명판의 DC 한계와 등가 모터 전류 한계를 별도 파라미터로 둔다.
고속 coast 구간, zero-voltage 구간, disarmed 구간, duty 포화 구간을 각각 연속적인 상태 전이로 처리한다. 회생·상세 스위칭이 목표 관측에 필요한 경우 해당 분기와 에너지 검증을 추가한다.

초기 적분은 RK2 또는 검증된 안정 방법으로 구현한다. 각 stage에서 같은 bus·omega 상태로 torque와 power를 평가한다. omega=0 경계와 전류 포화에서 시간 간격 변화에 대한 안정성을 확인한다.

### 5.3 열과 에너지

```text
P_copper = I_motor^2 * R
P_friction = tau_friction * omega
P_esc_loss = V_bus * I_dc - P_motor_in
C_motor * dT_motor/dt = P_copper + P_friction - H_motor * (T_motor - T_air)
```

ESC loss는 ESC 열 상태 또는 별도 loss ledger로 기록한다. 팬·프로펠러 유동에 따른 냉각 모델은 자료가 있는 범위에서 추가한다. 계수는 양의 물리 범위를 검사한다.

다음 에너지 관계를 시간 적분해 residual을 출력한다.

```text
E_bus_to_motors = change_in_rotor_kinetic_energy
                + E_propeller_shaft_work
                + E_motor_losses + E_esc_losses + numerical_residual
```

모터가 가속될 때 rotor kinetic energy가 증가하고 감속될 때 감소하는 부호를 시험한다. 생성한 P/I/V 곡선은 에너지 ledger와 같은 계산 상태에서 나온 값이어야 한다.

## 6. 프로펠러·기체 힘

### 6.1 프로펠러 성능

자료가 확보되면 RPM·유입 속도에 따른 CT/CQ table을 사용한다. UIUC 등의 공개 시험 자료는 목표 프로펠러와 시험 조건이 맞는 범위에서 활용한다 [S8].

```text
n = omega / (2*pi)
advance_ratio = axial_air_speed / (n * D)
T = C_T(advance_ratio, operating_point) * rho * n^2 * D^4
Q = C_Q(advance_ratio, operating_point) * rho * n^2 * D^5
```

CP 자료를 쓰면 `C_Q = C_P / (2*pi)` 관계와 원자료 정의를 확인한다.
정적 모델로 시작하는 경우 `T = k_T*omega^2`, `Q = k_Q*omega^2`를 사용하고 적용 범위를 hover·저속 기준으로 기록한다. 더 넓은 운용 조건을 요구하면 해당 영역의 table이나 모델을 추가한다.

omega≈0의 유효 계산 분기, table 보간, 범위 밖 요청의 처리·quality flag를 구현한다. extrapolation 정책과 최대·최소 유효 조건을 profile에 명시한다.

### 6.2 상대풍과 모멘트

로터 위치의 상대 공기속도는 다음과 같이 계산한다.

```text
v_air_at_rotor = body_velocity + body_angular_velocity × rotor_position - wind_body
```

회전축 방향 유입과 횡방향 유입을 나눠 공기역학 모델에 전달한다. 공식 MulticopterMotorModel의 횡방향 항력 구현을 활용할 때 계수·단위·중복 힘 여부를 대조한다 [S2].

모든 로터의 추력과 반작용은 하나의 force owner가 기체에 적용한다. rotor inertia를 core에서 별도 적분하는 구조에서는 reaction torque에 shaft load와 rotor acceleration을 반영한다.

```text
M_reaction_i = -spin_i * (Q_i + J_i * domega_i/dt) * axis_i
h_rotor_i = spin_i * J_i * omega_i * axis_i
M_gyro_i = -body_angular_velocity × h_rotor_i
M_body = sum(position_i × F_i + M_reaction_i + M_gyro_i) + M_drag
```

축 기준과 sign convention을 명확히 맞추고, 독립 회전 시험으로 검증한다. 시각적 프로펠러 애니메이션은 실제 물리 힘을 계산하는 경로와 분리한다. 기체 inertia에 포함한 rotor 관성과 별도 rotor 상태의 기여를 mass/inertia ledger로 설명한다.

### 6.3 동체 항력

초기 모델은 body axes별 `F_drag = -0.5*rho*CdA*abs(v_air)*v_air`로 시작한다. 공력 중심과 질량중심의 차이에서 moment를 계산한다.
같은 바람벡터를 동체와 로터가 공유한다. 일반 wind force plugin과 상세 공기역학의 적용 영역은 force registry로 관리한다. 각 힘의 원천·단위·적용 링크를 기록하고 합계를 시험한다.

## 7. 배터리 전기·열 모델

### 7.1 1RC 출발 모델

```text
U_oc = N_s * OCV_cell(SOC, T)
V_terminal = U_oc - I_pack * R0_pack - V_polarization
SOC_dot = -I_pack / (3600 * Q_actual_ah)
a = exp(-dt / (R1_pack * C1_pack))
Vp_next = a * Vp + R1_pack * (1 - a) * I_pack
P_heat = I_pack^2 * R0_pack + Vp^2 / R1_pack
C_thermal * T_dot = P_heat - H_pack * (T - T_air)
```

이 식은 dt 동안 전류와 해당 파라미터를 일정하게 평가하는 업데이트다. 동적 전류에서 stage별 상태를 일관되게 계산하고 dt 수렴성을 확인한다.
R1=0의 정적 모델 profile은 Vp=0인 별도 검증 분기로 처리한다. OCV의 온도 의존성과 가역 열은 자료 가용성에 맞춰 모델에 포함하고 적용한 가정을 보고한다.

초기 SOC는 physics truth로 관리한다. 운영자가 알고 있는 명판 용량과 실제 열화 용량은 서로 다른 필드로 둔다. 측정 전압·전류에서 PX4가 추정한 잔량은 추정 경로를 metadata에 기록한다.

### 7.2 셀에서 팩으로

동일한 셀을 Ns직렬·Np병렬로 구성하는 근사에서는 다음 스케일을 사용한다.

```text
Q_pack = Np * Q_cell
U_pack = Ns * U_cell
R_pack = (Ns/Np) * R_cell
C1_pack = (Np/Ns) * C1_cell
```

RC time constant가 동일하게 유지되는지 시험한다. 배선·커넥터·BMS 저항, 셀 불균형, 포장·냉각은 필요한 수준에서 별도 요소로 정의한다. 총 질량과 열용량은 기체 profile과 연결한다.

### 7.3 배터리–모터 결합 계산

전체 전류는 모터 DC 입력 전류와 보조 소비 전류의 합이다. 각 integration stage에서 다음 식을 푼다.

```text
G(V) = V + R0 * I_total(V, omega, command, temperature) + Vp - U_oc = 0
```

양의 물리 전압 범위에서 bracket을 만들고 Brent/bisection 등 수렴성이 확인되는 solver를 사용한다. residual, iteration count, 해의 범위를 기록한다.
보조 장치의 전력 소비 법칙을 constant current / resistance / power 중 실제 의도에 맞게 선언한다. constant-power 부하는 저전압에서 다중 해·collapse가 생길 수 있으므로 해당 모델의 해 선택과 전류 제한을 시험한다.
정상 해, 물리적 brownout, 에너지 소진, 입력 오류, 수치 solver 실패를 서로 다른 상태로 분류한다. 수치 실패 자료는 품질 상태를 포함해 종료·기록하며, 물리적 저전압 사건은 해당 시나리오의 실제 사건으로 기록한다.

### 7.4 센서 전달

physics terminal V/I → 센서 sampling·bias·noise·지연 → PX4 배터리 입력 → PX4 추정·안전 반응의 순서를 구현한다.
선택한 PX4에서 실제 배터리 생산자를 확인하여 한 가상 팩의 센서 경로를 일관되게 구성한다. 적용한 모듈·파라미터와 실행 후 읽어온 값으로 이를 검증한다.
센서가 관측하지 않는 온도·셀 전압은 센서 profile의 가용성으로 표시한다. 실제 센서가 있는 profile에서는 해당 채널을 같은 방식으로 생성한다.

## 8. 공개 자료 보정과 검증

### 8.1 자료 취득

공식 metadata에서 파일 이름·version·URL·크기·hash·라이선스를 확인하고 필요한 자료를 취득하는 도구를 만든다. 원자료 hash와 사용한 변환을 provenance로 남긴다.
CMU 배터리 데이터는 [S6,S7]의 공식 metadata를 따라 취득한다. import 과정에서 단위, 전류 부호, 시간 중복·공백, RPT 관계, 셀 ID를 검사한다.

자료 확보 상태는 실행 결과로 기록한다. 소프트웨어 연결 시험에는 작은 synthetic fixture를 제공하고 origin을 명시한다. 공개 실험 자료 기반 결과는 실제 취득·보정한 profile로 별도 표시한다.

### 8.2 파라미터 식별

OCV, 즉각적인 저항, 느린 분극·회복을 분리해서 추정한다. 낮은 전류 방전의 단자전압 곡선과 충분한 휴지 OCV의 차이를 모델 근거에 기록한다.
전류 step의 1초 유효 저항은 해당 시간에 나타난 동적 효과를 포함한다. RC 모델을 추가할 때 step·휴지 구간을 함께 fit하여 각 효과가 하나의 파라미터 집합으로 설명되게 한다.

물리 범위에 제약된 least squares로 Q, R0, R1, tau, OCV 파라미터를 추정한다. 자료로 구분 가능한 파라미터 수를 Jacobian/조건수/민감도와 holdout 잔차로 판단한다. 식별이 약한 파라미터는 가정 범위와 불확실성으로 표시한다.
보정에 사용한 pulse·cycle과 평가에 사용할 구간을 분리하고, 같은 셀의 RPT를 같은 계보로 묶는다.

### 8.3 독립 비교

분석식으로 검증 가능한 RC step·무부하·저항 부하 시험을 먼저 수행한다. 같은 OCV·R·RC·열 경계를 가진 PyBaMM 등가회로를 오프라인 비교 대상으로 사용한다 [S5].
모터는 정상상태 전압·회전수·전류·추력 표와 비교하고, 동적 응답 자료가 없는 부분은 가정과 sensitivity를 제시한다. 동일 모델끼리의 수치 일치와 실제 부품 정확도 평가를 구분한다.

## 9. 바람·센서·정상 환경

### 9.1 평균과 시간 변동

평균 풍속 벡터와 시간 상관을 가진 돌풍을 합성한다. 첫 후보는 Dryden 계열 과정이다. 구현의 속도·길이척도·잡음 정규화를 명시하고 hover에서도 유효한 convective speed를 설정한다 [S9].

단위 강도 연속 Gaussian 입력에 대한 한 가지 구현 기준은 다음과 같다.

```text
tau_i = length_scale_i / effective_convection_speed
H_u(s) = sigma_u * sqrt(2*tau_u) / (1 + tau_u*s)
H_v(s) = sigma_v * sqrt(tau_v) * (1 + sqrt(3)*tau_v*s) / (1 + tau_v*s)^2
H_w(s) = sigma_w * sqrt(tau_w) * (1 + sqrt(3)*tau_w*s) / (1 + tau_w*s)^2
```

선택한 정의의 output variance가 sigma²가 되도록 시험한다. 상태공간의 정확한 이산화 `Ad=exp(A*dt)`와 process covariance 적분 또는 검증된 동등 방법을 사용한다. seed와 상태를 저장·복원한다.
wind axes와 NED 축의 변환, 평균풍이 0인 경우의 기준 방향, 초기 stationary 상태 또는 warmup 정책을 정한다. 시간 간격을 바꾸어도 분산·자기상관 특성이 유지되는지 시험한다.

Gazebo WindEffects는 기본 연결·비교 시험에 활용할 수 있다 [S3]. 공간 바람장이나 CFD 결과가 실제 연구 질문에 기여하면 wind sampler의 위치·시간 입력으로 확장하고 추가 검증한다.

### 9.2 정상 조건 변화와 건강 적격성

선택한 X500 기준 구성의 건강한 모터·배터리·프로펠러·ESC·센서·기체 설정에 대해 정상 범위를 먼저 정의한다. 그 안에서 정상 제작 편차를 설정하고 초기 SOC, 허용 온도, 탑재량, 바람, 비행 임무를 다양하게 조합한다. 단순히 착륙에 성공했다는 이유로 정상 학습 적격이라고 판정하지 않는다.

`health_eligibility`는 `confirmed_healthy | uncertain | abnormal`로 기록하고 근거·normal_range_revision을 남긴다. 건강 경계와 검출 대상 열화 범위는 실험 전에 수치/근거로 정한다. 정상 저항·효율을 열화 범위까지 임의로 넓혀 AI가 이상을 정상으로 배우게 만들지 않는다. 제작 편차와 열화 강도는 구분한다.

X500의 기본 형상·모터 배치·선택 부품·펌웨어/제어 설정은 기준 profile과 revision으로 고정한다. 탑재량 변화는 선언한 범위에서 질량·관성을 일관되게 갱신한다. 부품 제작 편차를 사용하는 경우도 같은 기준 구성의 근거 있는 건강 범위 안에서만 적용하며, 다기종/부품 카탈로그 randomization은 요구하지 않는다.
질량·관성·부품 상수와 배터리·추진 assembly ID는 해당 가상 자산에 귀속한다. 매 실행의 초기화·seed·run ID·paired/retry 관계를 제공한다. 같은 기준 기체에서 독립 실행을 여러 번 생성할 수 있으며, 여러 기체나 새 asset을 확보해야 분할할 수 있다는 조건을 두지 않는다. 같은 부품의 누적 추적은 실제 연속 계보가 있을 때만 연결한다.

정상 자료에는 저잔량·큰 정상 부하·허용 저온 등의 어려운 정상 조건도 포함한다. 이상 세기만으로 조건이 분리되지 않도록 정상/이상에서 임무·환경의 분포를 대응시킨다. 정상과 열화 반응이 중첩되는 조건도 별도 보존한다.

환경 온도·실제 SOC·바람 truth를 기록하는 것과 AI 입력으로 제공하는 것은 다르다. 공통 관측에서는 개별 RPM·개별 모터 전류·모터/배터리 온도·개별 셀 전압·실제 건강 파라미터를 필수 입력으로 제공하지 않는다. 내부 열 모델은 유지하되 기본 관측 가용성 밖 값은 evaluation truth에 저장한다. 온도를 시뮬레이션한다고 AI에 온도 센서가 생긴 것으로 취급하지 않는다.

### 9.3 센서 모델

센서마다 sampling, bias, white noise, 저주파 drift, 양자화, saturation, transport delay를 필요 수준으로 구성한다.
IMU 특정힘·각속도와 PX4 estimator 출력은 각 생산 경로를 구분한다. 기체 정지·hover·정해진 회전에서 원시 센서와 PX4 출력의 의미를 시험한다.
데이터 공백·지연·일시적 누락은 별도 관측 품질 시나리오로 생성한다. 물리 고장과 데이터 품질 오류는 서로 다른 사건으로 기록한다.

## 10. PX4와 simulator 연결

### 10.1 명령 추적

선택한 PX4 release에서 actuator allocation → 출력 변환 → simulator transport 경로를 실제 소스로 확인한다. 추력 설정, 정규화 출력, ESC duty 요청, RPM 명령을 구분한다.
새 평균 전동계가 필요한 post-mapping 명령을 받도록 adapter 또는 작은 PX4 patch를 구현한다. 출력 변환을 적용한 단계와 횟수, arm/disarm/min/max 규칙을 문서화한다.

명령 protocol에는 schema version, motor order, command semantic, sequence, sample time을 넣는다. 실제 사용한 요청값을 기록할 수 있는 uORB 또는 exporter 경로를 만든다.
실제 rotor feedback이 있는 sensor profile은 계산된 회전 상태에 센서 모델을 적용해 feedback으로 발행한다. 명령과 측정은 별도 topic·필드로 관리한다.

### 10.2 기체·배터리 적용

선택한 엔진의 system callback과 PX4 clock 동기화에 맞춰 한 step의 순서를 확정한다.

```text
명령 수신 → 신선도·운용 상태 확인 → 바람/상대풍 계산
→ 결합 전동계 적분 → 힘·모멘트 적용
→ 엔진의 강체·센서 갱신 → PX4 관측 전달
→ 해당 timestamp의 관측·정답 기록
```

실제 엔진 callback상 순서가 다르면 각 timestamp가 어느 step 상태를 뜻하는지 명시한다. 지연 step을 golden trace로 검증한다.
명령 timeout은 설정된 coast/disarm 상태로 전이시키고 상태·원인을 기록한다. paused 상태와 실제 통신 손실은 wall/sim clock 기준으로 구분한다.

### 10.3 원시 exporter

PX4에서 실제 제어 명령, 목표·실제 각속도, 자세, 가속도, 수직 상태, 배터리, 운용 상태를 원시 timestamp와 함께 수집한다.
기존 uORB/ULog/transport가 요구를 충족하면 활용하고, 부족한 관측은 작은 exporter를 추가한다. 수집 주기·부하·손실률을 측정한다.
exporter queue는 수용량과 overflow 진단을 갖추고, 기록 장애가 생기면 종료 상태와 복구 가능한 파일을 남긴다. physics와 기록 thread 사이의 자료 전달은 snapshot을 기준으로 구현한다.

## 11. 자동 비행과 평가용 이상

### 11.1 실행 상태

실행기는 다음 상태와 오류 원인을 명시적으로 관리한다.

```text
created → preparing → simulator_ready → px4_ready → preflight_ready
→ armed → airborne → mission → landing → disarmed → finalizing → completed
```

각 상태의 진입 조건은 실제 heartbeat·mode·arming·위치·착륙 상태 등으로 확인한다. 시각 기반 timeout은 실패 원인과 관측 상태를 함께 기록한다.
실행마다 고유 port·process group·출력 경로를 배정한다. 종료 시 해당 실행이 소유한 프로세스와 스트림을 정리하고 flush·최종 metadata를 완성한다.
대상 endpoint와 simulation identity를 실행 시작 때 확인한다. 실제 장비 연결은 별도 deployment profile로 대상·권한·운용 절차를 명시한다.

### 11.2 정상 시나리오

초기 정상 세트는 takeoff–hover–land, 고도 변화, 수평 왕복, yaw 변화, 완만한 waypoint 임무로 구성한다. 각 임무를 바람·SOC·탑재량·온도 조건과 조합한다.
상태 기반 명령과 목표 도달 판정, mission timeout, 실제 제어기 상태를 기록한다. 정상 적격성은 물리 health 설정과 운용 결과를 함께 관리한다. 배터리뿐 아니라 추진계통이 함께 건강한 자료를 기본 정상 train/validation/calibration 후보로 생성한다.

campaign manifest는 임무×부하/탑재량×바람×안전한 방전 상태×온도의 표와 조합별 비행 수·유효 노출시간·실패/제외 사유를 제공한다. 모든 조합을 무조건 실행하기보다 모델 유효 범위와 목표 운용 범위에 맞춰 정하되 특정 동작/잔량에 편중된 사실을 숨기지 않는다.

정상 세트는 비행 단위로 독립 train/validation/calibration/test에 배정할 충분한 개체·seed 계보를 가진다. AI의 60초 문맥과 이후 target·경보 평가에 필요한 유효 구간을 확보한다. 짧아 warmup조차 끝나지 않는 비행을 유효한 정상 노출시간으로 세지 않는다. 긴 capacity 시험은 별도 protocol로 충분한 전하 사용 범위와 안전 종료를 정의한다.
PX4 안전 반응이 발생한 비행도 실행 결과로 보존하고 정상 학습 적격성·이상 평가 역할을 별도 분류한다.

### 11.3 이상 시나리오와 검증 질문

모든 변화는 물리 내부에 적용하고 동일 센서 경로를 거쳐 나타나게 한다. 로그 전압에 직접 상수를 빼거나 AI 점수를 높이는 후처리로 물리 열화를 대신하지 않는다.

| 시험 요인 | 고정할 건강 조건 | 적용/확인할 내용 |
|---|---|---|
| 배터리 저항 증가 | 건강한 추진계통, Q 등 나머지 기준 고정 | R0 증가에 따른 부하 전압 반응 |
| 배터리 RC 변화 | 건강한 추진계통, 다른 전기 조건 명시 | 분극·회복 변화, R0 변화와 구분 |
| 용량 감소 | 건강한 추진계통, R0/RC 우선 고정 | Q_actual 감소, 충분한 방전 진행의 차이 |
| 모터 손실 증가 | 건강한 배터리 | 마찰/권선저항 등 각 물리 요인을 우선 따로 변경 |
| 추진 특성 변화 | 건강한 배터리 | CT/CQ 또는 검증된 효율/부하 모델, 기체 보상과 전력 |
| 복합 이상 | 단일 요인 시험 이후 | 배터리+추진 변화의 상호작용, 원인 미구분 허용 |

임의 ‘효율’ 상수로 추력만 줄이고 전류/에너지 관계를 그대로 두지 않는다. CT와 CQ도 실제 고장과 연결되는 범위를 명시하고 자료가 없으면 등가 파라미터 변화 시험이라고 표시한다. 어떤 열화가 전류를 항상 증가시키거나 감소시킨다고 고정하지 않고 실제 closed-loop·에너지 결과로 확인한다.

#### 11.3.1 발생 형식

`pre_existing`: 시작 전부터 배터리 R0/Q 또는 추진 특성이 저하된 상태로 초기화하고 해당 health 파라미터를 비행 중 고정한다. 초기 SOC·온도·분극 상태는 명시한다. 같은 초기 SOC와 같은 초기 잔여 Ah는 다른 조건이므로 baseline을 혼용하지 않는다. 물리적인 최초 노화 발생 시각은 null일 수 있고 시뮬레이터 초기 적용 시각과 구분한다.

`in_flight_step`/`in_flight_ramp`: 비행 중 고장/변화 요청과 실제 적용 시각을 기록한다. 램프는 시작·속도·최종값·사전 정의한 열화 경계 도달을 구분한다. 용량·노화 파라미터를 임의로 급변시키는 경우 전하/에너지 보존과 상태 재매핑을 명시한다. 현실적인 capacity 열화의 기본 시험은 pre_existing이다.

배터리와 추진 각각에 대해 pre_existing과 비행 중 변화 시험을 구분한다. 비행 중 급변만 탐지하는 모델이 ‘누적 열화도 검출’한다고 보고되지 않도록 한다. 관측 차이가 약한 조건·제어 보상으로 기체 운동이 유지되는 조건도 보존한다.

#### 11.3.2 정상 대조와 열화 수준 비교

paired 정상/이상은 같은 nominal 기체·임무·외란 생성 seed를 사용하고 건강 요인만 달리한다. closed-loop 궤적이 달라지므로 관측 시계열이 동일하다고 가정하지 않는다. 약/중/강의 구체 수치는 데이터·물리 범위·시험 근거로 config에 정하고 검출 보장 수치로 쓰지 않는다.

같은 가상 자산의 여러 health 수준과 비행 순서를 연결하는 평가가 가능하도록 asset/부품 교체 계보를 제공한다. 단순 severity sweep은 실제 노화율/수명 모델이 아니다. 수백 번의 물리 노화를 새로 구현하지 않아도 고정된 열화 상태를 시험할 수 있지만, RUL/비행 횟수별 열화 예측은 그 관계를 뒷받침하는 별도 모델/자료가 필요하다.

정상 변동 범위·열화 세기·수명 순서·fault 종류·실제 내부 SOC는 evaluation에 둔다. 명판/형상 reference와 기본 observations에 실제 열화 파라미터를 섞지 않는다. 정상 적격성과 분할 정보는 dataset 선정용, 고장 종류/세기는 독립 evaluator용이며 neural feature로 사용하지 않는다.

### 11.4 제어 인터페이스

JSON 또는 typed RPC로 `status`, `set_scenario`, `apply_parameter_change`, `reset`을 제공한다. request ID, target, 유효값 검사, 요청 시각, 적용 시각, revision, 적용 결과를 응답한다.
중복 request ID의 처리와 동시 요청 순서를 명확히 하고, physics step 경계에서 원자적으로 적용한다. 정상/이상 쌍은 같은 초기 기체·환경 seed와 paired_case_id로 연결한다.

## 12. 실제 제공할 CLI

CLI 명칭은 아래 형태로 제공한다. 프로젝트 구조를 변경해도 사용자 기능과 help 예제를 일관되게 유지한다.

```bash
capstone-sim doctor --report artifacts/environment.json
capstone-sim bootstrap --profile development
capstone-sim build
capstone-sim profile validate profiles/x500_reference.yaml
capstone-sim data fetch --dataset cmu-evtol --out data/public/cmu
capstone-sim fit-battery --config profiles/fitting/cmu.yaml
capstone-sim bench --config profiles/bench/reference.yaml
capstone-sim run --vehicle x500_reference --scenario hover_normal --seed 17
capstone-sim campaign plan --config profiles/campaigns/reference.yaml
capstone-sim campaign run --config profiles/campaigns/reference.yaml
capstone-sim export --run runs/EXAMPLE --format drone-raw-2
capstone-sim verify --suite unit
capstone-sim verify --suite integration
capstone-sim verify --suite end-to-end
```

각 명령은 성공·입력 오류·환경 오류·실행 실패의 exit code와 JSON 결과를 제공한다. `--help`와 사용자 README에 실제 생성 파일, 실행 위치, 이어서 실행할 명령을 설명한다.
campaign plan은 비행 수·예상 시뮬레이션 시간·저장 용량·동시 실행 수를 계산한다. run은 진행률과 실패 이력을 저장하고 선택한 checkpoint에서 재개한다. 새로운 시도는 기존 결과와 연결되는 새 run ID를 가진다.

## 13. 시험·측정·완료 기준

Python lint·타입 검사와 C++ format·정적 분석을 실제 코드에 맞게 설정하고, 각 변경의 관련 시험과 함께 실행한다. 검사 도구와 결과를 verification report에 기록한다.

### 13.1 단위·수치 시험

| 시험 | 실제 확인할 결과 |
|---|---|
| 단위·좌표 | 알려진 벡터·자세의 FRD/NED 변환과 역변환 |
| 정상상태 모터 | 전압·omega·전류·토크 balance |
| 가속·감속 | rotor 에너지와 기계·전기 power balance |
| 포화·저전압 | 모터/ESC limit, zero-voltage 상태의 유한한 값 |
| 배터리 RC | 분석식 step·relaxation과 일치 |
| 전하 적분 | 알려진 전류 시간적분과 SOC 변화 |
| 셀·팩 | Ns/Np 스케일과 time constant 일치 |
| 결합 solver | 전압 residual·수렴·물리 collapse 분류 |
| 힘·모멘트 | 대칭 hover, 한 로터 변화, yaw 부호, 관성 기여 |
| 바람 | seed 재현, 평균·분산·자기상관·dt 변경 |
| 센서 | bias·잡음·지연·누락과 metadata 일치 |
| 요청 처리 | 중복·동시 요청의 적용 시각·revision 일치 |

수치 분석식 비교는 double precision과 문제 규모에 맞는 허용오차를 미리 설정한다. bus 잔차 1e-6V, 분석식 상대오차 1e-5 등은 시작 후보로 두고 문제별 scale을 기록한다.
수렴 시험은 dt, dt/2, dt/4에서 전압·전류·회전수·에너지를 비교한다. 개선 추세와 잔차를 보고하며 수치오차와 모델 오차를 분리한다.

### 13.2 통합 시험

실제 선택한 PX4와 simulator를 빌드하고, takeoff–hover–land를 실행한다. 명령→힘→기체→센서의 연결과 배터리 전압 feedback을 측정한다.
다양한 SOC에서 고정 명령 bench와 closed-loop hover를 각각 수행한다. closed-loop 보상 때문에 같은 방향으로 변하지 않는 신호는 실제 모델 결과로 해석한다.
고장 적용 전후의 물리 상태·제어 보상·관측을 같은 timeline에서 대조한다. stream gaps, exporter 지연, solver 실패, 실시간 배율을 보고한다.
프로세스 종료·재시작·디스크 기록 장애·부분 자료 복구·병렬 port 분리를 시험한다.

별도 검증으로 명령→DC 전류→단자전압, 전압→모터→추력→기체 운동, PX4의 보상 명령을 한 timeline에서 확인한다. 전압이 부하와 무관한 시간 함수라면 부하 반응 학습용 자료로 승인하지 않는다. 건강한 저잔량/고부하와 배터리 저항 증가를 구분하는 paired trace, 건강한 배터리와 추진 손실 증가의 trace를 보존한다.

pre_existing 시나리오가 t=0부터 건강 파라미터를 유지하고 실제 중간 급변이 없는지 시험한다. event2.1의 physical onset과 initialization 적용 시각을 구분하고 소비자가 관측 기준 인지시간을 계산하는지 연결 시험한다. 노이즈·필터 후 공통 관측에서 분리되지 않는 사례도 삭제하지 않는다.

### 13.3 검증 등급

```text
implementation: planned / implemented / unit_tested / integration_tested
physics_evidence: analytic_checked / public_data_compared / hardware_validated
campaign: prepared / running / completed / partial / failed
```

각 상태에는 명령·날짜·환경·결과 파일을 연결한다. 외부 자료나 장비가 필요한 상태는 그 필요 조건과 현재 확보 범위를 적는다.
가상 기체가 실행된 사실은 소프트웨어 통합 증거다. 실제 기체와의 오차는 해당 실측 자료를 확보한 시험에서 측정한다.

### 13.4 최종 인수 기준

아래 체크리스트는 X500 기준 구성의 현재 범위에만 적용한다. 회사 기체·다기체 사전학습·기종 간 이전·실기체 검증은 이번 필수 합격 조건이 아니다.

- [ ] 깨끗한 지원 환경에서 설치·빌드 명령이 실행된다.
- [ ] 선택한 버전·patch·라이선스·공개 자료 출처가 기록된다.
- [ ] 선택한 X500 profile에서 모델·PX4 4-모터 배치·센서 정의를 생성한다.
- [ ] 전동계·배터리 양방향 계산과 에너지·단위 시험이 통과한다.
- [ ] 평균풍·돌풍과 센서 관측이 재현 가능한 seed로 동작한다.
- [ ] 실제 X500 PX4 SITL에서 정상 임무를 수행하고 원시 관측을 생성한다.
- [ ] X500 정상 건강 적격성·운용 다양성/노출시간과 독립 run·paired/retry 계보가 기록된다.
- [ ] 평가용 이상과 실제 적용 시각이 정답 파일로 저장된다.
- [ ] 시작부터 열화된 배터리/추진계통·용량 시험이 급성 고장과 분리된다.
- [ ] 내부 RPM/온도/SOC/건강 파라미터가 기본 AI 관측으로 누출되지 않는다.
- [ ] 공통 계약으로 AI 파트가 실제 관측을 읽는 연결 시험이 통과한다.
- [ ] 사용 명령·실행 결과·남은 검증 범위가 README와 보고서에 기록된다.

## 14. 개발 진행과 최종 보고

단계는 `X500 환경·근거 조사 → 계약 fixture → physics bench → PX4 연결 → X500 자동 정상 비행 → 허용 환경 변동 → 이상 평가 자료 → AI 연결`로 진행한다. 다른 기체 추가와 파운데이션 모델 개발은 이 실행 순서에 포함하지 않는다.
각 단계에서 구현·시험·수정을 반복하고, 진행 메모에는 현재 단계와 다음 가장 작은 실행 작업을 남긴다. 테스트 결과·모델 보정·실제 비행 산출물을 제품과 함께 제공한다.

최종 보고는 구현한 기능, 바꾼 선택과 이유, 실제 실행한 시험, 주요 수치, artifact 경로, 사용자 실행 명령, 추가 자료가 필요한 검증 순서로 정리한다.
참조 자료는 `00_SOURCE_NOTES.md`의 [S1–S11]을 사용하고, 실제 채택한 추가 소스와 라이선스를 구현 저장소에 기록한다.

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
| `clock_id` | 시간축 식별. 리셋 시 새 clock/sC9 통과 결과는 유지하고 docs/AI_PROGRESS.md의 미완료 항목부터 계속 구현·시험하라. 정상 오경보 목표 미달 원인을 분석하되 기존 test로 튜닝하지 말고, 추가 정상 데이터가 필요하면 구체적인 조건을 시뮬레이션 담당에게 전달하라.egment로 구분 |
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

[S1] PX4 Gazebo 연결: https://docs.px4.io/v1.17/en/sim_gazebo_gz/

[S2] Gazebo MulticopterMotorModel: https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1MulticopterMotorModel.html

[S3] Gazebo WindEffects: https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1WindEffects.html

[S4] NASA ProgPy 모델: https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html

[S5] PyBaMM Thévenin: https://docs.pybamm.org/en/latest/source/api/models/equivalent_circuit/thevenin.html

[S6] CMU eVTOL 배터리 자료: https://doi.org/10.1184/R1/14226830.v3

[S7] CMU 원논문: https://www.nature.com/articles/s41597-023-02180-5

[S8] UIUC 프로펠러 시험 자료: https://m-selig.ae.illinois.edu/props/propDB.html

[S9] Dryden 공개 구현 참고: https://github.com/goromal/wind-dynamics

[S10] X500 V2 제조사 구성: https://holybro.com/products/px4-development-kit-x500-v2

[S11] 기존 프로젝트: https://github.com/capstone1-irion/simulation

이 문서의 architecture·방정식 조합·초기 숫자·완료 기준은 프로젝트 설계다. 실제 구현은 선택한 버전의 API·소스와 수치 시험을 대조하여 완성한다. 작업 지침은 앞부분에 기재한 네 원문 링크를 기준으로 적용한다.
