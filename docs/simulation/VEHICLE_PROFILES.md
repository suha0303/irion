# 다른 multirotor 기체의 프로파일

기존 X500 생산자는 보존하고, 별도 vehicle·추진·배터리 사양으로 다른 고정 수직축 multirotor의 물리 모델과 Gazebo/PX4 자산을 만든다. 지원 범위는 **4..8개 모터**다. PX4의 제어 할당은 더 많은 모터를 표현할 수 있지만 이 프로젝트가 고정한 Gazebo ESC 출력은8개다. AI 모델과 기존 X500 학습·평가 계약은 이 확장으로 변경하지 않는다.

## 입력 파일

[공학 예제](../../simulation/profiles/examples/README.md)의 폴더를 복사해 `vehicle.yaml`, `propulsion.json`, `battery.json`을 작성한다. 경로는 프로젝트 설치 위치에 묶지 않고 두 `parameters_file`은 vehicle 프로파일 폴더 내부의 상대 경로로 지정한다. 추진 수치는 vehicle 파일 내부에 직접 쓸 수도 있다. 신뢰할 수 있는 측정·제조사·피팅 자료와 공학 가정을 구분해 출처를 남긴다.

| 영역 | 필수 내용 | 단위·의미 |
|---|---|---|
| 식별·배치 | `id`, `rotor_count`, `frame: FRD`, `motor_order`, `rotors` | 각 rotor의 FRD 위치[m]와 회전 방향`spin: ±1`을 같은 순서로 지정 |
| 질량·관성 | `geometry.body_mass_kg`, `rotor_mass_kg_each`, `body_inertia_diagonal_kg_m2`, `body_size_m` | body 질량·관성은 선택한 배터리와 프레임을 포함하고 rotor 질량은 별도 |
| 추진 | KV, 권선 저항, rotor 관성, 추력·토크 계수, 프로펠러 반경 | rad/s 기준 계수. 정상·허용 운전 범위의 근거를 별도 기록 |
| 배터리 | Ns/Np, Ah, 팩 R0/R1/C1,31점 셀 OCV와 SOC, 열용량·냉각 | 셀 OCV를 팩 전압으로 미리 배수하지 않음. Ns는 팩 전압 계산에 따로 적용 |
| 센서·운용 범위 | 센서 주기·잡음·범위, 최대 속도·각속도·회전속도, 최저 SOC·전압 | 새 셀 수에 맞는 전압 측정 범위와 실제 유효 범위를 명시 |

현재 body 관성은 중심 기준 대각3성분을 받는다. 배터리를 바꾸면 질량뿐 아니라 관성·저항·전압·추진 부하도 함께 맞춰야 한다. 기울어진 회전축·비대칭 전체 관성 행렬·접촉 형상 정밀 재현은 이 프로파일의 지원 범위 밖이다.

한 기체의 모든 모터·프로펠러에는 같은 추진계 사양을 적용한다. 로터 간 후류·간섭은 식별하지 않으며, 작성자가 프로펠러 간격과 기구적 적합성을 확인해야 한다. 모터마다 다른 사양이나 실제 기체의 자세제어 이득 자동 튜닝은 지원하지 않는다.

모터 마찰·쿨롱 마찰·전류 상한·ESC 효율·열 상수와 기체 항력은 프로파일에서 재정의할 수 있다. 생략 시 `profile.py`의 명시된 공학 기본값을 사용한다. 기본값이 입력된다고 해당 기체의 측정값이나 비행 적합성이 확보되는 것은 아니다.

## 형상과 제어가 연결되는 방법

Gazebo에는 `base_link`, 모터 수만큼의 `rotor_<i>`와 fixed joint를 생성한다. body·arm·프로펠러 시각 형상은 단순 도형이다. body 충돌 형상은 `body_size_m`의 상자다. 프로파일 질량·관성을 사용하므로 시각 도형으로 질량을 다시 추정하거나 rotor 질량을 body에 중복 합산하지 않는다.

PX4는 FRD 좌표를 사용하고 Gazebo는 FLU 좌표를 사용하므로 위치`(x,y,z)`를`(x,-y,-z)`로 옮긴다. 같은 rotor 순서·위치·spin이 컴파일된 물리 모델과 PX4의 `CA_ROTOR_COUNT`, `CA_ROTOR<i>_P*`, 회전축·추력·토크 할당에 들어간다. yaw 모멘트 계수는 프로파일의 토크/추력 계수 비율과 spin을 사용한다.

기존 Gazebo MulticopterMotorModel 플러그인을 함께 사용하지 않는다. 커스텀 PowertrainSystem 하나가 배터리·ESC·모터·프로펠러와 추력·반작용 토크를 계산한다. 센서4종(IMU·자기계·기압·위성항법)은 고정된 upstream 자산에서 복사해 기존 의미·잡음 형식을 유지하고 주기는 선택 프로파일을 따른다.

PX4 startup은 각 run의 `reference/px4_etc/`에 복사해 수정한다. vendor의 X500 airframe이나 다른 run의 설정은 바꾸지 않는다. 출력1..8 가운데 실제 모터만 연결하고 나머지 FUNC를0으로 비운다. 인스턴스별 offboard 포트와 GCS 포트도 분리한다. 비행 전에는 FC가 반환한 실제 PARAM_VALUE를 사용해 배터리·기하·할당·출력 매핑 값을 검증한다. INT32의 byte encoding과 REAL32를 구분하며 미응답·불일치는 성공으로 처리하지 않는다.

새 기체는 ESC 출력의 DIS/MIN=0, MAX=1000, `THR_MDL_FAC=0`을 명시하고 FC 응답으로 검증한다. 이때 정규화된 제어값과 물리 duty가 일치하므로 계산한 hover duty를 `MPC_THR_HOVER`에 넣을 수 있다. `MPC_THR_MIN=0.1`도 명시해 정적 hover 허용 하한과 맞춘다. 기존 X500 경로의 MIN150 정책은 바꾸지 않는다. DIS0은 모터 정지 출력이고 arming 조건을 우회하지 않는다.

기체 유형은 quad/hex/octo에 각각 MAV_TYPE2/13/14를 사용한다.5/7모터는 현재 PX4 commander가 generic multirotor enum을 인식하지 않아 FC 호환 분류2를 사용한다. 이 경우 FC heartbeat의 유형은 quad로 표시되지만 실제 rotor count·제어 할당·물리·관측 metadata는5/7을 유지한다. 일반 기체 enum0이나 최신 generic multirotor43을 넣어 회전익 분류를 잃게 하지 않는다.

## 확인 범위

프로파일 검증·헤더 생성·컴파일·수치 계산·합성 SDF/PX4 자산 일치 시험과 실제 SITL 비행은 별개다. 예제 quad/hex는 공학 fixture이며 실제 비행·데이터 수집·새 모델 학습을 자동 실행하지 않는다. 새로운 기체의 로그가 생성돼도 기존 X500 학습 데이터에 자동으로 섞지 않는다.

`quad_example`은 기존 X500과 같은 rotor 위치를, `hex_example`은 다른 모터 수와6S 배터리를 사용해 새 경로의 연결을 시험한다. 두 예제 모두 제조사 기체 재현이나15~20분 비행을 검증한 결과로 해석하지 않는다. 기존 X500의 동결 입력·생산자 증거·데이터·모델은 별도 보존한다.

## 사용 순서

프로젝트 최상위 폴더에서 다음 명령을 실행한다. 최초 native 환경 설치는 기존 `./Data/generation/drone-data setup --simulation`을 사용한다.

```bash
# 파일·기하·전기적 호버 여유만 검사한다. 비행을 시작하지 않는다.
./simulation/capstone-sim profile inspect simulation/profiles/examples/hex_example/vehicle.yaml

# 별도 플러그인 컴파일, 전하·에너지 수치 시험, SDF·FC startup 생성만 수행한다.
./simulation/capstone-sim profile prepare simulation/profiles/examples/hex_example/vehicle.yaml --jobs 4
```

준비 파일은 `simulation/build/vehicles/<기체 ID>/<입력·소스 recipe 해시>/`에 저장한다. 세 입력과 해석된 값, 컴파일 헤더·배터리/추진 입력·플러그인 SHA와 기준 native runtime SHA를 함께 기록한다. 입력 또는 시뮬레이션 소스가 달라지면 이전 준비본으로 실행하지 않는다. 실패는 `FAILED_PRESERVED`로 남기고 자동 재시도하지 않는다. 같은 입력의 실패·오래된 준비본을 다시 준비할 때만 사용자가 명시적으로 `--retry`를 붙인다. 이전 폴더는 `build/vehicles/failed/`에 보존한다.

이후 **사용자가 실제 시험 비행을 선택할 때만** 아래처럼 실행한다. 준비 명령이 이 단계를 자동으로 호출하지 않는다.

```bash
./simulation/capstone-sim run --vehicle-profile simulation/profiles/examples/hex_example/vehicle.yaml --scenario hover_normal --duration 120 --seed 17
```

`--instance 0..15`, `--real-time-factor 2`, 기존 유한 시나리오를 사용할 수 있다. 처음 선택한 기체는 짧은 hover부터 비행·FC gate·운용 범위를 따로 확인해야 한다. `run` 응답의 경로 아래 `run.json`, `reference/vehicle-profile.json`, `reference/fc-configuration.json`, `reference/hover-estimate.json`, `native_logs/`, `observations/`가 해당 실행의 근거다. payload 시나리오는 추가 질량을 포함해 호버 여유와 FC 호버 추정치를 다시 계산한다. 추정은 정상 SOC0.9의 정적 계산이며 온도·바람·과도 상태·비행시간 보장은 아니다.

다른 기체의 고장 시나리오는 원래 X500 절대값을 그대로 덮어쓰지 않는다. 마찰·R0·용량의 X500 기준 대비 비율을 선택한 기체의 정상값에 적용하고, R0의 고정 배선 저항은 보존한다. 예를 들어 `propulsion_loss_existing`은 해당 모터의 정상 점성 마찰을20배로 설정한다. 이는 동일한 검출 난이도나 실제 손상률을 보장하는 값이 아니다. 마찰이0인 프로파일에서 이 배율 고장은 명시적으로 거절한다. 적용된 실제 값은 `evaluation/scenario.json`에 기록한다. 기존 X500 시나리오의 값은 바뀌지 않는다.

현재 수집 명령은 `./Data/generation/drone-pretrain`이며 동작의 기본값은 기존 X500이다. 현재 300run 수집·정제/AI 인수는 X500 전용이다. 새 기체의 `drone.observation.metadata/3.0`, `drone.raw.event/3.0`은 simulator 전용이며 `ai_compatible: false`다. 검증은 `capstone-sim verify --run <경로>`, 관측 경로 출력은 `capstone-sim export --run <경로> --format drone-raw-3`을 사용한다. 이 검증이 기존 X500 C9·AI 성능 통과를 뜻하지 않는다. `recover`는 완료 비행이 아닌 관측 단편을 만들며 학습 인수 대상으로 표시하지 않는다.

## 기본 기체와 범용 파일명

기체를 지정하지 않은 `capstone-sim run`은 기존 X500 reference 설정으로 실행한다. `--vehicle x500`도 같은 선택이다. 계약 식별자 `x500_reference`는 유지한다. 수집 콘솔은 기존 X500 장시간 수집 설정을 유지한다. 두 진입점의 배터리·비행 설정을 이번 이름 정리로 서로 바꾸지 않는다.

수집용 native 설정 이름은 `collection`이며 `--configuration collection`으로 선택한다. 설정 파일은 `simulation/profiles/collection/`에 있고 현재 X500 수집 배터리 ID는 `vtc6_4s4p`다. 이 이름은 수집 설정과 실제 배터리 사양을 구분하며, 다른 기체 선택은 계속 `--vehicle-profile`로 명시한다. 새 실행은 과거 회차 이름을 사용하지 않는다. 과거 run의 설정·배터리 식별자는 검증·판독 경계에서만 읽고 해당 파일의 내용이나 해시를 고치지 않는다.

새 run은 reference·수집·사용자 기체 모두 `run.json`에 PX4 인스턴스 번호를 남긴다. 완료·실패한 run의 reset과 시나리오 변경은 같은 번호로 새 run을 만든다. 과거 기록에 번호가 없거나 범위를 벗어나면 실행을 거절하며 임의로 0번을 선택하지 않는다. 기존 기록을 보존하고 별도 `run --instance <번호>`로 시작해야 한다. 사용자 기체는 기존 프로파일·준비 recipe 일치 검사도 통과해야 한다.

```bash
# 점검만 수행하며 실제 비행·수집은 시작하지 않는다.
./simulation/capstone-sim profile inspect
./simulation/capstone-sim profile prepare
./Data/generation/drone-data doctor
```

`profile`의 경로를 생략하면 X500이다. X500 `prepare`는 설치기로 이미 빌드·봉인된 기본 플러그인을 확인하며, 다른 기체는 프로파일 경로를 명시하고 별도로 컴파일한다. `--vehicle`과 `--vehicle-profile`을 동시에 지정하면 거절한다. 캠페인 YAML은 기존 X500용이며 지원하지 않는 기체 선택 키를 조용히 무시하지 않는다. 다른 기체 시험은 `run --vehicle-profile`을 사용한다.

현재 실행 파일은 `drone-data`, `drone-pretrain`, 플러그인은 `libDronePowertrain.so`, 생성 헤더는 `vehicle_generated.hpp`다. 실제 X500 프리셋명·원본 모터 자료·기존 계약 식별자·AI 전용 이름과 과거 증거는 의미를 유지한다. 진단 `doctor`는 보고서만 쓰며 소스 봉인은 설치기의 명시적 준비 절차에서만 갱신한다.
