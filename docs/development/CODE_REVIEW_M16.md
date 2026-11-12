# M16 — PX4·ArduPilot 단위와 값의 의미 감사

기준일: 2026-11-15. **BIN 3개의 단위 변환 불일치는 발견하지 못했다.
PX4 실기체 로그 어댑터의 결측/유효성 처리 오류 3종은 재현하고 수정했다.**
전체 기능이 실기체에서 검증되었다거나 전압 예측 성능이 해결된 것은 아니다.

## 1. 점검 범위와 단위

실제 자료는 ArduCopter 4.4.0 `502702df`의 BIN 3개다. 최신 온라인 문서만
보고 과거 로그의 단위를 가정하지 않았다. 각 파일의 FMT/FMTU/UNIT/MULT,
설치된 pymavlink 2.4.50의 DFReader, 실제 어댑터를 함께 대조했다.
PX4는 로컬 소스 `6ea3539157ca358c70a515878b77077af7d4611d`의 메시지 정의와
공식 문서를 읽었다. 실제 PX4 기체 ULog는 아직 제공되지 않았다.

| 신호 | 이번 ArduPilot BIN | PX4 ULog | 공통 입력 |
|---|---|---|---|
| 몸체 각속도/목표 | RATE R/P/Y와 RDes/PDes/YDes: deg/s | angular_velocity/rates_setpoint: rad/s | rad/s, BIN에 π/180 적용 |
| 롤·피치 | ATT 저장은 0.01도 정수, DFReader 출력은 deg | FRD→NED wxyz 쿼터니언 | rad; BIN 파서 뒤에는 π/180만 적용 |
| 상승속도/목표 | CTUN CRt/DCRt: cm/s, 상승 양수 | local_position/setpoint vz: m/s, 하강 양수 | 상승 양수 m/s: BIN ×0.01, PX4 부호 반전 |
| 가속도 | IMU AccX/Y/Z: m/s/s | vehicle_acceleration: m/s², body FRD | m/s²; 중력값을 임의로 빼지 않음 |
| 팩 전압·전류 | BAT Volt/Curr: V/A | battery_status voltage_v/current_a: V/A | V/A; 모델 직전에만 셀 수/Ah로 나눔 |
| 모터 명령 | RCOU C1…: μs, 설정 1000…2000 | actuator_outputs: 장치별 natural output units | 명시된 범위로 0…1; 추진력/RPM으로 해석하지 않음 |

ArduCopter 4.4.0의 [ATT/RATE 형식](https://github.com/ArduPilot/ardupilot/blob/Copter-4.4.0/libraries/AP_AHRS/LogStructure.h),
[CTUN 기록 소스](https://github.com/ArduPilot/ardupilot/blob/Copter-4.4.0/ArduCopter/Log.cpp),
[저장 형식·단위 정의](https://github.com/ArduPilot/ardupilot/blob/Copter-4.4.0/libraries/AP_Logger/LogStructure.h)를 기준으로 확인했다.
PX4의 [각속도](https://github.com/PX4/PX4-Autopilot/blob/6ea3539157ca358c70a515878b77077af7d4611d/msg/versioned/VehicleAngularVelocity.msg),
[자세](https://github.com/PX4/PX4-Autopilot/blob/6ea3539157ca358c70a515878b77077af7d4611d/msg/versioned/VehicleAttitude.msg),
[가속도](https://github.com/PX4/PX4-Autopilot/blob/6ea3539157ca358c70a515878b77077af7d4611d/msg/VehicleAcceleration.msg)는 로컬 원문과 대조했다.

MAVLink는 PX4/ArduPilot 브랜드별 단위가 아니라 **메시지별 규약**을 따른다.
ATTITUDE는 rad/rad/s, HIGHRES_IMU는 m/s², SCALED_IMU/2/3는 milli-g이므로
가속도에 9.80665/1000을 곱한다. BATTERY_STATUS는 mV/cA이므로 전압 합을
1000으로, 전류를 100으로 나눈다. 개별 셀을 모르면 첫 전압 칸이 팩 전압일
수 있으므로 배열 칸 수를 셀 수로 간주하지 않는다. 무효값과 목표값 ignore
mask는 별도로 처리한다. [MAVLink 공식 정의](https://mavlink.io/en/messages/common.html).

단위가 같아도 필터·바이어스 보정·샘플 시각이 같은 신호라는 뜻은 아니다.
온보드 로그와 실시간 입력은 각각 검증해야 한다. PX4 송신부가 출력 배열을
SERVO_OUTPUT_RAW에 직접 복사하는 것도 확인했다. 실제 장치의 출력 프로토콜과
범위는 확인이 필요하며, 메시지 이름만으로 PWM 범위를 자동 입력하지 않는다.

## 2. 실제 BIN 전 구간 대조

| 원본 | RATE/RCOU/ATT/CTUN/BAT 및 IMU 0·1·2 메시지 수 | 파서 경고 | 단위 변환 불일치 |
|---|---:|---|---|
| 00000005.BIN | 184,689 | 없음 | 없음 |
| 00000011.BIN | 219,838 | 없음 | 없음 |
| 00000198.BIN | 225,495 | 없음 | 없음 |

합계 **630,022개**. 건강 상태는 계속 `unknown`이다. 각 IMU를 별도 감사했으며
학습 입력으로 혼합하지 않았다. 저장 정수/실수, 파서 반환값, 단위 메타데이터,
실제 어댑터 결과를 독립 변환식에 대조했다. 파서·변환식의 최대 차이는 모두 0.
이것은 센서 자체의 정확도나 단위가 다른 센서값 간 일치를 뜻하지 않는다.

예: `00000011.BIN` 첫 ATT.Roll은 저장값 -25 → DFReader -0.25도 →
-0.0043633231rad다. 다시 0.01을 곱하지 않는다. CTUN.CRt의 저장형 `h`는
파서가 0.01을 적용하지 않으므로 `cm/s` 메타데이터에 따라 어댑터에서 변환한다.

세 파일 모두 MOT_PWM_TYPE=0, MOT_PWM_MIN/MAX=1000/2000,
MOT_THST_EXPO≈0.81, BATT_CAPACITY=44000, AHRS_ORIENTATION=0이다.
기록된 설정은 실물 사양·배선·센서 교정이 맞다는 증명이 아니다.

PWM 정규화값을 시뮬레이션 모터 입력의 평균과 같게 만들거나 추력곡선을
임의로 역변환하지 않았다. 모터/ESC의 비선형 응답은 기체별로 다르다.
[ArduPilot 추력곡선 설명](https://ardupilot.org/copter/docs/motor-thrust-scaling.html).
이 감사로 [M15 전압 성능 미해결](CODE_REVIEW_M15.md) 상태는 바뀌지 않는다.

## 3. 재현한 오류와 수정

1. **수직속도 유효성**: 고도 `z_valid`를 사용하던 실기체 경로를
   `v_z_valid`로 수정했다. 고도가 유효해도 속도가 무효면 제외하고, 고도가
   무효여도 속도가 유효하면 상승속도를 보존한다. 표시가 없으면 결측이다.
   [PX4 정의](https://docs.px4.io/main/en/msg_docs/VehicleLocalPosition).
2. **전류 미측정**: PX4 `current_a=-1`은 미측정이므로 결측으로 바꾼다.
   다른 유한 전류를 학습 범위로 잘라내지 않는다. 기존 MAVLink의 -1cA 처리와
   별개다. [PX4 배터리 정의](https://docs.px4.io/main/en/msg_docs/BatteryStatus).
3. **비활성 모터 채널**: 배열 필드 존재만으로 사용하지 않고 `noutputs` 안에
   있는 채널만 사용한다. 유효 출력 수가 없거나 정의 범위를 벗어나면 결측이다.
   [PX4 출력 정의](https://docs.px4.io/main/en/msg_docs/ActuatorOutputs).

누락 필드 진단에 `v_z_valid`와 `noutputs`를 추가했다. 새 PX4 준비 결과에는
`timing.decoding.adapter_revision=px4-common19-units-v2`를 남긴다.
이전 패키지를 수정하거나 재해석하지 않는다. 과거 PX4 준비 자료는 이번
유효성 수정이 필요하면 원본에서 **새 패키지**로 준비해야 한다.

수정은 `ai/localflight/prepare.py`, `logs.py`에 한정했다. 동결된 학습/시뮬레이션
`capstone_ai.mapping`의 계약에는 `z_valid`만 있으므로 이를 바꾸지 않았다.
이번 실기체 어댑터 수정이 과거 시뮬레이션 자료의 속도 유효성을 소급해서
재판정했다는 뜻은 아니다. 모델·scaler·threshold·기존 결과는 보존했다.

## 4. 실행 검증

- 신규 교차 시험은 실제 DFReader와 실제 ULog 바이너리 파서를 사용한다.
  숫자를 맞추기 위한 파서 mock이나 모델 출력 수정은 없다.
- 최초 속도 유효성 회귀: **3 실패·13 통과**로 오류 재현.
- 속도 수정 후 전체 AI: **1,515 통과**, 외부 PyTorch 경고 1개, 227.68초.
- 추가 결측/채널 회귀: **5 실패·24 통과**로 오류 재현.
- 추가 수정 후 실기체 분석/전압 진단 묶음: **100 통과**, 4.27초.
- AI 중복 제거 합계는 **1,528개**. 최종 소스로 전체를 다시 한 번 실행한
  수치와 구분한다. 변경하지 않은 UI/Data/시뮬레이션 전체 시험은 반복하지 않았다.
- 원자료·모델·기존 평가·봉인 **29,427개 해시 동일**, 변경한 파일의 Ruff 통과.

로컬 증거: `ai/reports/unit-audit-20261115/`의 `audit_bin.py`, `bin-units.json`,
`regression-before.log`, `quality-before.log`, `ai-final.{log,xml}`,
`localflight-final.{log,xml}`, `preservation.json`.

재검사 명령(저장소 루트):

```bash
PYTHONPATH=ai:ai/src:Data/generation:simulation/sim/src \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
ai/.venv/bin/python -B -m pytest -q \
  ai/tests/test_localflight*.py ai/tests/test_voltage_diagnostics.py
```

## 5. 작동 확인과 아직 할 수 없는 확인

프로파일 등록·16종 CPU 실행·BIN 분석·전체 런/구간 결과·CPU 출력층 적응
절차는 소프트웨어 시험 증거가 있다. 실시간은 합성 입력의 전체 분석 흐름과
PX4/ArduPilot SITL의 수신 시험까지다. SITL에서도 일부 목표/배터리 입력이
없어 분석 보류가 발생했으며 실기체 비행 분석 성공으로 보고하지 않는다.

**추가 자료가 필요한 검증**:

- 실제 PX4 원본 `.ulg`와 해당 비행의 펌웨어/파라미터. 로그에 포함된 값은
  자동 추출하고, 빠진 설정만 추가 요청한다. CSV만으로 대체하지 않는다.
- 실제 모터 위치↔출력 채널/프로토콜, 배터리 셀 수·정격 용량 확인.
  전체 기체 CAD·무게·모터 스펙을 필수로 요구하지 않는다. 12S는 전압 추정,
  44Ah는 기록된 설정이며 실물 확인이 남아 있다.
- 전압/전류가 맞는지 확인할 센서 교정 기록 또는 독립 측정값.
  데이터만 보고 교정 계수를 정답처럼 맞추지 않는다.
- 정상 근거가 있는 적응용 비행과 별도의 검증 비행. 정상/이상 판정 성능에는
  상태 근거가 있는 독립 자료가 필요하다. 현재 BIN 3개를 정상으로 간주하지 않는다.
- 실제 FC/USB/무선 연결에서 19개 입력의 연속 수신과 종료 후 분석 검증.

현재 자료만으로 기체가 정상인지, 1~3회 파인튜닝이 정확도를 높일지, 개별
모터의 실제 추력이나 고장 원인을 확정할 수 없다. 누락된 목표값·중단 구간을
임의 생성하지 않으며 근거 없는 보정으로 정상 판정을 만들지 않는다.

**미구현**은 Windows 패키지, 전송 지연 자동 추정, 4로터 외 모델,
ULog 복수 부팅 자동 복구, 전체 사용자 데이터 경로 이관이다.
RC12 아카이브는 이번 수정으로 재배포하지 않았다. 최신 소스 실행은
[Ubuntu 실행 안내](../ai/LOCAL_DESKTOP_UBUNTU.md)를 따른다.

프로젝트 지정 네 지침은 기존 원문 캐시
`ai/reports/code-review-20261115/guidelines/`에서 재확인했다.
확인한 오류만 수정하고 최초 실패·범위·검증 한계를 기록했다.
Python 작업에 TypeScript용 Oxlint를 추가하지 않았다.
