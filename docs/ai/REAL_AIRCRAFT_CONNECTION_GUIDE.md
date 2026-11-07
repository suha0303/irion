# PX4·ArduPilot 실기체 연결 안내

조회·구현 기준: 2026-11-14. 목적은 수신·기록·로컬 CPU 분석이다.
SITL의 MAVLink 처리 경로를 재사용하고, 실기체에서는 접속 장치/주소·전송 주기·기체 매핑을 맞춘다.
SITL 통신 성공은 실제 센서·전송 지연·기체별 모델 성능 검증과 구분한다.

## 공식 자료

- [MAVSDK: Connecting to Systems](https://mavsdk.mavlink.io/main/en/cpp/guide/connections.html): UDP·TCP·시리얼 연결 및 기체 발견. 이 앱은 MAVSDK를 새로 설치하지 않고 기존 pymavlink 수신기를 사용한다.
- [PX4: MAVLink Peripherals](https://docs.px4.io/main/en/peripherals/mavlink_peripherals): MAVLink 인스턴스별 포트·모드·전송 대역폭 설정. TELEM2 Onboard의 예시는 MAV_1_CONFIG=TELEM2, MAV_1_MODE=Onboard, SER_TEL2_BAUD=921600이다. 실제 보드/펌웨어의 설정을 먼저 확인한다.
- [PX4: Serial Port Configuration](https://docs.px4.io/main/en/peripherals/serial_configuration): USB/TELEM 포트의 배치와 구성. MAV_x_RATE는 메시지 대역폭이고 UART baud는 SER_TELx_BAUD다.
- [ArduPilot: Companion Computers](https://ardupilot.org/dev/docs/companion-computers.html): 비행제어기와 외부 컴퓨터 사이의 MAVLink 연결.
- [ArduPilot: Requesting Data](https://ardupilot.org/dev/docs/mavlink-requesting-data.html): 메시지별 SET_MESSAGE_INTERVAL 또는 스트림 그룹 주기 설정. 4.7부터 MAVn_, 이전 버전은 SRx_ 이름이므로 버전을 확인한다. 파라미터/미션 전송 중에는 요청한 주기보다 느려질 수 있다.
- [MAVProxy: Telemetry Forwarding](https://ardupilot.org/mavproxy/docs/getting_started/forwarding.html): 하나의 시리얼 입력을 QGC와 분석 앱의 서로 다른 UDP 포트로 분배.
- [MAVLink 주기 요청 예제](https://mavlink.io/en/mavgen_python/howto_requestmessages.html): 메시지별 주기 명령과 ACK 확인.
- [MAVLink Command Protocol](https://mavlink.io/en/services/command.html): 대상 system/component와 명령 결과 처리.
- [MAVLink Common 메시지 정의](https://mavlink.io/en/messages/common.html): 필드 단위·마스크·좌표계의 기준.

## 접속 순서

1. **프로파일 입력**: PX4/ArduPilot, 펌웨어, 4로터 채널 순서, 배터리 셀 수/정격 Ah, 출력 최솟값/최댓값과 매핑 근거를 저장한다.
2. **실시간 연결**: USB/텔레메트리 시리얼이면 `/dev/serial/by-id/...`와 해당 링크 baud를 지정한다. UDP이면 이 앱에 전용으로 전달할 포트를 지정한다.
   같은 PC의 router/SITL은 `127.0.0.1`, 외부 장치에서 직접 수신하려면 이 PC의 LAN IPv4 또는 `0.0.0.0`을 선택한다.
   UDP 수신 주소 변경은 웹 UI 서버 공개와 무관하며 웹 서버는 계속 127.0.0.1이다.
3. **기체별 로컬 AI → 기체와 모델 연결**: 저장한 프로파일 버전, system/component ID, 비행 중 8종 중 하나, 종료 후 8종 중 하나를 선택한다.
   실제 기체와 시뮬레이터 통신 시험을 구분한다. 채택된 파인튜닝 모델 사용을 켜면 해당 프로파일·기본 모델과 일치하는 버전을 고정해 사용한다.
4. 기본으로 켜진 **필수 메시지 주기 자동 설정**은 대상 HEARTBEAT 확인 후 주기 명령을 전송한다. 끄면 수신만 시작한다. 설정 내역에서 ACK·설정값 조회·실제 수신률을 확인하고 필요하면 다시 요청한다.
   HEARTBEAT로 autopilot을 확인하고 필수 입력을 확인한다. 부족한 입력은 화면에 표시하며 0이나 이전 비행 값으로 채우지 않는다.
5. 완전한 연속 이력이 쌓이면 약 1초 간격으로 1초 뒤 각속도 3축·팩 전압 예측을 표시한다.
6. **비행 종료 · 결과 분석**을 누르면 수신을 종료하고 원본을 보존한 뒤 run_health 분석을 실행한다. 결과는 같은 카드에 자동 표시되고 재시작 후에도 남는다.
   착륙·시동 해제 명령은 보내지 않는다. 자동 disarm 종료는 현재 구현하지 않았다.

QGC가 이미 같은 USB 장치를 사용한다면 장치를 중복으로 열지 말고, 별도 링크 또는 라우터를 사용한다.
MAVProxy를 사용하는 경우의 예시는 다음과 같다. 장치와 baud는 실제 설정으로 바꾼다.
MAVProxy 자체는 이 앱에 포함하지 않았다.

```bash
mavproxy.py --master=/dev/ttyACM0 --baudrate 115200 --out 127.0.0.1:14550 --out 127.0.0.1:14560
```

QGC는 14550, 이 앱은 14560으로 연결한다. 이 예시는 공식 forwarding 방식에서 포트를 하나 추가한 구성이다.

## AI 입력에 필요한 메시지

| 메시지 | 사용 값 | 설정/조건 |
|---|---|---|
| HEARTBEAT | autopilot, armed | 선택 system/component와 PX4=12 또는 ArduPilot=3 일치 |
| ATTITUDE | roll/pitch, body p/q/r | rad, rad/s |
| ATTITUDE_TARGET | 목표 body p/q/r | rate ignore mask가 해제되어야 함 |
| HIGHRES_IMU 또는 SCALED_IMU/2/3 | body FRD 가속도 | HIGHRES m/s², SCALED milli-g→m/s². HIGHRES IMU ID는 별도 선택 |
| LOCAL_POSITION_NED | 실제 vz | NED 아래 양수 → 상승 양수 변환 |
| POSITION_TARGET_LOCAL_NED | 목표 vz | LOCAL_NED 좌표계, vz ignore mask 해제 |
| SERVO_OUTPUT_RAW | 모터 출력 채널 | 프로파일의 port·채널·출력 범위로 0~1 정규화 |
| BATTERY_STATUS | 팩 전압·전류 | 프로파일 battery ID, mV/centiamp 변환, unknown sentinel 제외 |

빠른 신호는 기본 최대 200ms, 배터리는 350ms만 유지한다. 통신 시험에서는 빠른 신호 20Hz,
배터리 5Hz, HEARTBEAT 2Hz로 구성했다. 이는 링크별 대역폭과 실제 수신률 확인을 대체하지 않는다.

Onboard 모드만 선택했다고 모든 목표값이 유효해지는 것은 아니다. 지상 상태·비행 모드·펌웨어에 따라
목표 상승률 또는 목표 각속도가 없거나 무효 마스크가 설정될 수 있다. 이때 연결은 성공해도 현재
common19 모델은 예측하지 않는다. 주기 설정은 없는 센서나 무효 목표값을 만들어내지 않는다.

자동 설정은 `MAV_CMD_SET_MESSAGE_INTERVAL`(511)과 `MAV_CMD_GET_MESSAGE_INTERVAL`(510)만 사용한다.
대상 system/component와 HEARTBEAT의 UDP peer를 고정하고, 시리얼도 같은 명령 경로를 사용한다.
ACK 성공만으로 완료 판정하지 않는다. 조회값이 요청 주기의 ±20% 이내이며 최근 실제 수신률이
요청률의 70% 이상이어야 각 메시지를 ‘확인됨’으로 표시한다. 이는 통신 상태 기준이며 모델 오차 허용치가 아니다.
거절은 해당 메시지에 표시하고 계속 진행한다. ACK 시간 초과는 다음 요청과 혼동하지 않도록 요청 묶음을 중단한다.
HEARTBEAT 유실·peer 변경·연결 종료 시 요청도 중단한다. 재요청은 버튼으로 명시적으로 시작한다.
기체/GCS/router가 주기를 다시 바꾸거나 요청을 전달하지 않으면 미확인으로 남는다.
요청·대상·메시지 ID는 원본 기록에, 최종 설정 확인은 `.rates.json`에 남긴다.
종료 후에도 기체가 설정 주기를 유지할 수 있으며 앱은 이전 주기를 임의로 복원하지 않는다.

## 시간축과 결과 해석

- 실시간은 **수신 monotonic 시각**으로 10Hz 인과 표본을 만든다. 기록된 source 시각의 중복은 제외하고 역행/재연결은 이력을 분리한다.
- 전송 지연·서로 다른 원격 시계의 offset/drift를 자동 추정하지 않는다. 오프라인 ULog/BIN의 근거 있는 시계 보정 경로와 구분한다.
- 수신이 끊기면 예측을 숨긴다. 재연결 후 새 연속 이력을 모은다.
- 시동 해제 자동 종료는 로컬 AI 설정에서 선택한다(기본 꺼짐). 대상 FC의 armed를 관측한 후 fresh disarmed HEARTBEAT 3개가 2초 이상 이어져야 한다. 수신기를 닫고 원본을 확정한 뒤 비행 후 분석을 시작한다. 기체를 disarm시키는 제어 명령은 보내지 않는다.
- 종료 후에는 11초 이상 완전한 구간만 분석한다. 긴 연속 구간은 최대 1000초 단위 패키지로 균등 분할하고 각 조각에서 초기 상태를 다시 만든다. 부분 구간 결과를 전체 비행 건강 인증으로 합치지 않는다.
- 프로파일·가중치·scaler는 고정한다. 건강 미확인 비행을 정상 파인튜닝 자료로 자동 채택하지 않는다.
- 실시간 한도는 2시간 또는 변환 메시지 500000개다. 한도 도달 시 AI 작업은 오류로 표시하고 기존 수신기의 원본 기록은 유지한다.
- 기체별 보정값이 일치하면 **정상 범위 / 이상 의심**을 표시하고, 없거나 입력이 부족하면 **판정 불가**로 표시한다. 실시간은 1초 뒤 실측이 도착한 예측만 판정한다.
- 실시간 수신 시계와 온보드 로그의 보정값은 구분한다. 기체 버전·가중치·scaler·전처리·입력 유지 시간·IMU 선택이 달라지면 같은 보정값을 자동 적용하지 않는다.
- 완료된 실시간 구간을 준비된 자료로 등록할 수 있다. 정상 확인 근거를 직접 입력해야 보정에 쓸 수 있으며 원본과 원본 해시는 보존한다. 고장 원인·SOH/RUL 확정 기능은 아니다.

## 이번 실행 증거

`ai/reports/local-real-flight-20261114/px4-link-attempt-02/result.json`:
기존 PX4 SIH 실행 파일과 실제 로컬 UDP 연결, HEARTBEAT/7종 핵심 메시지 수신, disarmed 유지,
종료·자체 프로세스 반납 확인. 목표 상승률이 유효하지 않아 완전 구간 0개, 분석 보류가 정상 동작했다.
첫 실행자료 경로 실패는 `px4-link-attempt-01/`에 보존했다.

실제 USB 비행제어기·무선 링크·비행 중 입력 적격성·예측 성능은 아직 검증하지 않았다.

`ai/reports/local-real-flight-20261114/ardupilot-link-attempt-01/result.json`:
공식 ArduCopter 4.7.2 SITL 실행 파일을 `ai/.local/flight-link-tools/ardupilot/`에만 설치했다.
UDP 수신·HEARTBEAT 34개·disarmed 상태·종료/반납 확인. 그룹 스트림 설정에서 RAW_IMU와
SCALED_IMU2/3은 수신되었으나 선택한 SCALED_IMU 및 목표값/배터리 메시지가 없어 분석 보류.
따라서 ‘ArduPilot 연결 통과’와 ‘전체 AI 입력 연결 검증 대기’를 분리한다.


### 주기 자동 설정 후속 시험 (RC8)

`px4-auto-rates-rc7/result.json` (RC8과 같은 주기 요청 구현): PX4 SIH, disarmed, 7종 입력은 ACK/설정 조회/수신률 확인.
HEARTBEAT 변경은 ACK=4로 거절되었으며 기존 약 2Hz 수신은 유지됐다. 목표 상승률 무효로 분석 보류.

`ardupilot-auto-rates-rc7/result.json` (RC8과 같은 주기 요청 구현): ArduCopter 4.7.2, disarmed, 8종 ACK/설정 조회 성공.
실수신은 HEARTBEAT·ATTITUDE·ATTITUDE_TARGET·SCALED_IMU2·LOCAL_POSITION_NED·SERVO_OUTPUT_RAW
6종 확인. POSITION_TARGET_LOCAL_NED·BATTERY_STATUS는 0Hz로 미확인. AI 분석은 보류됐다.
두 시험의 자체 프로세스 종료 확인. 실제 FC/USB·무선·비행 중 검증과 구분한다.
