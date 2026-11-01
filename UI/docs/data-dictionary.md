# 표시용 데이터 사전

기준은 제공된 `ai-telemetry-contract.md`의 arion_cmu_v1 계약입니다. 새 센서·새 기체에 이를 무조건 적용하지 않습니다.

| 화면 값 | 원문 | 변환 / 의미 | 유효성 |
|---|---|---|---|
| 팩 전압 V | BATTERY_STATUS.voltages[0:4] | 합계 ÷ 1000. 현재 4S 모델의 셀 그룹 | 생산자 arion_cmu_v1, 네 슬롯 모두 유한·0 이상·65535 미만 |
| 팩 전류 A | current_battery | cA ÷ 100 | -1 제외. arion 프로필에서 legacy -100도 제외. 다른 음수를 모두 삭제하지 않음 |
| SOC % | battery_remaining | 충전 잔량. SOH 아님 | 0~100, -1 미지원 |
| 소비량 mAh | current_consumed | 누적 소비. 정격 용량 아님 | 0 이상 |
| 모터 1~4 RPM | ESC_STATUS.rpm[0:4] | 보고 RPM, 실기체 실측 아님 | index=0, 유한값. 정지한 모터의 0은 유효 |
| 모터 1~4 전류 A | ESC_STATUS.current[0:4] | 모델에서 생성한 전류 | index=0, 유한값 |
| 동·북 위치 m | LOCAL_POSITION_NED.y / x | 로컬 좌표, GPS 위·경도 아님 | 유한값 |
| 상대 높이 m | 동일 epoch의 첫 유효 z - 현재 z | NED의 아래 방향 양수 부호 반영. 지면고도 보장 없음 | 해당 epoch에 기준 z 필요 |
| 속력 m/s | sqrt(vx²+vy²+vz²) | NED 속도 벡터 크기 | 세 축 모두 유한값 |
| Roll/Pitch/Yaw ° | ATTITUDE roll/pitch/yaw | rad × 180/π | 유한값, 진단에 표시 |
| Arm 상태 | HEARTBEAT.base_mode | bit 128 | 신선한 heartbeat 필요 |
| 비행 모드 | HEARTBEAT.custom_mode | PX4 main/sub mode | autopilot=12인 경우만 PX4 규칙 사용 |
| 지상·비행·이륙·착륙 | EXTENDED_SYS_STATE.landed_state | 1/2/3/4 | unknown을 지상으로 바꾸지 않음 |

온도 25°C는 고정 모델값이므로 열 이상 지표를 만들지 않습니다. actuator 출력은 명령이며 RPM으로 변환해 실측치와 비교하지 않습니다. `ACTUATOR_OUTPUT_STATUS.active=16`을 고장 bitmask나 모터 16개로 표시하지 않습니다. 제어 목표 3종은 수신 진단만 제공하고 모델 입력 채택·모터 물리 대응은 AI/시뮬레이션 계약에 남겨 둡니다.

## 시간·품질

수신 monotonic ns를 원문 그대로 보존합니다. 상대 t_ms와 epoch를 UI에 전달합니다. 미래 보간은 하지 않습니다. 각 시점 이전 최신 메시지만 유지하며 아래 나이를 초과하면 값 대신 null과 stale을 보냅니다.

| 메시지 | 최대 관측 나이 |
|---|---:|
| BATTERY_STATUS | 600ms |
| ESC_STATUS | 250ms |
| ATTITUDE / LOCAL_POSITION_NED | 300ms |
| HEARTBEAT | 1500ms |
| EXTENDED_SYS_STATE | 600ms |
| POSITION_TARGET_LOCAL_NED / ATTITUDE_TARGET | 300ms |
| ACTUATOR_OUTPUT_STATUS | 250ms |

이는 첨부 전처리 설정과 초기 표시 정책입니다. 실기체 센서 SLA가 아닙니다. `valid`, `missing`, `stale`, `invalid`를 구분하며, 메시지가 신선하더라도 특정 필드가 미지원이면 invalid입니다. 1초 초과 공백·수신 시계 역행·같은 메시지의 source 시계 역행에서 epoch를 분리합니다. 서로 다른 메시지의 source 시각 순서를 비교해 재부팅으로 판단하지 않습니다.

## 기준 입력 검사

- 입력 SHA-256: `487299172cb2d519d062565f0ab0117ca5952d4e51b3b86a897ae4519309932e`
- 원문 4,282행 / 유효 4,281행 / 제외 1행 / 재생 약 47.040초.
- 321번째 행: `receive_monotonic_ns` 필드명 손상. 필드명을 추측해 복원하지 않고 제외한다.
- NPZ: 470×19, window 85, data_ready 84, model_ready 0, mask false 14 cells.
- NPZ의 입력 SHA-256은 `dc7aad6e995aaae3ff09a326c87f90dff8f2d672cbd0ae34bab7ad273887ed62`. 위 JSONL과 불일치하므로 자동 결합하지 않는다.
- 기업 BIN 요약에는 실측 모터 RPM·개별 전류가 없으므로 기업 자료에 같은 모터 카드를 실제 측정처럼 채우지 않는다. 실제 BIN은 현재 입력으로 등록하지 않았다.
