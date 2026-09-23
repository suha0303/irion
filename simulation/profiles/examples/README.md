# 다른 기체의 공학 예제

두 구성은 프로파일·헤더·Gazebo 형상·PX4 제어 할당을 연결하기 위한 **공학 시험 fixture**다. 실제 비행 검증, 특정 제품 재현, 권장 배터리 사양이나 비행시간 보장이 아니다. 기존 `x500_reference`와 `collection` 설정은 변경하지 않는다.

| 구성 | 형상·질량 | 배터리 | 목적 |
|---|---|---|---|
| `quad_example/` | X500과 같은4축 위치, body2.3928kg + rotor4개 | 4S4P·12Ah | 기존 형상과 같은 기하를 새 프로파일 경로로 표현 |
| `hex_example/` | 반경0.28m의6축, body3.2kg + rotor6개 | 6S2P·6Ah | 모터 수·할당·셀 수 변경의 일관성 검증 |

각 폴더의 `vehicle.yaml`은 `propulsion.parameters_file: propulsion.json`과 `battery.parameters_file: battery.json`으로 추진·배터리 정의를 연결한다. SOC축은0..1의31점이며 `ocv_cell_v`는 **셀 하나의 전압**이다. 두 예제는 기존 CMU/VTC6 기반 셀 곡선을 재사용하고 Ns/Np와 팩 R/C·열 조건을 명시했다. 임의 시간 배율이나 배터리 용량 배율로 비행시간을 늘리지 않는다. Hex 예제의 모터·프로펠러와 열 조건은 별도 실측이 없는 가정이다.

지원 범위와 작성 방법은 [기체 프로파일 안내](../../../docs/simulation/VEHICLE_PROFILES.md)를 따른다. 현재 pinned PX4 Gazebo ESC 출력은8개이므로 고정 수직축 multirotor4..8개를 지원한다. 기울임 모터·가변 피치·고정익·VTOL은 이 프로파일로 표현하지 않는다.
