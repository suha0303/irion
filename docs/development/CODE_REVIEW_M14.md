# M14 — 시뮬레이션 실패 22개 원인과 수정

2026-11-15. M13의 22개 실패를 사례 ID로 추적했고, **기존 255개 + 새 6개,
합계 261개가 한 번의 최종 전체 실행에서 통과**했다. skip/xfail은 없다.
시뮬레이션 실행 코드·정책·원래 시험·감사 패키지·생산자 봉인은 그대로다.

## 원인

| 기존 실패 | 원인 | 수정 위치와 검증 |
|---|---|---|
| 13개 | runner는 `sync_interval_s`를 전달하고 기록하지만, 봉인된 RecorderFixture는 그 인수를 받지 못함 | 시험용 확장 fixture가 인수와 값을 보관; 원래 실행/정리 검증을 그대로 수행 |
| 2개 | 40 ms 기준으로 작성된 duration 시험이 현재 3 s 정책을 읽음. 만든 60 ms 공백은 현재 정책에서는 허용됨 | 해당 역사 시험에만 임시 40 ms 정책 지정; 별도 시험에서 현재 3 s 정책의 60 ms 허용과 참조행 누락 거절 확인 |
| 7개 | 감사 코드는 `joint-native-consumption-c2-v1`, 시험의 generator import는 현재 v2. receipt revision과 duration 필드가 섞임 | 감사 manifest에 고정된 v1 generator/정책을 복원하여 원래 7개 시험 실행 |

실패는 이 범위에서 현재 시뮬레이터 물리나 실제 비행 실패를 뜻하지 않았다.
반대로 시험용 의존성이 어긋나 실질적인 검사가 실행되기 전에 멈추거나,
다른 정책에 대해 옛 기대값을 적용하고 있었다.

## 보존 방식과 현재 시험 환경

`simulation/conftest.py`와 `simulation/test_support/contracts.py`가 파일 경로를
명시한 시험에만 의존성을 연결한다. 통과하도록 verifier·assertion·오류
처리를 교체하지 않는다. pytest 바깥에서는 이 연결이 실행되지 않는다.

- RecorderFixture의 기존 생성/종료 동작을 상속하고 새 설정값만 기록한다.
- duration 시험은 임시 폴더에서 40 ms 정책의 합성 run을 만든다. 현재
  `profiles/collection/joint-phase-policy.json`은 수정하지 않는다.
- 과거 감사용 generator와 정책은 보존된 `offline-source`에서 바이트 그대로
  복사하고, 로드 전에 원래 audit manifest의 SHA-256과 대조한다.
  복사본은 `simulation/test_support/frozen_joint_v1/`에 역사 의존성으로 둔다.
  기존 감사 패키지 안의 시험을 포함한 모든 파일은 변경하지 않는다.

새 시험 6개는 40 ms 거절/3 s 허용/참조 coverage 부족 거절, 기록기 설정과
정리, v1/v2 구분, 역사 의존성 변조 차단을 확인한다. 시험 환경 연결을
제외하고 봉인된 옛 시험만 직접 실행하면 원래 의존성 불일치가 재현될 수
있다. 현재 명령은 pytest가 `simulation/conftest.py`를 읽는 표준 실행이다.

## 검증과 최초 실패 보존

첫 전체 실행은 **254 통과·7 실패**였다. 복원한 v1 파일의 상대 import가
패키지명 없이 로드되어 생긴 실패다. 로더에 `capstone_sim` 패키지 범위를
지정한 뒤 영향 범위 13개가 통과했고, 최종 전체 **261 통과, 21.53초**를
확인했다. 기존 실패 22개 각각이 최종 JUnit에서 통과했는지도 대조했다.

생산자 source/runtime 검증도 통과했다. 초기 점검 명령이 상대 루트를 넘겨
SONAME 절대 경로 비교에서 실패한 기록은 별도로 남겼고, 실제 호출부와
같은 절대 루트로 확인했다. 라이브러리나 봉인을 다시 생성하지 않았다.
물리 플러그인·PX4 바이너리·기존 생산자/런타임 해시는 동일하다.

기존 모델 전달물·준비된 실제 로그·기존 결과·보존 소스를 포함한
**29,427개 파일의 전후 해시가 동일**하다. 원래 실패/성공 증거도 보존했다.
이번 결과는 CPU 회귀 검사이며 새 SITL 비행·수집·실기체 검증은 아니다.

로컬 증거:

- `ai/reports/failure-investigation-20261115/m14/simulation-final.xml`
- `ai/reports/failure-investigation-20261115/m14/resolved-failures.json`
- `ai/reports/failure-investigation-20261115/m14/runtime-verification.json`
- `ai/reports/failure-investigation-20261115/preservation-before.json`
- `ai/reports/failure-investigation-20261115/preservation-final.json`

실행 명령은 [TESTING.md](TESTING.md), 전압 추적은 [M15](CODE_REVIEW_M15.md)를
참고한다. M13의 22개 실패 기록을 소급해서 통과로 고치지 않는다.

GitHub main 소스: `44001f4`(시뮬레이션 시험 연결), `884ad11`(전압 진단).
[공개 CI](https://github.com/suha0303/irion/actions/runs/43663656866) 통과. 공개 clone Data 253개 통과·로컬 자료 의존
86개 제외, 프런트엔드 26개·빌드·포맷·Ruff 통과. 로컬 전체 Data 327개에서는
제외하지 않았다.
