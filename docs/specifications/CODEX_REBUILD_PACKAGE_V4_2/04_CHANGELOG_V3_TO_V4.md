# 과거 변경 이력 — v3 → v4 → v4.1

문서 버전: 4.2 · 검토일: 2026-10-13

v3 → v4 통합 당시의 기준은 Library에서 확인한 실제 `CODEX_REBUILD_PACKAGE_V3.zip`이었다. 당시 원본 대조 기록은 `checks/baseline_provenance.json`에 보존한다. 이번 v4.1은 대화의 `CODEX_REBUILD_PACKAGE_V4.zip`에서 18절과 관련 비교 과제·참조를 삭제한 수정본이다. 아래 표는 당시 v4.1의 기록이다. 원본 v4는 수정하지 않았다.

> 이 파일은 과거 변경 기록이며 현재 구현 지시가 아니다. 회사 기체 이전·다기체/새 개체 평가 등 아래의 당시 범위는 v4.2의 현재 X500 범위에 의해 대체된다. 현재 변경은 [04_CHANGELOG_V4_1_TO_V4_2.md](04_CHANGELOG_V4_1_TO_V4_2.md)를 따른다.

## 당시 수정·명확화한 항목

| ID | 문제/모호한 부분 | 반영한 내용 | 위치 |
|---|---|---|---|
| R01 | 학습 과제와 구현 경로의 불일치 | forecast 학습 과제로 통일. decoder/target/loss/scorer/CLI/bundle을 같은 미래 예측 계약으로 맞춤. | AI 1.1, 8~12 |
| R02 | 정상 비행 = 부품 건강이라는 오해 | confirmed_healthy/uncertain/abnormal과 선정 근거. 모터·배터리·프로펠러·ESC·센서까지 건강 기준을 관리. | SIM 9.2, 11.2 / AI 7.2 / C13 |
| R03 | 정상 다양성과 열화가 섞일 위험 | 환경/제작 편차와 검출 대상 건강 파라미터를 분리. 건강한 저잔량/고부하/허용 저온 대조 포함. | SIM 9.2, 11.3 / AI 7.2, 13.6 |
| R04 | 고장 라벨 없음 = 학습 정답 없음이라는 오해 | X는 과거 관측, Y는 정상 로그의 이후 센서값. 정상 선정·평가 정답과 분리. | AI 6.3, 7.2 |
| R05 | 입력 19개와 예측 출력의 불명확함 | 입력 common19/15 유지. 첫 target은 실제 각속도 3개+팩 전압 1개. 전류는 입력이고 기본 forecast target 아님. | AI 4.1, 6.3.1, 10.1, 11.1 |
| R06 | 10초만 관측한다고 한 설명과 60초 문맥의 충돌 | 기본은 100시점 history+60초 통계+segment 누적량. ctx0도 필터 state가 있음을 표시. | AI 6.2~6.4 / C14 |
| R07 | 온도/셀 센서를 다시 요구할 위험 | 기본 제외 신호와 truth allowlist 명시. 팩/Ns는 셀당 평균 환산값이며 개별 셀 측정 아님. | SIM 9.2 / AI 4.1, 6.1 / C11 |
| R08 | 예측오차가 열화율 또는 건강 증명으로 바뀌는 문제 | 낮은 오차도 건강 인증 아님. SOH/RUL·특정 모터 원인 확정은 별도 검증 과제. | AI 1.1, 12.3, 13.6~13.8 |
| R09 | 미래의 실제 전류·명령 누출 | origin cutoff, teacher forcing 차단, 불변 forecast. 새 실측을 사용한 진단은 별도 conditional_response task. | AI 8~9, 12.2, 13.8 / C14 |
| R10 | 미래 예측과 고장 사전 경고 시간 혼동 | forecast 발행과 target 도착 후 decision을 분리. late/pending/미채점·필터/수신/계산 지연 기록. | AI 6.3.2, 12.2~12.3 |
| R11 | 미래 target과 관계없는 loss·점수 잔존 | forecast target loss와 lead별 정상 error scale을 사용. context는 조건 입력으로만 사용. | AI 10.1, 11.1 |
| R12 | 정상 데이터만으로 검출 성능까지 평가하는 오해 | 정상 train/validation/calibration, 독립 정상 test와 이상 test. test를 개발에 쓰면 새 test 필요. | AI 7, 13 |
| R13 | 그룹 분할과 동일 부품 장기 추적의 충돌 | unseen_asset과 same_asset_future protocol을 분리. 모든 profile 공유를 동일 개체로 묶지 않음. | AI 7.1, 13.7 / C6,C13 |
| R14 | 이미 열화된 비행을 급변 감지로만 평가 | pre_existing R0/Q/추진 손실 시험과 in_flight step/ramp를 분리. 관측이 겹치는 사례도 남김. | SIM 11.3 / AI 13.6 |
| R15 | pre-existing의 실제 노화 발생시각을 t=0으로 오인 | 적용 시각과 physical onset을 분리. unknown onset은 null, 인지시간은 관측/첫 판정 기준. | AI 13.2 / C6,C13 |
| R16 | 짧은 창으로 모든 capacity/고주파 결함 검출 보장 | 용량 감소는 충분한 방전 관측을 별도 시험. 10Hz로 고주파 베어링 진단을 보장하지 않음. | AI 4.1, 13.6 / SIM 11.3 |
| R17 | 비행별 점수 상승 = 동일 부품 열화라는 오해 | 실제 battery/assembly ID·교체·운용 조건·bundle revision을 매칭. 정보 부족 시 추적 보류. | AI 13.7, 14 / C13 |
| R18 | 누적 열화를 정상으로 재학습하거나 정규화로 제거 | train scaler 고정, 비행/윈도별 재정규화·무검증 온라인 적응 기본 해제. 변경은 건강 확인+revision. | AI 6.4, 14 |
| R19 | 시뮬레이션 severity sweep = 실제 노화율/RUL 검증 | 여러 고정 열화 수준과 실제 비행횟수/수명 곡선의 증거를 구분. 물리·센서 연결은 별도 시험. | SIM 11.3.2, 13.2 / AI 13.7 |
| R20 | 문서/JSON 검사 = 소프트웨어/검출 성공이라는 오해 | 문서 검사기를 패키지에 제공. 원본 대조·예시 산술·무결성과 실제 SITL/AI/실기체 시험을 분리. | 03_FINAL_REVIEW / checks / C12 |

## 유지한 기존 설계

PX4 유지와 X500 기준, 공개 자료/공학 가정, 평균 전동계·1RC 배터리/열·바람·센서, 기체 프로파일 확장, 시뮬레이션/AI 책임 분리, 원시 관측·truth 분리, 인과 전처리, 10Hz common19/common15, 정상 fit/calibration, 독립 평가, CLI·replay/live·회사 이전·스킬 네 원문을 유지했다.

당시 시뮬레이션 2~8절 및 10절, AI 2~3절·5절·10.3절·11.3절 본문을 v3와 대조했다. 이는 v4.1 작성 당시의 이력이다. v4.2의 실제 직접 기준과 보존 범위는 `checks/baseline_provenance.json`에 다시 기록했다. 다른 절은 이번 학습 과제와 평가 변경에 맞춰 수정했으므로 ‘전체 기술 본문 무변경’이라고 보고하지 않는다.

## v4 → v4.1 삭제 수정

사용자 요청에 따라 AI 명세의 18절 전체를 삭제했다. 다른 절·README·AGENTS·공통 계약의 해당 비교 과제 유지 지시, 전용 출처, 요구사항 항목도 제거했다. LSTM·Transformer 예측 구조와 기본 forecast 설정, persistence/ridge baseline, 물리 모델·전처리·열화 검증 기준은 유지했다. 관련 참조·검사기·파일 hash와 ZIP을 함께 갱신했다.

## 의도적으로 바뀐 기본값과 새 비교 설정

기본 task는 forecast다. 예측 target 4개, 10시점 horizon, context는 조건 입력, 입력 noise 기본 0, persistence/ridge forecast 비교, pending forecast 저장과 output schema 3.0을 유지한다. 이는 현재 대화의 미래 예측 설계를 구체화한 **검증 전 출발 설정**이며 실험상 최적이라는 뜻은 아니다.

raw 관측·watermark 2.0은 유지하고 evaluation event는 2.1 확장이다. 실제 producer/consumer의 기존 버전 호환성은 구현 시험이 필요하다. 두 주요 명세의 공통 계약 사본을 함께 갱신했다.

이번 변경은 문서·패키지 작업이다. 사용자 저장소·전역 스킬 설치·PX4 코드·가중치를 직접 변경하거나 실제 비행 성능을 측정하지 않았다.
