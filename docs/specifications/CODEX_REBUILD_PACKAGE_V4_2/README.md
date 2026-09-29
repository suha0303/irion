# X500 시뮬레이션·AI CODEX 전달 패키지 v4.2

문서 버전: 4.2 · 수정일: 2026-10-13 · 직접 원본: `CODEX_REBUILD_PACKAGE_V4_1.zip`

**현재 범위를 X500 한 기준 구성으로 확정한 검토용 최종 수정본이다.** 모델·물리·전처리의 핵심 설계는 유지하고, 회사 기체 이전·다기체 사전학습·기종 간 검증을 현재 실행과 완료 조건에서 분리했다. 이전 버전을 현재 명세와 동시에 적용하지 않는다.

## 현재 범위

<!-- CURRENT-SCOPE-BEGIN -->
**현재 단계의 개발·학습·평가 대상은 X500 기준 기체 한 구성으로 한정한다.** 하나의 정상 비행만 사용한다는 뜻은 아니다. 같은 X500 기준 구성에서 건강한 모터·배터리·프로펠러·ESC·센서로 다양한 허용 기동·바람·탑재량·잔량·온도 조건의 정상 비행을 생성한다.

**이번 산출물은 X500 전용 시계열 예측·이상탐지 모델과 실행 가능한 전체 흐름이다.** 시뮬레이션 → 원시 기록 → 전처리 → LSTM·Transformer 미래 예측 학습 → 경보 → 독립 평가를 연결한다. 배터리·추진계통의 기존 열화 상태와 비행 중 이상을 구분해 평가하며, 같은 X500/부품의 비행 간 집계는 기존 설계 범위에서 유지한다.

**다양한 실기체에 커스터마이징 가능한 파운데이션 모델은 장기 목표다.** 다기체 데이터 수집·공통 사전학습·가중치 이전·기체별 미세조정·회사 기체 적용·기종 간 일반화 검증은 현재 필수 구현 및 완료 조건에서 제외한다. 다른 기체·모터 수를 지원하는 범용 모델이나 확장 프레임워크를 미리 구현하지 않는다.

X500의 기체·센서 설정, 전처리, 모델 코드는 기존의 단순한 경계로 분리한다. 후속 확장을 위한 빈 adapter·추상 계층·미세조정 도구를 추가하지 않는다. 새 기체 지원은 사용자의 별도 요청 후 범위와 검증 기준을 정한다.

현재의 필수 실행 검증은 X500 PX4 SITL 기준이다. 실측 X500 검증과 다른 실기체 검증은 자료·장비가 필요한 별도 단계로 표시하며, 시뮬레이션 결과를 실기체 성능이나 파운데이션 모델 완성으로 보고하지 않는다. 현재 범위 밖 작업은 `not_in_scope`로 분리하고 미완료 필수 항목에 합산하지 않는다.
<!-- CURRENT-SCOPE-END -->

## 검토와 전달 파일

| 파일 | 역할 |
|---|---|
| [00_SCOPE_AND_REVIEW.md](00_SCOPE_AND_REVIEW.md) | 사용자 검토용 현재 범위·유지 조건 요약 |
| [01_SIMULATION_CODEX_SPEC.md](01_SIMULATION_CODEX_SPEC.md) | X500 PX4·물리·원시 기록·정상/이상 campaign·AI 연결 |
| [02_AI_PREPROCESSING_CODEX_SPEC.md](02_AI_PREPROCESSING_CODEX_SPEC.md) | X500 파싱/전처리·예측 학습·경보·독립 평가·같은 부품 추적 |
| [00_SHARED_DATA_CONTRACT.md](00_SHARED_DATA_CONTRACT.md) | 양쪽의 현재 범위·관측/정답/시각 계약 |
| [03_FINAL_REVIEW.md](03_FINAL_REVIEW.md) | 이번 변경 검토와 실제 문서 검사 범위 |
| [04_CHANGELOG_V4_1_TO_V4_2.md](04_CHANGELOG_V4_1_TO_V4_2.md) | 이번 범위 정리의 변경 위치 |
| [04_CHANGELOG_V3_TO_V4.md](04_CHANGELOG_V3_TO_V4.md) | 과거 이력만 보관. 현재 구현 지시가 아님 |
| [00_SKILL_LINKS.md](00_SKILL_LINKS.md), [AGENTS.md](AGENTS.md) | 사용자 지정 네 원문과 작업 책임 |
| [00_SOURCE_NOTES.md](00_SOURCE_NOTES.md) | 기존 기술 참고·적용 범위·원문 접근 기록 |
| [checks/verify_package.py](checks/verify_package.py) | 재실행 가능한 문서/예제/무결성 검사기 |
| [checks/validation_report.json](checks/validation_report.json) | 이번 문서 검사 실행 결과 |
| [checks/requirements_trace.json](checks/requirements_trace.json) | 현재 요구와 후속 범위의 문서 위치 |
| [checks/baseline_provenance.json](checks/baseline_provenance.json) | 직접 원본 대조와 실제 보존 구간 hash |
| [MANIFEST.json](MANIFEST.json) | 최종 파일별 크기·SHA-256 |

두 주요 명세는 단독 전달용으로 현재 범위·스킬 링크·공통 계약·참고자료를 포함한다. 검토할 때는 범위 요약 → 이번 변경 내역 → 각 담당 명세 순서로 읽으면 된다.

## 무엇을 학습하고 무엇을 출력하는가

건강한 X500에서 과거 명령·부하·반응으로 이후 센서값을 예측한다. 정답 Y는 같은 정상 비행의 이후 실측이다. 기본 학습은 정상 자료만 사용하고, 정상 validation/calibration과 독립 정상·이상 시험을 분리한다. 같은 X500의 여러 독립 비행으로 이 분할을 구성할 수 있다.

새 비행에서 예측을 불변 저장하고 이후 실측 도착 후 실제 시각에 오차·경보를 발행한다. 출력은 배터리/추진 응답의 이상 구간·지속·근거이며, 특정 고장 원인·SOH·RUL 또는 건강 인증이 아니다. 같은 X500/부품의 비교 가능한 비행은 별도 계층에서 집계한다.

첫 설정은 10Hz, 과거 100시점, 미래 10시점, common19_ctx6, 측정 각속도 3개와 팩 전압 target이다. ctx6에는 60초 통계와 segment 누적량이 있다. 목표값이 실제로 부족한 경우의 common15, ctx0 비교는 선택 경로이며 모든 조합을 학습할 필요는 없다. 수치는 검증 전 출발값이다.

개별 RPM·개별 모터 전류·모터/배터리 온도·개별 셀 전압·BAT.Res·내부 truth를 기본 입력에 추가하지 않는다. 기존 내부 물리 열 모델과 처음부터 열화된 상태/비행 중 이상 시험은 유지한다.

## 최우선 적용 원문

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

## 시뮬레이션 담당에게 전달

```text
첨부한 v4.2의 01_SIMULATION_CODEX_SPEC.md를 기준으로 구현해줘.
이번 대상은 X500 기준 한 구성과 선택 배터리 profile 하나야.
파운데이션 모델·회사 기체·다기체 지원은 장기 목표이지 지금의 구현 과제가 아니야.
기존 물리·센서·바람·PX4 연결을 바탕으로 건강한 X500의 다양한 정상 비행을 생성해줘.
배터리/추진계통의 시작부터 열화된 상태와 비행 중 이상은 별도 평가 자료로 만들어줘.
원시 관측·정답·run/paired/retry 계보를 분리하고 실제 물리 반응 연결을 검증해줘.
전처리는 AI 담당으로 유지하고 공통 계약의 X500 연결 시험까지 완료해줘.
실제 실행한 명령·환경·결과를 보고하고 범위 밖 작업은 not_in_scope로 구분해줘.
```

## AI·전처리 담당에게 전달

```text
첨부한 v4.2의 02_AI_PREPROCESSING_CODEX_SPEC.md를 기준으로 구현해줘.
이번 대상은 X500 PX4의 DRONE-RAW/ULog와 같은 자료의 replay/live야.
회사 BIN/다기체 이전·공통 사전학습·미세조정 도구를 새로 만들지 마.
기본은 LSTM/Transformer 미래 예측이며 기존 persistence/ridge 예측 baseline을 비교해줘.
같은 X500의 독립 run으로 정상 train/validation/calibration과 독립 정상·이상 test를 분리해줘.
미래 실측을 forward 입력에 넣지 말고 예측 저장 후 실측이 도착한 시각에 채점해줘.
처음부터 열화된 상태·급성 이상·정상 오탐·coverage/지연과 한계를 보고해줘.
같은 X500/부품의 비행 간 집계는 유지하되 SOH/RUL·실기체 성능을 단정하지 마.
검증 결과는 실제 실행 증거로 보고하고 장기 목표를 현재 미완료 항목으로 합산하지 마.
```

## 문서 검사 재실행

패키지 루트에서 기존 검사 의존성을 설치한 뒤 실행한다.

```bash
python -m pip install -r checks/requirements.txt
python checks/verify_package.py
python checks/verify_package.py --zip ../CODEX_REBUILD_PACKAGE_V4_2.zip
```

검사기는 문서·예제 산술·참조·hash를 검사한다. 실제 PX4/SITL 실행, 실제 신경망 학습, 물리적 검출 성공을 검사하거나 증명하는 실행기가 아니다.
