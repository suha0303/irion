# 출처와 적용 범위

문서 버전: 4.2 · 확인일: 2026-10-13

## 사용자 지정 스킬 원문

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

스킬의 선택 근거는 사용자가 직접 지정한 네 링크다. 이 패키지는 원문 링크와 적용 지시를 제공한다. 각 원문의 상세 절차는 해당 원문에서 확인한다.

원문 확인 당시 `anti-slop`의 자동 검사 도구는 TypeScript/JavaScript와 Oxlint를 대상으로 한다. 해당 언어를 사용하는 구성 요소에서 원문 도구를 적용하고, C++·Python 구성 요소는 그 언어에 맞는 검사 도구와 실제 시험으로 검증한다. 적용한 원문 범위와 검사 도구를 결과에 기록한다.

## 자료를 사용하는 방법

아래 기술 URL은 구현 시 실제 release와 대조할 참고 자료다. 런타임 버전은 대상 환경에서 설치·빌드·시험한 결과로 lock에 기록한다. [S]·[A] 번호는 각 구현 명세 본문의 출처 식별자다.

## 물리·시뮬레이션

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

## 데이터와 AI

[A1] PX4 메시지 문서: https://docs.px4.io/main/en/msg_docs/
[A2] ArduPilot 로그 메시지: https://ardupilot.org/copter/docs/logmessages.html
[A3] DataFlash parser: https://github.com/ArduPilot/pymavlink/blob/master/DFReader.py
[A4] ULog parser: https://github.com/PX4/pyulog
[A5] PyTorch LSTM: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
[A6] PyTorch TransformerEncoder: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
[A7] PyTorch 재현성: https://docs.pytorch.org/docs/stable/notes/randomness.html
[A9] 시계열 이상 탐지 평가 검토: https://arxiv.org/abs/2109.05257
[A10] Dynamics randomization 연구: https://arxiv.org/abs/1710.06537

각 자료는 해당 기능·방정식·API의 근거로 사용한다. 공개 모델의 존재, 소프트웨어 시험 통과, 실제 기체 정확도는 각각 별도의 증거로 관리한다. 문서의 초기 하이퍼파라미터·실험 범위·수치 허용오차는 이 프로젝트의 설계 제안이다.


## v4 정상 예측·열화 해석 검토에 추가한 출처

[A11] LSTM 예측오차 기반 이상탐지 (Hundman et al., 2018): https://arxiv.org/abs/1802.04431

[A12] 데이터 누출·전처리 fit 경계 (scikit-learn): https://scikit-learn.org/stable/common_pitfalls.html

[A13] 그룹/시계열 분할 (scikit-learn): https://scikit-learn.org/stable/modules/cross_validation.html

[A14] 짧은 배터리 응답과 건강상태 기준을 사용하는 실험 연구: https://www.nature.com/articles/s41467-024-54454-0

[A15] PX4 배터리 부하 보상·전류 적분의 의미: https://docs.px4.io/main/en/config/battery

[A16] PX4 고속 IMU 기록·필터/진동 분석: https://docs.px4.io/main/en/config_mc/filter_tuning

[A11]은 센서값 예측과 잔차 평가를 분리하는 근거다. 논문의 threshold를 이 프로젝트에 그대로 이식하거나 배터리·모터 열화 검출 성능의 증거로 삼지 않는다.
[A12,A13]은 fit/transform 분리와 독립 그룹 평가의 근거다. 이 프로젝트의 구체 asset 계보·chronological protocol은 본 패키지 설계다.
[A14]는 조건과 건강상태 정답을 갖춘 짧은 배터리 반응 연구다. ‘임의 비행 5초면 SOH 추정 가능’ 또는 ‘정상 예측오차가 SOH’라는 근거가 아니다. 정확한 펄스 길이/성능을 이번 구현 요구로 전용하지 않는다.
[A15]는 부하·온도·저항과 전압/잔량 추정의 혼동을 점검하는 데 사용했다. 실제 SOC와 관측 전류 적분은 같지 않다.
[A16]은 고속 진동 정보와 10Hz 공통 반응 진단의 범위가 다름을 확인하는 참고다.

v4 작성 당시 위 추가 출처와 네 스킬 원문/적용 파일을 2026-10-13에 확인한 기록을 승계한다. v4.2에서도 [S1–S11,A1–A7,A9–A16]의 식별자와 URL을 유지한다. [A2,A3]의 DataFlash/ArduPilot과 [A10]의 이전 관련 참고는 현재 X500 필수 구현을 넓히는 요구가 아니다. 이번 범위 수정에서 물리/AI 기술 출처를 새로 검증하거나 release/API를 빌드한 것은 아니다. ‘main/stable/latest’ 문서는 변할 수 있으므로 실제 구현 시 선택 release/commit을 다시 확인한다.

자문 형태의 이전 대화에 등장한 다른 논문은 이번 명세의 검증 근거로 자동 승계하지 않았다. 이번 설계의 검출률·최적 window·SOH 정확도·RUL 정확도에 대해 아직 측정 결과는 없다.

## v4.2 작업 지침 접근 기록

2026-10-13 이번 수정에서 attention-span README, andrej-karpathy-skills의 CLAUDE.md, verification-before-completion 원문을 조회했다. attention-kind 적용 파일과 anti-slop 원문은 재접근 오류로 새 본문을 확인하지 못했다. 기존 링크와 앞선 적용 범위 기록을 보존했으며 네 원문 모두를 이번에 새로 읽었다고 주장하지 않는다. 원문 사본이나 설치 도구는 추가하지 않았다.
