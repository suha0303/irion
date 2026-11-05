# VAH01 실측 배터리와 모터 연결 — 구현·검증 보고

적용일: 2026-11-07 KST. 모델 revision: `cmu-vah01-baseline/1`.
현재 비행 정책은 `cmu-vah01-rpt16-only/1`이다. **정상은 rpt_00, 열화는 rpt_16만 사용**한다.
배터리 열화·복합 고장 시나리오는 옵션을 생략해도 rpt_16을 선택하고, 비행 중 변화는
rpt_00에서 rpt_16으로 이동한다. 중간 단계 선택은 실행 전에 거절한다.

2026-11-07 추가 검증: 랜덤 비행의 충돌 경로와 장시간 병렬 종료 시의 메모리 사용을 수정했다.
일반 Python 모드 189개 시험, 반복 경로 검산, 수집 길이의 16개 병렬 비행 및 후처리 검증이
통과했다. 상세 증거는 `simulation/artifacts/endurance-20261107-01/verification.json`이다.
아래 이전의 짧은 비행·진단 모드 시험 기록은 당시 증거로 보존한다.

사용자가 지정한 세 baseline 후보 중 **VAH01 한 셀만 사용**했다. VAH17/VAH27을
추가 보정·검증 셀로 사용하지 않았다. 정상, 배터리 열화, 모터/프로펠러 고장 모두
같은 배터리 코어를 사용한다. 비행에서는 PX4의 명령과 모터 부하로 전류·전압·SOC를
계산한다. 실측 전압 로그 재생은 오프라인 적합·검증에만 사용한다.

## 출처와 선택 근거

[CMU eVTOL Battery Dataset v3](https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830/3),
DOI `10.1184/R1/14226830.v3`의 `VAH01.csv`와 `VAH01_impedance.csv`를 사용했다.
원논문은 [Bills et al., Scientific Data 10, 344 (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10238407/)이다.
논문 Table 2에서 VAH01은 baseline으로 지정되어 있고, 알려진 데이터 문제 목록에
VAH01 문제가 따로 열거되어 있지 않다. 후보 중 첫 baseline을 최종 전압 오차를 보기
전에 선택했다. 이 선택이 세 후보 중 성능이 가장 좋다는 뜻은 아니다.

| 파일 | SHA256 |
|---|---|
| VAH01.csv | `596ab1e1e80eeac3509f789609f751527c2073a4310deec8274cbfaf4fef39c4` |
| VAH01_impedance.csv | `f92a2e6b141d732863fbf5ba09e8867c073f15df2d35c80c83887051a1d31b25` |

공식 v3 metadata, 다운로드 MD5/SHA256, 파일 크기는
`simulation/artifacts/baseline-20261107/{cmu-v3-metadata,retrieval-vah01}.json`에 남겼다.
라이선스는 CC-BY-4.0이다. 이 구현은 원자료에서 별도로 추정한 공학 모델이며 CMU가
제공한 배터리 시뮬레이터나 논문의 전기화학 모델을 그대로 구현한 것은 아니다.

## 보정과 시간순 검증

`battery_fit.py`가 Ns 순서·전류·종단전압·전하 counter로 완전한 RPT와 임무를 찾는다.
원자료 1,091,477행에서 시간 역행은 없었고, 완전한 RPT 18개와 임무 828개를 찾았다.
원자료 RPT를 0부터 순서대로 번호 붙이면 마지막은 rpt_17이다. 이 마지막 RPT 뒤에는
검증할 완전 임무가 없어서 제외했다. 따라서 rpt_16은 후속 임무 검증까지 통과한 마지막 단계다. 최종 사용 단계는 17개다.
마지막 사용 단계는 초기 실측 용량의 약 82.45%다. 자료가 없는 70% 등의 단계를
외삽하지 않는다. 행 범위와 제외 이유는 catalog에 기록한다.

각 RPT의 방전 Ah를 Q로 고정한다. RPT 저율 전압을 OCV 근사로 사용하고, DOD 20%와
60%에서의 1~30초 전류 완화 응답을 저항·시정수 제약으로 사용한다. 1초 DCIR를
순수 R0로 간주하지 않는다. 공개 impedance 파일은 같은 RPT에서 계산된 자료이므로
독립 검증으로 세지 않는다.

각 단계의 이후 임무를 **적합 4개 → 개발 4개 → 최종 시간순 holdout 4개**로 나눴다.
개발 단계에서 OCV의 양 끝을 이동시키지 않는 내부 보정 `offset*4*SOC*(1-SOC)`을
선택했다. 최종 holdout을 연 뒤 계수·경계·허용치를 변경하지 않았다.
`frozen-coefficients.json`의 계수와 최종 catalog의 계수를 직접 비교한다.

적합 변수는 R0, R1, tau, OCV 내부 보정 4개이며 `C1=tau/R1`이다. Q는 적합 변수가
아니다. 저율 전압에 `0.6A*(R0+R1)` 보정을 더하고 단조 OCV 31개 절점을 사용한다.
임무와 pulse의 잔차 가중치는 표본 수로 정규화한 30mV/5mOhm 공학 척도다.
이는 측정 불확도에서 추정한 통계 가중치가 아니다. 시작 SOC는 각 임무의 첫 휴지
전압을 모델 OCV에 역보간해 정하고, 이후 실측 전류를 적분해 전압을 예측한다.

68개 최종 임무의 셀 전압 RMSE는 **22.72~61.29mV**, 절대 bias 최대 **7.88mV**,
95백분위 절대 오차 최대 **82.10mV**다. 사전 기준 RMSE 100mV, 절대 bias 50mV,
95백분위 200mV를 모두 통과했다. 다만 **최대 순간 절대 오차는 249.69mV**다.
이 기준은 소프트웨어 개발용 응답 기준이며 실기체 정확도·고장 탐지 성능 인증 기준이 아니다.
모든 적합에서 optimizer 경계에 걸린 계수는 없었지만, 원래 단위의 Jacobian condition
number는 약 2,934~4,410이다. 이를 고유한 물리 파라미터가 식별되었다는 근거로 쓰지 않는다.
RPT pulse 적합 RMSE는 약 4.10~4.85mOhm으로, 하나의 RC가 SOC별 응답을 완벽히 설명하지 못한다.

`fit-audit.json`에서 같은 holdout에 정상 단계 모델과 해당 열화 모델을 적용해 비교했다.
초기 약한 열화 단계는 두 모델의 차이가 잔차보다 작을 수 있다. 예를 들어 rpt_01의
모델 간 RMS 차이는 평균 약 12.6mV이고 해당 모델 RMSE는 약 24.5mV다.
따라서 약한 열화를 확실히 탐지할 수 있다고 주장하지 않는다. 마지막 단계의 같은
비교에서는 초기 모델 RMSE 약 124mV가 해당 열화 모델 약 58mV로 감소했다(첫 holdout).
이 비교도 같은 한 셀에서의 검증이다.

## 팩과 모터의 계산

동일하고 균형 잡힌 셀 가정으로 `Q=Np*Qcell`, `R=Ns/Np*Rcell`,
`C=Np/Ns*Ccell`, `U=Ns*Ucell`을 사용한다. R0에 고정 배선저항 5mOhm을 더한다.
X500은 4S4P이며 정상 **실제 Q=12.35735Ah**, 명판 용량은 **12Ah**다.
정상 셀은 Q=3.089338Ah, R0=15.514mOhm, R1=18.713mOhm, tau=5.5233s이다.
팩의 Q·OCV·저항·열화 단계는 evaluation에 보관하고 neural 입력에 추가하지 않는다.
기존 X500 ID와 관측/raw/C9 schema, 명판 4S4P ID를 유지한다.

배터리는 `V=Ns*OCV(SOC)-I*R0-Vp`, `dSOC/dt=-I/(3600Q)`,
`dVp/dt=(I-Vp/R1)/C1`을 계산한다. 전압이 모터 전류를 결정하고 그 전류가 다시 전압을
떨어뜨리므로 같은 시각의 버스 방정식을 이분법으로 푼다(잔차 1e-8V).
RK2 적분의 내부 최대 간격은 0.25ms다. 배터리 열은 `I²R0+Vp²/R1`과 냉각으로 계산한다.

모터 정상 정적 기준은 기존 [Holybro X500 모터 표](https://cdn.shopify.com/s/files/1/0604/5905/7341/files/X500MotorSpec.png?v=1678791632)와
`simulation/artifacts/propulsion-comparison.json`이다. `e=ke*omega`,
`Im=max(0,(duty*V-e)/Rm(T))`, `J*domega/dt=ke*Im-kq*omega²-b*omega-c*tanh(omega)`,
`thrust=kF*omega²`를 사용한다. 전류 제한과 ESC 효율을 적용한 DC 입력 전류의 합에
보조전류를 더해 배터리 부하를 만든다. 모터 권선의 구리 온도 계수는 0.00393/K다.
권선 저항·마찰·프로펠러 계수를 바꾸는 모든 고장도 이 경로로 배터리에 반영된다.
모터 열화 정도는 실측 수명 곡선으로 보정한 값이 아니라 기존 공학적 고장 주입값이다.
모터 관성·냉각·배터리 냉각 역시 별도의 공학 가정이다.

## 고정 열화와 비행 중 이동

| 시나리오 | 배터리 |
|---|---|
| 정상 9종 | rpt_00 고정 |
| battery_aging_existing | rpt_16 고정, 초기 대비 용량 82.45% |
| propulsion_loss_existing/step/ramp | 기본 rpt_00, `--battery-stage rpt_16`으로 열화 배터리 동시 적용 |
| battery_aging_step | rpt_00에서 rpt_16으로 선형 이동, 기본 1초 |
| battery_aging_ramp | rpt_00에서 rpt_16으로 선형 이동, 기본 20초 |

`p(t)=p0+clamp((t-onset)/duration,0,1)*(p1-p0)`를 R0/R1/C1/Q와 OCV 31개 절점에
각각 적용한다. 전환 후 목표 단계로 고정한다. SOC와 Vp는 연속으로 유지하며 사용 가능
전하·에너지의 변화를 `battery_parameter_charge_c`, `battery_parameter_energy_j`,
`battery_rc_parameter_work_j`로 명시적으로 기록한다. 전환으로 접근 불가능해진 에너지를
임의로 발열로 변환하지 않는다. 전환 시간은 가속 고장 시험 조건이며 실제 노화 속도가 아니다.

기본 reference X500과 collection 모두 새 VAH01 모델을 사용한다. 기본 reference도
예전 6Ah/4S2P 구성에서 collection의 명시적인 12Ah/4S4P 질량·관성 구성으로 통일했다.
따라서 과거 reference 결과를 새 모델과 같은 정상 분포로 취급하지 않는다. 기존 VAH07 프로파일,
분리된 6절점 해석식 fallback, R0/RC/Q 배율 기반 배터리 시나리오와 피팅 진입점은 활성
경로에서 제거했다. 실행 중 개별 R0/R1 수동 변경도 Python과 native RPC에서 거절한다.
이전 소스·설정·잠금·바이너리는 `docs/archive/battery-baseline-20261107-01/`에 보존했다.
과거 run·fixture·동결 패키지·기존 모델/scaler/threshold는 덮어쓰지 않았다.

## 현재 추진계의 정상·열화 두 프로파일

`simulation/profiles/propulsion-health.json`의 정상·열화 두 상태를 사용한다. 권선 저항 증가,
점성마찰 증가, 프로펠러 추력계수 저하와 부하 토크계수 증가를 한 프로파일에 함께 넣었다.
정상값은 위 Holybro 정적 성능표 보정을 그대로 유지한다. 열화값은 정상 대비 권선 저항
1.04배, 점성마찰계수 +2e-6 N·m·s/rad, 추력계수 0.99배, 부하 토크계수 1.01배다.
쿨롱 마찰·ESC 효율·회전자 관성·냉각 계수는 그대로다. 모든 로터가 같은 건강 파라미터를
사용하되, 로터별 RPM·전류·온도와 PX4의 개별 명령은 독립적으로 계산한다.

| 실행 시나리오 | 추진계 프로파일 |
|---|---|
| 정상 비행 / 배터리 단독 열화 | 정상 고정 |
| propulsion_loss_existing | 열화 고정 |
| propulsion_loss_step | 정상→열화, 기본 1초 선형 전환 |
| propulsion_loss_ramp | 정상→열화, 기본 20초 선형 전환 |

동적 실행은 `--propulsion-transition-s`로 전환 시간을 지정할 수 있다. Rm·b·kF·kQ의
이동 비율은 동일한 `clamp((t-onset)/duration,0,1)`이며 전환 후 열화값으로 고정한다.
RPM·SOC·분극 전압·온도를 전환 시 재설정하지 않는다. 기존의 모터 전기/회전/열 식과
배터리 버스 연립 계산을 그대로 사용하므로 소비전류·전압 강하·추력 보상도 계속 계산된다.
운전 중 온도로 변하는 실제 저항과 지정한 건강 프로파일의 기준 저항은 구분한다.

약·중·강, 권선/마찰/프로펠러 단독 열화, `combined_existing`은 현재 실행 목록에서
제거했다. 추진계 고장과 열화 배터리의 조합은 추진계 시나리오에 `--battery-stage rpt_16`을
지정한다. 수동 마찰계수 변경과 개별 배율로 프로파일을 덮어쓰는 것도 거절한다.
옛 자료는 당시 생산자 hash로 해석하고, 옛 마찰 단독 run을 같은 이름의 새 통합 프로파일로
조용히 reset하지 않는다. 기존 원시 자료·동결 수집 계획·AI 모델의 내용은 유지한다.

첫 후보(+8% 저항, +8e-6 마찰, -3% 추력계수, +3% 토크계수)는 열화 배터리·SOC0.4·
탑재0.2kg·1.5배 추력 요구에서 duty가 1을 넘었다. 최종 프로파일은 이 모서리 조건의
운용 여유를 유지하도록 낮췄다. 실패 후보와 검사 기준 변경은 개발 자료로 보존했다.
최종 정상/열화 추진계 × rpt_00/rpt_16 배터리 × 두 온도/탑재/추력 조건 × 네 SOC의
정상상태 계산 128건을 수행했다. 그중 SOC0.4/0.9의 64건은 전류·duty 제한을 만족했다.
검사한 호버 조건의 최대 duty는 0.6701, 1.5배 추력 요구의 최대 duty는 0.9958이다.
후자는 낮은 잔량 모서리에서 한계에 가까우며 실제 운용 여유나 실기체 인증으로 해석하지 않는다.
SOC0.2/0.3 조건과 전압 하한 위반은 별도 보고하고 정상 운용 통과로 확대하지 않는다.

25°C·무탑재·SOC0.9·정상 배터리의 정적 호버 계산에서 전류는 정상 22.820A에서
열화 24.014A, duty는 0.5425에서 0.5527로 증가했고 팩 소비전력 증가는 약 4.91%다.
이는 같은 요구 추력에서의 공학 모델 예측이며, 실측 열화율·고장 검출 성능이 아니다.
운전 시간/사이클에 따른 자동 마모 누적이나 SOH/RUL 대응도 주장하지 않는다.
상세 프로파일·폐기 후보·수치 결과·변경 전 파일은
`simulation/artifacts/propulsion-profile-20261107-01/`에 보존한다.

동일 seed·초기 운용 조건의 실제 X500 SITL 호버 임무 30~120초 평균에서는 정상/열화의
팩 전류가 23.319/24.538A, 소비전력이 358.975/376.493W, 평균 duty가 0.5477/0.5580이었다.
소비전력 증가는 약 4.88%다. 원시 자료와 시간 가중 평균 계산은 같은 검증 디렉터리의
`paired-hover-comparison.json`과 `paired_hover.py`에 있다.

```bash
# 세 효과를 함께 가진 고정 열화 추진계 + 정상 배터리
./simulation/capstone-sim run --configuration collection \
  --scenario propulsion_loss_existing --duration 480

# 세 효과를 함께 1초 동안 전환 + 열화 배터리 고정
./simulation/capstone-sim run --configuration collection \
  --scenario propulsion_loss_step --battery-stage rpt_16 \
  --fault-at-s 180 --duration 480

# 세 효과를 함께 30초 동안 점진 전환
./simulation/capstone-sim run --configuration collection \
  --scenario propulsion_loss_ramp --propulsion-transition-s 30 \
  --fault-at-s 180 --duration 480
```

최종 프로파일에서 일반 Python 회귀 196개, 참조/수집 네이티브 CTest 각 6개,
Ruff와 mypy를 통과했다. 실제 PX4/Gazebo에서는 정상 900초 4조건, 고정/급격/점진 추진계
열화와 rpt_16 배터리 조합, 배터리 대조군 및 저잔량을 포함한 총 16조건을 검증했다.
첫 묶음은 비행만 16병렬, 후처리는 최대 3개였고 목표 2배속으로 실행했다.

최초 묶음은 14/16건 비행 완료, 13/16건 데이터 검증 통과였다. 이륙 실패 2건은 native
모터 명령 sequence=0, sample_us=-1, rejected=0 상태였고, 정상 대조군도 포함됐다.
또 다른 완료 비행 1건은 타임스탬프 인과 순서 검사에서 탈락했다. 동일한 코드·파라미터·
seed·비행 길이·검사 기준으로 실패한 3조건만 재시험해 모두 통과했다. 따라서 결과는
**19회 시도 중 검증 통과 16회, 조건별 16/16 통과(`passed_with_retries`)**로 기록한다.
최초부터 모든 시도가 통과했거나 시동 시 명령 전달의 신뢰성을 해결한 것으로 주장하지 않는다.

첫 16건 묶음은 516.21초, 재시험 3건 묶음은 473.28초 걸렸고 실측 임무 배속 범위는
1.863~1.999배였다. 생성한 PX4/Gazebo 종료와 setup/collection/32개 인스턴스 잠금 반납을
확인했다. 옛 개별 마찰·권선 저항·복합 고장의 봉인된 run 3건도 새 코드로 읽기 전용
재검증해 통과했다. 원시 실패와 모든 재시험 결과, 조건별 통과 여부 및 파일 hash는
`simulation/artifacts/propulsion-profile-20261107-01/verification.json`에 연결되어 있다.
기존 AI 모델·학습 데이터·동결 수집 계획은 변경하지 않았고 새 수집 캠페인·AI 학습·
C9 재실행·검출 성능 평가·GPU 작업도 시작하지 않았다.

## 실행 검증 결과

Python 회귀는 개발·메모리 진단 모드에서 27개 모듈 185개 시험을 통과했다. 일반 모드에서는
이번에 변경하지 않은 `test_random_flight`의 간헐 프로세스 충돌(exit -11)이 한 번 재현됐다.
그 원인이 해결되었다고 주장하지 않으며 실패 로그와 별도 진단 모드 결과를 함께 보존했다.
Ruff와 mypy는 통과했다.

참조·수집 두 네이티브 빌드의 CTest(각 5개), 17단계의 실제 파라미터를 넣은 전환/장부/모터 연결
수치 시험, 독립 SciPy DOP853 비교를 실행했다. 독립 해와의 최대 단자전압 차이는
약 1.00e-8V, 전하/SOC/온도/에너지 잔차도 사전 수치 허용치 이내다.
이 작은 값은 수치 구현의 일치도이며 실측 정확도 수치가 아니다.

아래 표는 rpt_16 전용 정책 적용 **이전**에 수행한 기록이다. 같은 초기 SOC0.9, 298.15K, 무풍·무탑재, seed711로 순차 실행한 60초 실제
PX4/Gazebo X500 SITL이다. 열화 목표는 rpt_12, 전환 시작은 임무 20초 후,
급격/점진 전환은 각각 1초/15초다. 표는 두 전환이 모두 끝난 임무 40~55초 평균이다.

| 비행 | 전압 V | 전류 A | duty | 결과 |
|---|---:|---:|---:|---|
| 정상 | 15.395 | 22.778 | 0.5418 | 통과 |
| 열화 고정 | 15.171 | 23.105 | 0.5497 | 통과 |
| 급격 전환 | 15.175 | 23.099 | 0.5496 | 통과 |
| 점진 전환 | 15.178 | 23.096 | 0.5495 | 통과 |
| 모터 마찰 | 15.021 | 31.943 | 0.5854 | 통과 |
| 열화 + 모터 마찰 | 14.705 | 32.617 | 0.5979 | 통과 |
| 모터 권선 저항 | 15.369 | 23.423 | 0.5573 | 통과 |

7건 모두 관측 계약·필수 topic 누락/지연·인스턴스 격리·물리 상태·배터리 고정/선형 경로
검사를 통과했다. 자체 생성 PX4/Gazebo가 종료되었고 자체 소유 소켓이 정리되었다.
원시 run은 결과 JSON에 연결되어 있다. 상세 실행 명령·exit code·원시/집계 결과와
실패 후 수정한 기록은 `simulation/artifacts/baseline-20261107/`에 보존한다.
이 시험은 짧은 호버 기능 검증으로, 15분 전 비행 영역·모든 고장 강도 검증이 아니다.

## rpt_16 전용 정책의 추가 검증

정상은 rpt_00, 열화는 rpt_16으로 실행 대상을 제한했다. 배터리 열화·복합 시나리오의
단계 생략은 rpt_16으로 해석하며 `--battery-stage rpt_12` 등 다른 단계는 실행 전에 거절한다.
이 정책은 catalog hash에 함께 묶었고, 과거 catalog에 묶인 rpt_12 run의 해석은 보존한다.

보정 계수는 하나도 변경하지 않았다. 정책 적용 후 고정 열화·1초 전환·15초 전환·모터
복합 고장을 각각 60초 X500 SITL로 새로 실행해 **4건 모두 통과**했다. 전체 Python 회귀는
개발·메모리 진단 모드에서 **27개 모듈 187개 통과**했고, 두 네이티브 빌드 CTest와
Ruff/mypy도 통과했다. 이전 일반 모드의 간헐 충돌이 해결되었다는 주장은 하지 않는다.
이전 정상·rpt_12 고정·rpt_12 전환 run 3건도 새 코드로 읽어 계약 검사를 다시 통과했다.

실행 명령·관측 계약·선택 단계·모터/배터리 truth 경로·프로세스 종료·소유 잠금 해제의
증거는 `simulation/artifacts/rpt16-policy-20261107/verification.json`에 있다.

## 재현 명령과 다음 데이터 회차

```bash
# 오프라인 보정, 이어서 계수 변경 없는 시간순 최종 검증
./simulation/capstone-sim fit-battery-aging
./simulation/capstone-sim validate-battery

# 아래 명령들은 실제 비행을 시작한다.
./simulation/capstone-sim run --configuration collection --scenario hover_normal --duration 90
./simulation/capstone-sim run --configuration collection --scenario battery_aging_existing \
  --battery-stage rpt_16 --duration 90
./simulation/capstone-sim run --configuration collection --scenario propulsion_loss_existing \
  --battery-stage rpt_16 --duration 90
./simulation/capstone-sim run --configuration collection --scenario battery_aging_ramp \
  --battery-stage rpt_16 --battery-transition-s 30 --fault-at-s 180 --duration 240
```

보정은 원자료와 catalog를 변경하는 개발 작업이다. 새 계수를 채택할 때 소스 잠금·네이티브
준비를 다시 검토해야 한다. 기존 봉인 파일을 임의로 갱신해 실행 검사를 우회하지 않는다.
새 회차의 정상 train/validation/calibration 및 정상/이상 test 데이터는 새 생산자 revision으로
다시 생성하고, 정상만으로 학습·보정한 뒤 독립 test에서 검출을 재평가해야 한다.
기존 동결 수집 계획과 기존 학습 결과는 이전 revision의 증거로 보존한다.
이번에는 대량 수집·AI 학습·경보 보정·GPU 작업을 실행하지 않았다.

rpt_16 전용 정책의 실행 증거는 `simulation/artifacts/rpt16-policy-20261107/`에 별도 보존한다.
이전 catalog·정책·시험은 각 run의 봉인된 원본으로 해석하며 새 정책으로 다시 표시하지 않는다.

최종 검증 상태와 파일 hash는 `simulation/artifacts/battery-baseline-validation.json`과
`simulation/artifacts/baseline-20261107/verification.json`을 따른다. 한 셀의 시간순
검증을 독립 셀 검증으로 부르지 않는다. 셀 편차·불균형 팩·SOC/온도 의존 저항·히스테리시스·
저온 고부하·실제 조립 팩·실기체는 검증되지 않았다. 현재 근거는 측정된 단계에 맞춘
공학 시뮬레이터의 보정 및 소프트웨어/SITL 검증이며 실배터리 완전 재현은 아니다.

## 개발 지침 적용 기록

2026-11-07 KST에 프로젝트가 지정한 attention-span, andrej-karpathy-skills의 CLAUDE.md,
anti-slop, verification-before-completion 원문을 조회했다. 적합/개발/최종 검증을 구분하고,
수정 범위를 배터리와 그 연결·시험으로 제한하고, 원시 실패 증거를 보존하며 실제 명령의
exit code로 완료를 판단했다. anti-slop의 JS/TS 전용 lint는 이 Python/C++ 작업에 적용하지
않고 기존 Ruff/mypy/C++ 컴파일·수치/SITL 시험을 사용했다. 원문 링크는
`docs/specifications/CODEX_REBUILD_PACKAGE_V4_2/AGENTS.md`에 있다.

## 2026-11-07 충돌 경로 수정과 수집 길이 검증

기준 일반 실행에서 `kinematics()` 내부의 중첩 zip/생성자 평가가
`TypeError: cannot unpack non-iterable float object`로 실패했다. 보존된 native core도
같은 생성자 프레임을 가리켰다. 고정된 north/east/down 3축을 명시적으로 계산하도록
수정했다. 기존과 새 구현의 1,283개 시각, 15,396개 스칼라 출력이 정확히 일치했다.
물리 식·배터리 계수·운용/추종 허용치·동결 경로 membership·RNG·데이터 계약은 바꾸지 않았다.

수정 후 `PYTHONMALLOC=debug`나 `-X dev` 없이 전체 189개 시험을 통과했다. 랜덤 비행
시험 12회도 일반 실행에서 모두 통과했고, 1,200개의 동일 동결 계획을 20회 반복해
총 24,000건을 검산했다. 이는 서로 다른 경로 24,000개를 만들었다는 뜻이 아니다.
시스템 Python 내부의 최초 메모리 손상 원인까지 규명하거나 하드웨어를 수리한 것으로
주장하지 않는다. 이 증거는 수정한 애플리케이션 경로의 유한 반복·장시간 검증이다.

장시간 고장 비행 종료 시 truth 전체를 리스트로 읽던 부분은 사건 시각만 순차 추출하도록,
큰 파일 전체를 읽던 SHA256은 1MiB 단위 처리로 바꿨다. 약 99MB의 실제 truth 파일에서
두 작업의 Python 추적 메모리 peak는 2,105,422 bytes였고 기존 integrity의 해시와 일치했다.
약 38MB의 합성 로그에서도 메모리 4MiB 미만을 확인하는 회귀 시험을 추가했다.

비행만 16개 병렬, 비행 외 작업과 후처리는 최대 4개 병렬로 실행했다. 모든 비행은
`real_time_factor=2`를 요청했다. 실측 임무 배속은 약 1.936~1.973배였으며,
16건의 비행 묶음은 시작·이륙·착륙·종료를 포함해 529.58초(약 8분 50초) 걸렸다.
후처리 시간은 이 비행 묶음 시간과 별도로 기록했다.

| 검증 구분 | 건수 | 각 요청 비행 시간 | 기동·운용 조건 |
|---|---:|---:|---|
| 정상 | 9 | 900초 | 호버·고도·직선·yaw·혼합·바람·저온·탑재·연속 랜덤 |
| 저잔량 정상 | 1 | 180초 | 저잔량 + 직선 |
| 열화/모터 고장 | 6 | 480초 | rpt_16 고정·1초 전환·30초 전환·마찰·권선 저항·복합 고장, 혼합/타원/랜덤 |

**16건 모두** 착륙·disarm, 실제 FC 설정, 인스턴스 격리, 관측 계약과 integrity,
필수 topic의 최대 간격 0.5초, 고정/선형 배터리 경로, 물리 범위 검사를 통과했다.
모든 비행에서 위치 오차 0.75m 이내 비율은 100%였고, 최대 위치 오차는 약 0.455m였다.
방향 추종도 기존 0.25rad/95% 기준을 통과했다. 정상 비행 최저 전압은 약 13.268V였다.
자체 생성 PX4/Gazebo 프로세스 종료와 수집·16개 인스턴스 잠금 반납을 확인했다.

원시 run ID, 요청 조건, 모든 통과·실패 로그, 수정 전 archive 및 상세 수치는
`simulation/artifacts/endurance-20261107-01/`에 보존했다. 데이터 생산 캠페인이나
AI 학습·GPU 실행·검출 성능 평가를 시작하지 않았다. 이 결과는 16개 유한 사례의
시뮬레이션 검증이며 모든 seed·환경·실기체의 보편적 안전성 증거가 아니다.

## 2026-11-07 32병렬·목표 2배속 단일 부하 시험

사용자의 추가 지시에 따라 위 16개 조건을 두 벌로 복제해 32개 비행 작업을 한 번만
실행했다. 정상 900초 18건, 저잔량 180초 2건, 배터리/모터 고장 480초 12건이며,
모든 작업에 2배속을 요청했다. 비행 외 작업은 최대 4개, 후처리는 3개 병렬로 제한했다.
자동 재시도는 하지 않았다. 동시 실행 중인 run은 최대 32개였으며, 임무 airborne 상태가
겹친 비행은 최대 27개였다.

32개 인스턴스를 격리하기 위해 collection의 허용 인스턴스를 0~31로 확장하고,
GCS 수신 포트를 `14700+instance`로 배치했다. 기존 `14600+instance`를 그대로 확장하면
offboard 로컬 포트 `14580+instance`와 겹치므로 포트 구간을 분리했다. 설치/보정 잠금도
32개 인스턴스를 검사한다. 수집의 기본 병렬도는 16으로 유지했다. 이 시험을 위해
배터리 계수·모터 물리식·데이터 스키마·품질 기준을 변경하지 않았다. 일반 Python 시험
191개, 참조/수집 네이티브 CTest 각 5개, Ruff와 mypy를 통과한 빌드를 사용했다.

| 결과 | 건수 |
|---|---:|
| 비행 완료 | 27/32 |
| 비행 실패 | 5/32 |
| 비행·관측 데이터 전체 검증 통과 | 22/32 |
| 비행 완료 후 데이터 검증 탈락 | 5/27 |

비행 실패는 `takeoff target not reached` 4건과
`recording_stalled: PX4 clock advanced but raw capture did not` 1건이었다.
기록 정체 시 PX4 시각과 기록 시각의 차이는 5.012초로 기존 5초 한도를 초과했다.
완료된 비행 중 4건은 임무 도중 `vehicle_local_position_setpoint`의 비정상 품질 필드가
관측돼 탈락했고, 1건은 `causal order`를 위반했다. 후자의 첫 battery_status 사건은
sample_time_us=5,201,000인데 available_time_us=0이었다. 실패한 원시 자료와 검사 결과는
보존했으며, 품질 기준을 완화하거나 통과 데이터로 편입하지 않았다.

비행 묶음의 실제 소요 시간은 657.03초(약 10분 57초)였다. 완료된 임무의 실측 배속은
1.210~1.534배로 목표 2배속을 유지하지 못했다. 호스트 전체의 사용 중 swap은 최대
3.799GiB, CPU 사용률은 보조 샘플 구간 평균 88.2%/최대 98.1%, load average 1분 값은
최대 165.75였다. 가용 메모리는 최소 6.856GiB였고, 호스트의 oom_kill 누적 카운터는
시작과 종료 모두 0이었다. 이 자원 수치는 다른 애플리케이션도 포함한다. 자원 압박은
관측됐지만 각각의 실패 원인이 CPU나 메모리라고 단정하지 않는다.

자체 생성한 64개 PX4/Gazebo 프로세스의 정상 종료와 setup/collection/32개 인스턴스
잠금 반납을 확인했다. 단일 32병렬 시험은 실패로 기록했다. 이번 측정에서 검증을 통과한
기준은 앞선 16병렬 설정이며, 32병렬 결과를 배터리 보정의 실패나 하드웨어 최대 병렬도
확정으로 해석하지 않는다. 상세 수치·개별 실패·원시 run 연결·변경 전 보존 자료는
`simulation/artifacts/parallel32-20261107-01/verification.json`과 같은 디렉터리에 있다.
