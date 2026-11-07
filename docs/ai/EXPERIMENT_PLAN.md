# X500 현재 실험 계획

## 2026-11-07 현재: 승인된 LR108·전체1674 사용자 실행 연결

사용자는 **전체1,674런 구성을 승인**했고, 총괄의 [연결 구현 승인](/home/sa/CapstoneV3/ai/reports/main-collection-connection-20261107/attempt-01/authorization.json)에 따라 기존 수집기의 **1 LR108 / 2 전체1674 / 0 종료**를 실제 유한 실행 경로에 연결한다. 이 구현 작업에서 실제 비행·자료 통계 fit·학습·모델 forward·GPU를 시작하지 않는다. 사용자가 명령을 실행해야 수집이 시작된다. 아래 C04/FINAL16·이전 메뉴 기록은 당시 이력이며 현재 본수집의 READY나 재사용 근거가 아니다.

전체1674는 정상 train540 / validation81 / calibration81 / 개발243(별도정상·paired정상·이상 각81) / 최종729(각243)다. LR108은 같은 case/seed/group/역할을 가진 train81·validation27 부분집합이다. **2번만 실행해도 LR108이 포함되며**, 1번을 먼저 완료했으면 검증된108개 원시/두19 산출물과 같은 계보를 재사용하고 추가1566개만 비행한다. validation을 train으로 옮기거나 예비45/FINAL16·옛300/324 자료를 입력·학습·READY에 연결하지 않는다. [main 계획](/home/sa/CapstoneV3/Data/operations/training19/plans/main1674-20261107-01/manifest.json)과 [LR 계획](/home/sa/CapstoneV3/Data/operations/training19/plans/lr108-20261107-01/manifest.json)의 원문/hash는 바꾸지 않는다.

공통 조건은 X500 한 구성, 9기동×3바람(0 / 0초과2.5이하 / 2.5초과5m/s이하), 초기SOC90%·25°C·추가탑재0·각480초·19입력이다. 저SOC·중간SOC·저온·추가탑재를 넣지 않는다. 16비행·요청RTF2, 비비행 전체최대4개이며 비행이 모두 끝난 뒤 원시평가·압축검증·추출·두19·독립검산·무손실 보관을 한다. 계획의 옛 native v1/execution_ready=false 표시는 역사적 계획 정보로 보존하고, 별도 본수집 실행 계약으로 현재 `joint-native-consumption-c2-v2`와 source `f2c38847…`/runtime `b9af577d…`를 검증한다. 물리·40ms 소비간격·기록·문맥/target 제외·전처리 수치 기준을 완화하지 않는다.

각 case는 최초1+재시도2, **총최대3물리시도**이며 같은 seed·case·pair·group·부모계보를 보존한다. 알려진 독립 비행실패3회 후에는 실패 분모를 남기고 다음 case를 진행한다. 성공 raw의 전처리 실패는 같은 raw에서 총최대3처리시도만 한다. 물리 COMPLETED의 공식 독립품질거절은 원래 COMPLETED와 원본·거절 근거를 보존한 경우에만 남은 물리시도를 사용한다. source/무결성/보호목록/공유자원·미확인 결과·사용자 중지는 안전 정지하며 전체 자동 재시작이나 시도 횟수 초기화가 없다. 중지 뒤에는 사용자의 명시적 resume만 허용한다.

모든1674 case의 성공·실패·불가용 및 모든 시도를 JSONL index·작은 case별 원장에 남긴다. 완결된 경우에만 bounded 합산과 train 전용 고유 유효 mission 프레임 통계를 만든다. main540의 공통 scaler를 두 표현에 같이 쓰고, LR81의 scaler는 같은 raw에서 별도 파생한다. `aggregate` 내부의 독립 검산은 한 번만 호출하며 성공 scope는 hash-bound 체크포인트로 재개 시 재사용한다. 최종729의 입력 적격성 검사는 모델 성능평가·튜닝 개봉이 아니며, calibration81도 이번에 임계값 보정에 사용하지 않는다. 실패/누락이 있으면 전체 PASS/학습 READY를 표시하지 않는다.

저장은 기본 `Data/` 또는 사용자가 선택한 기존 루트다. 원시1674개를 모두 펼쳐 쌓지 않고, 이번 작업의 성공 raw·기록·두19·감사·pair·계보를 byte 검증한 무손실 xz capsule과 영구 export로 먼저 동결한 뒤 **그 작업 소유의 중간 raw/gzip/consumer만** 제거한다. 부분압축·공식 품질거절·실패·과거 자료는 보존한다. 실제180초4조건 압축 측정을480초로 외삽한 영구보관 점추정 약321GB/계획상한 약394GB, 배치 scratch 약24GB·합산 allowance 약10GB·여유50GB를 적용한다. 이 수치는 검증 전 외삽이며480초 전체량을 실측했다고 하지 않는다. 실제 사용량·잔여 예측·디스크/RAM을 실행 중 확인해 부족하면 현재 성공 결과를 남기고 정지한다.

진행 조회는 기존 CLI의8+8 화면과 실제 비행 elapsed/target/RTF, 합산 단계·건수·train540/LR81을 사용한다. 입력검산100% 뒤에도 worker 실제 종료·Invocation/PID·종료코드0·소유 정리·잠금/포트 반환을 별도로 확인한 뒤 `MAIN19_INPUTS_PASSED_STOPPED`로 표시한다. systemd unit이 정상 GC된 경우에도 bound 종료 증거와 현재 소유 프로세스/cgroup 부재가 필요하며, not-found의 기본 success 값으로 성공을 추정하지 않는다. 프로그램은 유한7일 상한/전체자동재시작 없음/LLM·예약·콜백 없음으로 종료한다.

구현·정적검사와 실제 수집 완료는 구분한다. 최종 AI 인수·코드 봉인·시험·정적 factory/preflight의 상태와 사용자 명령은 [이번 인계](/home/sa/CapstoneV3/docs/data/reports/main-collection-connection-20261107/attempt-01/INTERFACE.md)를 따른다. 이 문서는 별도 학습/LR선정/보정/성능평가 승인을 만들지 않는다. CURRENT_STATE는 총괄만 갱신한다.

## 2026-11-07 현재: C04 비비행 겹침 배제 대조실험 준비·실행 대기

총괄의 명시 지시에 따라 같은 Sim source `f2c38847…`/runtime `b9af577d…`를 유지하는 C04를 준비했다. CPU 경합은 미확정 가설이다. `nonflight_when_flights_idle=true`로 기존 유한 runner의 비행·무거운 비비행 동시 입장을 막되 비행16/요청RTF2/비비행합계4/180초/40ms 소비간격/준비59초·물리·19입력·누적max3를 유지한다. 기본480 정책과 LR/본학습 메뉴는 변경하지 않는다.

원45case/seed/group/부모·과거48시도를 보존한다. 원본·view가 적격한28개는 현재 reader/audit 검산28/28과 기존파일해시를 확인해 재비행·재전처리 없이 재사용한다. 추가 물리 시도17개는 공식 fullgate 거절3개+native 실패1개의2차 시도와 미시도13개의1차 시도다. 거절3개 원본의physical COMPLETED는 유지하고 AI snapshot만 hash-bound `quality_rejected`로 투영한다. 모든 원래 실패/거절과 조건을 유지하며 데이터 보간·품질 기준 완화·성공 원본 재생성은 하지 않는다. 성공 가정의 누적65/물리실패17/품질거절3/유효45는 계획값이며 실제C04 성과가 아니다.

실제 factory·AI 정적 전제검사·첫 admission 직전 경계·호스트 자원/잠금 검산까지 통과했으며 서비스/비행 제출0이다. `stop_on_first_physical_failure=true`로 새 실패의 자동3차 소모를 막는다. [준비 기록](/home/sa/CapstoneV3/docs/data/reports/short-cycle-20261107/attempt-01/resume-c04/prepared.json), [인수 근거](/home/sa/CapstoneV3/docs/data/reports/short-cycle-20261107/attempt-01/resume-c04/acceptance.json), [실행 인계](/home/sa/CapstoneV3/docs/data/reports/short-cycle-20261107/attempt-01/execution-handoff.json)를 총괄이 검토한 뒤 최종 단독 제출 지시를 별도로 한다. 학습·LR108/본학습1674 수집·모델/fit/GPU는 이번 범위 밖이다.

## 2026-11-07 최신: LR108·본학습1674 최종 승인과 번호 메뉴

사용자가 본학습 생성 전 최종 확인을 요구한 뒤 **“전체1,674런 구성 승인”**으로 답했다. [승인기록](/home/sa/CapstoneV3/Data/operations/training19/authorizations/main1674-20261107-01.json)을 남긴 후 [LR108 계획](/home/sa/CapstoneV3/Data/operations/training19/plans/lr108-20261107-01/manifest.json)과 [본학습1674 계획](/home/sa/CapstoneV3/Data/operations/training19/plans/main1674-20261107-01/manifest.json)을 준비했다. LR108=정상train81/validation27이며 본학습의 같은 역할 부분집합이다. 본학습은 train540/validation81/calibration81/개발243/최종729, 추가1566개다. LR108을 중복 가산하거나 validation을 train으로 옮기지 않는다. 실제 수집 task·수집·학습은0이다.

기존 수집기 실제 메뉴는 **1 LR108 /2 본학습1674 /0 종료**로 교체했다. 옛300/104/324·예비/통합은 새 메뉴·입력·재사용·READY·자동후속으로 연결하지 않는다. 과거 자료와 실제 진행중/복구 작업은 삭제·중단하지 않았다. 실제 active service/잠금 부재를 확인한 뒤 메뉴/순수계획 소스3개와 관련시험2개만 적용했고, live 전처리/AI/native는 바꾸지 않았다. 관련146+메뉴/서비스7=153시험과 새소스 Ruff/format·실제0/1/2 메뉴를 검증했다.

9기동×3바람·초기SOC90%/25°C/추가탑재0/각480초·공통19 및 같은물리 endpoint를 유지한다. 모든 원시/재시도/paired/group 분할과 최종729 격리를 유지한다. AI [읽기전용 LR계약](/home/sa/CapstoneV3/ai/reports/lr108-input-contract-20261107/readonly-01/DATA_INTERFACE.md)에서 LR 독립 native 경계·bounded 합산/고유train81통계·LR-only 실행 연결이 미완료임을 확인했다. 따라서 **계획/메뉴 준비완료와 실제 수집·학습 READY는 구분**하며 execution_ready=false다. 예비의 approved_for_pilot/실제결과를 LR 인수승인으로 확대하지 않는다. [준비/배포/검증 보고서](/home/sa/CapstoneV3/docs/data/reports/training19-preparation-20261107/attempt-01/REPORT.md), [completion.json](/home/sa/CapstoneV3/docs/data/reports/training19-preparation-20261107/attempt-01/completion.json)을 따른다. CURRENT_STATE는 총괄 소유로 남긴다.

## 2026-11-07 최신: 같은 예비45를 실제16병렬·요청RTF2로 전환

사용자가 총괄에 지정한16비행/요청2배속·비비행합계4슬롯 [승인](/home/sa/CapstoneV3/ai/reports/p27-pilot45-20261107/parallel16-01/authorization.json)을 적용했다. 기존flow03은 이미보낸drain으로3개case의두19/독립audit까지마친뒤정지했고, 실제PID0·소켓/16인스턴스잠금반환을확인한뒤Data소스4개만변경했다. 기존성공3 raw/NPZ와통합1 raw는재비행하지않았으며동일45 case/seed/group/pair/retry namespace와시도수를새동결작업에상속했다.

새flow `/home/sa/CapstoneV3/Data/operations/pretrain/tasks/p27-parallel16-20261107-01`/PID594707, 하위 `/home/sa/CapstoneV3/Data/operations/pretrain/tasks/p27-parallel16-20261107-01-45`/PID594726를제출했다. 2026-11-07T06:26:01.536949+00:00 단발host조회에서실제native16개·instances0~15·소유UID1000/cgroup·각argv요청RTF2를확인했다. 움직이는10슬롯의실측RTF는1.82~1.98였고나머지초기0/미확인도보존했다. CPU전단계는한4슬롯pool/장부는한coordinator이며회귀129+실제임시소켓/서비스15=144개와Ruff/format을통과했다. 실제관측CPU작업은0이므로전처리전체완료를주장하지않는다.

instance13의첫recording_stalled1건은실패raw/parent/로그를보존하고기존최대3시도정책으로동일case2번째시도중이었다. 원인해결이나재시도성공을주장하지않으며현재16개작업/성공3개를유지한다. 새AI사전검사는actualnative6/새capsule/전체45분모·pending42로통과했으며실제미래45입력완료가아니다.

실행·보존·시험·오류·단발관측/중지는 [parallel16 보고서](/home/sa/CapstoneV3/docs/data/reports/p27-pilot45-20261107/parallel16-01/REPORT.md), 실제최종은flow completion.json/FINAL_REPORT.md 또는failure.json·하위quality/ledger를따른다. 독립유한프로그램이수집/두19/독립품질/9pair/AI45/실제종료까지만진행한다. LLM감시·예약·콜백/학습·fit·forward·성능평가·GPU/본수집·다음회차는없다. CURRENT_STATE는총괄소유다. Learning rate용정상108(train81/validation27)은별도미승인제안이며예비45를학습에전용하지않는다.

## 2026-11-07 최신: 종료조회 수정·성공원본 재사용·예비45 실제 시작

flow02의종료판정오류는Result를검사하면서실제systemctl조회에그property를넣지않은Data버그였다. 요청property만돌려주는fixture로red7→green7을확인하고실제Result/ExecMainCode/ExecMainStatus/SubState/InvocationID조회·loaded/PID0/실제exit0검증과stop전후증거보존을수정했다. 누락값/실패/미종료/unloaded기본값은완료로받지않는다. 전체관련105+서비스5=110개·Ruff/format통과다.

통과한정상raw b46a56213ab94358bf1e7c30a0198ade와기존두19(N470창/target470초/run초기100이후470.1초)는재비행없이읽기전용으로재검산했다. 실제SERVICE_RESULTsuccess/phase별exit0·owner·cgroup/원시integrity·NPZ/수치/동일producer를확인했고unload된unit의기본Result는사용하지않았다. 기존flow02/one/config/ledger/log/raw/NPZ를보존했으며새flow는통합1편을참조한다. 최초import실패1회계보와논리시도2도그대로다.

실제Sim6편/3pair와AI READY후Data의새45AI prerequisite를실행해45pending/actual_data_verifiedtrue/ai_prerequisites_passedtrue를확인했다. 아직미래45입력이나성능완료가아니다. `/home/sa/CapstoneV3/Data/operations/pretrain/tasks/p27-recovery-20261107-flow03`/PID574522와새sealed45 `/home/sa/CapstoneV3/Data/operations/pretrain/tasks/p27-recovery-20261107-flow03-45`/PID574614를제출해실제ownerUID1000·active/running·첫case비행중을한번확인했다. 정상1편재생성은0이며예비45전체완료는아니다.

승인된정상27+대조9+이상9·최대3시도/같은seed·성공raw재사용·두19/독립품질/실제9pair/AI45/실제종료후정지를기존유한프로그램이이어간다. CPU/RAM/disk/swap/개발격리/no-fit/no-GPU/본수집금지와LLM감시·예약·콜백금지를유지한다. CURRENT_STATE는총괄소유다. 오류/회귀/인수/배포·조회/중지는 [종료수정 보고서](/home/sa/CapstoneV3/docs/data/reports/p27-pilot45-20261107/termination-03/REPORT.md), 최종결과는flow completion.json/FINAL_REPORT.md 또는failure.json을따른다. 아래95개시험·대기flow기록은이전이력이다.

## 2026-11-07 최신 연결: AI 확정pair/ID API 반영, flow02 재개

Data는AI의 [publisher 인터페이스](/home/sa/CapstoneV3/ai/reports/p27-pilot45-admission-20261107/recovery-01/publisher-interface.json)를확인해native pair 보고서의boolean/field명추정을제거하고AI `project_pair_receipt`를호출하도록수정했다. CLI의 `{paired_case_id}`는현재각caseID로치환하며같은pair만ID를공유한다. `native_sources_sha256`는두19view/receipt에그대로보존하고실제closing검산의AI joint_native source도동결목록에묶는다. 관련90+서비스5=95시험·Ruff/format이통과했다. 앞서빌드중차단된producer봉인연동5개도이번에포함해통과했다. 실제개발6편/예비45인수완료는아니다.

기존flow01은하위작업없는대기상태에서자체unit만중지했다. PID0/inactive/dead/not-found/성공과소켓정리를확인하고config/ledger/로그를보존했다. 원래Data01 실패21파일도유지했다. 실행중봉인을수정하지않고새 `/home/sa/CapstoneV3/Data/operations/pretrain/tasks/p27-recovery-20261107-flow02`/`drone-pretrain-p27-recovery-20261107-flow02.service`로동결·제출했다.13:59:16KST 실제PID545503/UID1000/active-running·WAITING_DEPENDENCIES·하위phase없음을한번확인했다.

Sim의실제6편/3pair-check후AI one-shot publisher를호출하는정확한계약이확정됐다. Data프로그램만유한6시간최종READY를기다리고검산한뒤기존승인순서(안정봉인회귀→동일case/seed/실패1회계보의정상1편·두19/종료→새sealed45·두19/9pair/AI45품질/실제종료→정지)를진행한다. READY/fixture/실제실행을구분하고false값을수정하지않는다. CPU/RAM/disk/swap·최대3시도·성공raw재사용·개발자료격리와본수집/학습/fit/forward/평가/GPU/LLM감시금지를유지한다.

실행/중지/재개·검증·남은실제인수는 [adapter02 보고서](/home/sa/CapstoneV3/docs/data/reports/p27-pilot45-20261107/adapter-02/REPORT.md), 실제unit/PID는 [deployment.json](/home/sa/CapstoneV3/docs/data/reports/p27-pilot45-20261107/adapter-02/deployment.json)에있다. CURRENT_STATE는총괄소유로남긴다. 아래flow01·publisher미확정기록은당시이력이다.

## 2026-11-07 최신 복구: 오류 수정 후 예비45까지 유한 진행

사용자가 총괄에서 “해결 될때까지 재시도 해서 최종적으로 예비수집까지 완료해”, “그과정에서 생기는 오류 찾은 오류는 수정하고”라고 갱신 승인했다. [recovery 승인](/home/sa/CapstoneV3/ai/reports/p27-pilot45-20261107/recovery-01/authorization.json)이 아래의1run이후일괄중지/기존feasibility중단 순서를 갱신한다. source/환경 오류를 고치고 유한 재시험하며 성공 raw와 실패 계보를 보존한다. 무한 재제출·시도수/seed 초기화·품질완화·본수집/학습/fit/forward/평가/GPU/LLM감시는 승인하지 않았다.

기존Data01은pymavlink import 오류로 실제raw0/실패attempt1이었다. Data의비행을공식Sim launcher/환경으로분리하고 현재export명run_flight도확인했다. 새task는원래case/seed1668195275/group을유지하고실패1회를참조장부에보존한다. 코드/설정/실패로그는oldtask에서변경하지않는다. 실제인터프리터/CLI와종료JSON·stdout/stderr, 유한복구·분모·자체stop hook을검산해 관련82+서비스5개가통과했다. 실제producer봉인연동5개는Sim빌드중source/runtime불일치로차단됐으며최종안정봉인뒤프로그램이한번검증한다.

현재 **FINITE_FLOW_RUNNING_WAITING_DEPENDENCIES**. `/home/sa/CapstoneV3/Data/operations/pretrain/tasks/p27-recovery-20261107-flow01`를 user service `drone-pretrain-p27-recovery-20261107-flow01.service`/PID534327/UID1000으로제출했고13:41:59KST에실제active/running·WAITING_DEPENDENCIES를한번확인했다. Data새비행은0편이며예비45완료가아니다. AI의기계판독data-handoff가WAITING이어서실제Sim6편·최종봉인·native소비경계/위상정책·AI매핑·공식CLI/pair verifier를기다린다. Sim수정/빌드중옛봉인으로단독재시험도시작하지않는다.

독립프로그램만유한6시간대기후 최종검산→같은최종producer정상1편/두19/종료→새sealed45(정상27+paired정상9+이상9)→각원본별두19/독립품질·실제9pair/AI45검산→종료한다. 각case최대3시도, 성공raw재생성금지, CPU처리합계≤4(현재1)/가용RAM1GB/디스크50GB/swap허용을유지한다. shared/native/기록·source gate를우회하지않는다. 같은raw의입력·임무시작포함/landing제외·초기100/tail·유효시간·실측자원과모든실패/불가용은별도기록한다.

AI최종READY를실제native완료후작성할순수publisher/helper연결은담당에게요청했고현재완료확인전이다. 미완료면정확한의존파일/사유와유한만료를남기며45를강제로시작하지않는다. LLM폴링/예약/자동콜백은없다. CURRENT_STATE는총괄소유다. 실행/조회/중지/재개·수정/시험·남은의존성은 [복구보고서](/home/sa/CapstoneV3/docs/data/reports/p27-pilot45-20261107/recovery-01/REPORT.md), 실제PID/owner감사는 [deployment.json](/home/sa/CapstoneV3/docs/data/reports/p27-pilot45-20261107/recovery-01/deployment.json)에있다. 이하 이전상태는당시이력으로보존한다.

## 2026-11-07 최신: Data 두19 연결 수정·총괄1run 먼저, 예비45 차단

상태: **DATA_FIXTURES_PASSED_INTEGRATION1_READY_NOT_STARTED_PILOT45_BLOCKED**. 최신 사용자 전달 지시에 따라 Data는 Sim/AI 채팅과 실제 결과를 확인해 연결부를 수정했고, 총괄이 실행할 정상1run을 별도로 동결했다. Data가 실제 비행·실제raw 전처리·학습·보정·추론·성능평가·GPU를 시작하지 않았다. 관련82개 시험(77+서비스5), Ruff/format·shell·현재producer754항목의binding/import 검사가 통과했다. 전체Data 시험은50초 제한으로 종료돼 전체 통과로 합산하지 않는다.

Sim의 [paired 원점 결과](/home/sa/CapstoneV3/simulation/artifacts/paired-origin-20261107/validation-01/completion.json)는 `passed=false`, native 예약/소비 구현 미완료, feasibility exit4, 개발6편0회다. AI의 [인터페이스](/home/sa/CapstoneV3/ai/reports/p27-pilot45-admission-20261107/attempt-01/interface.json)도 실제 인수 미완료이며 native receipt projection은 미동결이다. **현재45의seal/start는 명시적으로 막았다.** native·실제6편·AI 매핑 후 Data의 최종 연결 검증도 필요하며, 임의 PASSED 요약으로 해제할 수 없다. 아래 바람40조건 통과와 이 신규paired gate는 별개다.

- 총괄 실행용: `Data/operations/pretrain/tasks/integration19-20261107-data01`, 상태READY_NOT_STARTED. 건강한X500/rpt_00/정상추진계·직선왕복/strong bounded·SOC0.90/298.15K/탑재0·요청480초·물리시도1회. 실제path.kind=line이며 scenario=hover_normal 이름으로 호버 분류하지 않는다. 최초실제통합은 총괄이 명시start하고 결과를 남긴 뒤 정지한다.
- 예비45 초안: `Data/operations/pretrain/tasks/pilot45-20261107-draft01`, BLOCKED_DEPENDENCIES. 정상27+paired정상9+이상9, 역할36normal_test/9anomaly_test.9기동·3환경·9이상방식의 제안 의미는 아래 표를 유지한다. 실제45실행·본수집·학습 준비 완료라는 뜻이 아니다.
- 두 입력: 같은 causal10Hz common19에서 온라인X[100,19]/Y[10,4]와 physical fullT×19/int64 time. 온라인만V/4,A/12를한번변환. 실제mission 시작 포함/landing 시작 제외, 초기100·tail·결측/종료 제외 증거를 보존한다.60초ctx6·SOC/Q/R/건강 truth·바람trace는 신경망 입력에 넣지 않는다.
- 단일run 작업도 `pilot_development_reference`/normal_test다. 학습·validation·calibration·최종독립시험에 편입하지 않는다. 기존개발참고77개와실패/계보/해시도 보존한다. 과거대형stream 전부를 새로재해시했다고 주장하지 않는다.
- 실행기는 flight/prepare/audit마다 별도자식프로세스를 종료한다. 실제RSS반환·native/socket/잠금 종료 및유효시간/창수/바이트는 총괄1run에서 측정해야 한다. 가용RAM1GB·디스크50GB 보호, CPU합계1, swap허용, 성공raw보존·수동재개를 유지한다.16비행병렬을 검증했다고 하지 않는다.

명령·출력·중지/재개·합격기준: [총괄1run 인계](/home/sa/CapstoneV3/docs/data/reports/p27-pilot45-20261107/attempt-01/HANDOFF.md). 수정파일·시험·남은gate: [Data보고서](/home/sa/CapstoneV3/docs/data/reports/p27-pilot45-20261107/attempt-01/REPORT.md). 수집량은 아직 확정하지 않는다. 실제유효율·wall/RSS/저장량·실패/불가용분모를 보고 다음수집을 판단하며,1run완료로45·학습·최종평가를 자동시작하지 않는다. LLM감시/폴링/예약/완료자동호출은 만들지 않았다. 공통CURRENT_STATE는 총괄 소유로 유지한다.

## 2026-11-07 완료: 순간풍 최대5m/s의 시간변동 바람 지원

상태: **WIND_SUPPORT_VERIFIED_PAIRED_INTAKE_PENDING_COLLECTION_NOT_STARTED**. 사용자가 확정한 최대5m/s는 실제 순간3차원 바람 벡터의 크기다. 새 `bounded-latent-ou-c2-enu-v1`은 생성 단계에서 이 상한을 보장하며 풍속과 방향을 시간에 따라 바꾼다. 운용 축은9기동×무풍/약풍/강풍의27조합을 유지한다. 기본profile은 무풍0, 약풍0.25~2.25m/s, 강풍2.75~4.75m/s로 사용자 구간 안에 여유를 둔다. 평균풍5에 무한지지Gaussian돌풍을 더하는 중간 해석은 현재 수집에 사용하지 않는다.

최종producer에서 기본정상27＋추가/고정풍회귀8＋강풍열화5의 **40/40조건**이 통과했다. 실제40회 실행, 실패·재시도0회, 임무480.000~480.097초다. 완료 전 Python220개·네이티브14개·Ruff·mypy, 오프라인108경로·장시간바람18/잠재과정2조건·결합정적40조건을 확인했다. 임무 안 모델 범위 이탈은0이고, 임무 전/후를 포함한 전체 이탈55,668표본도 보존했다. Data의13개 원자료/계보/봉인 검산과12개 완료 검산, Sim 최종32개 검사가 통과했다. 공식runtime 검증, 소유 네이티브80개·소켓40건 종료와34개 잠금 반납을 확인했다. 부품endpoint·센서잡음·raw/C9 schema·신경망19입력은 유지됐고 Data 보호8파일·Sim 보호18파일 해시가 같았다. C9 전체 인수시험·실기체 정확도·학습 충분성·AI 성능 통과로 확대하지 않는다.

최종 source SHA256 `cac87a2526e7dc92a46415880f3b93a56a3016b5009db79cb089df507f647b65`, runtime SHA256 `9bdfb2787fa7b5fe888de0d52385098f1b60de44fa6a034478ffa915ca0392fd`다. battery catalog `08ae8e7cabb835a5b1d59e053ad524c14629c047b942649e56583132c984719e`, propulsion catalog `fc974a80265c6d32753d25a4dd63a322bb3b300f5d9d3a2f24181cf135b220c2`는 유지됐다. [Sim 보고서](/home/sa/CapstoneV3/docs/simulation/reports/wind-bounded-support-20261107/REPORT.md), [Data 보고서](/home/sa/CapstoneV3/docs/data/reports/wind5-gust-support-20261107/REPORT.md), [Data 완료 검산](/home/sa/CapstoneV3/docs/data/reports/wind5-gust-support-20261107/completion-audit.json), [사용법](/home/sa/CapstoneV3/docs/simulation/WIND_ENVIRONMENTS.md)을 근거로 한다. bounded 환경은wind_profile.environment와wind_model/trace로 해석한다. 이때 legacy wind_e/wind_n/gust_sigma의0값은 무풍 판정 근거가 아니다. 실제 기동도 공통scenario 이름이 아니라operation_variation의path kind와 실행 임무로 확인한다.

**paired 수집 인수는 대기다.** 열화5편과 공유정상1편은 같은 경로·seed·profile 및 절대시각의 바람 함수를 사용했고 공통시각 바람 차이는0이었다. 그러나 임무 시작은84~180ms 달랐고5비교 모두mission_phase_exact=false였다. 임무 상대 바람 차이는RMS 최대0.548667m/s, 최대3.247137m/s다. 이 차이를 무시하거나 평균전력 차이를 건강 상태만의 효과로 단정하지 않는다. 예비paired 수집 전 Sim/Data/AI가 임무·날씨 원점 정렬 정책과 실제 차이 검사 기준을 확정해야 한다. [동일 forcing 비교](/home/sa/CapstoneV3/simulation/artifacts/wind-bounded-support-20261107/validation-01/paired-forcing.json)를 보존했으며 이번 지원 완료를paired 준비 완료로 표시하지 않는다.

새40편은 [개발참고 목록](/home/sa/CapstoneV3/docs/data/reports/wind5-gust-support-20261107/development-reference-runs.json)에 학습·예비45·최종독립시험 제외로 등록했다. 이전 고정풍 성공32/실패2와 중간 평균풍OU3편도 원래 판정/계보/해시로 보존했다. 예비45·본수집 후보1,674·학습·보정·AI평가·GPU는 시작하지 않았다. 바람 지원과 별도로 새producer/metadata의두19입력 인수, Data45case/중앙계보·누출 검사·유한 실행/재개·per-run 메모리 반환 및paired 시각 정책을 확인한 뒤 사용자가 실제 수집을 실행한다. 아래 진행 중·평균풍 가정·고정풍만의 상태는 당시 이력으로 보존하며 위 완료 범위가 우선한다.

## 2026-11-07 사용자 확정 정정: 순간풍 최대5m/s의 시간변동 환경

사용자가 최대5m/s의 의미를 **"순간 풍속까지 최대5m/s"**로 선택했고, "단순하게 바람이 많이 부는 환경으로 생각했다"고 설명했다. 따라서 아래 평균풍5+무한지지Gaussian돌풍 지원 가정은 폐기한다. 그 구현/단위/수치/행렬 증거는 중간 개발 이력으로 보존하고 현재 수집 조건이나 돌풍 지원 완료 근거로 사용하지 않는다. Data는 Sim에 이 최신 조건으로 구현·유한 실제 검증을 계속하도록 전달했다. 잠시 보류한 새launch는 새 정의의 설계·봉인·검사 준비 후 진행할 수 있다.

현재 운용 축은9기동×3바람환경=27조합을 유지한다. 바람 환경은 **무풍0 / 0<순간풍속<=2.5m/s / 2.5<순간풍속<=5m/s**다. 비무풍 환경은 각 구간 안에서 바람의 세기와 방향이 시간에 따라 변하도록 한다. 상한은 실제 적용3차원 합성 바람 벡터의 크기이며, 기체 지상속도와 합쳐지는 상대 대기속도8m/s 모델한계와 구분한다. 확률적3sigma를 하드상한으로 취급하거나 사후 로그clipping으로 최대5를 주장하지 않는다. seed재현 가능한 연속적 유계 바람 생성과 해당 바람집합을 고려한 계획예산7m/s·유입비0.1·기존 품질gate를 함께 검증한다.

Sim은 기존Gaussian OU 의미와 과거 경로·원자료를 보존하면서, 필요한 최소 유계wind revision/metadata와 시나리오를 준비한다. 변환 전sigma를 실제풍 표준편차로 표시하지 않으며 실제 풍속 최소/최대·풍향/축·변동·연속성·시간상관·재현성과 paired 실현을 검사한다. 새 정상27조건, 강풍random의 소수 추가seed/풍향·변동세기, 고정풍회귀 및 강풍열화5조건을 포함하는 유한40조건 안팎 계획을 새로 동결한다. 이전 평균풍40조건 목록은 이 정의로 재명명하지 않는다. 이미 실행된 중간case가 있다면 별도 개발참고로 보존한다.

Sim은 `bounded-latent-ou-c2-enu-v1`과 별도40조건을 `simulation/artifacts/wind-bounded-support-20261107/validation-01/`에 동결했다. 기본 생성 범위는 무풍0, 약한 바람0.25~2.25m/s, 강한 바람2.75~4.75m/s로 사용자 구간 안에 경계 여유를 둔다. 이는 기본profile 설정이며 시간변동 바람이0~5 전 범위를 균등하게 채운다는 뜻이 아니다. 잠재OU의 매끄러운 변환과1초 매듭의C2 보간으로 실제 풍속/풍향을 만들고, 모든 물리 단계에서 적용한다. 경로계획은 각 환경의 전체3차원 풍속ball 상한0/2.5/5를 고려한다. 잠재과정의tau2초와 무차원변동강도0.5~1.5는 변환된 실풍의 상관시간·표준편차와 구분한다. 실제40조건 검증은 아직 완료 전이다. 중간 평균풍OU3편은 원래 판정으로 모두 종료·검사를 마쳤고 [별도 개발참고 목록](/home/sa/CapstoneV3/docs/data/reports/wind5-gust-support-20261107/superseded-mean-ou-runs.json)에 보존했으며 새40조건에 합산하지 않는다.

상태: **INSTANTANEOUS_WIND5_IMPLEMENTATION_AND_VERIFICATION_REQUESTED**. 신경망19입력·raw/C9 schema·배터리/추진계endpoint·기존 모델/scaler/threshold·성공/실패 증거는 유지한다. 예비45·본수집 후보1,674·학습·보정·AI성능평가·GPU·서비스/예약은 시작하지 않으며, 수량은 지원 변경만으로 늘리지 않는다. 아래 고정풍sigma0만의 배치는 이전 제안으로 보존하고 실제 예비 수집manifest는 최신 시간변동3환경을 반영해 검토한다. 새 최종producer/지원표 확인 전에는 수집 준비 완료로 보고하지 않는다.

진행 확인: 새 모델의 단위220개·네이티브14개와 수치 검사를 통과했고 작은gate3편에 이어 추가 강풍8편까지 **11/40조건**이 통과했다. Data의 [새40조건 입력 검산](/home/sa/CapstoneV3/docs/data/reports/wind5-gust-support-20261107/bounded-matrix-audit.json)과 [작은gate3편 검산](/home/sa/CapstoneV3/docs/data/reports/wind5-gust-support-20261107/bounded-gate-audit.json)도 통과했다. 시험 중 source `cac87a2526e7dc92a46415880f3b93a56a3016b5009db79cb089df507f647b65`, runtime `9bdfb2787fa7b5fe888de0d52385098f1b60de44fa6a034478ffa915ca0392fd`이며 아직 최종40조건 완료/인수를 뜻하지 않는다. bounded 환경의 실제 바람은 `wind_profile.environment`와 평가용wind_model/trace로 해석하며, 이 경우 legacy wind_e/wind_n/gust_sigma의0값을 무풍 판정 근거로 쓰지 않는다.

## 2026-11-07 추가 지시: 시간변동 돌풍 지원

아래는 상한 의미를 확인하기 전의 개발 이력이다. 사용자의 순간풍 최대5m/s 확정 답변으로 위 최신 절이 우선하며 아래 평균풍 가정·40조건 목록·봉인을 현재 수집에 적용하지 않는다.

상태: **GUST_SUPPORT_IMPLEMENTATION_AND_VERIFICATION_REQUESTED**. 고정풍5m/s 개발 검증 완료 후 사용자가 "고정풍 아닌것도 지원 가능하게 해"라고 직접 지시했다. 기존 시뮬레이션 담당에게 시간에 따라 풍속·방향이 바뀌는 OU 돌풍의 지원 구현·빌드·단위/수치/회귀·유한 실제SITL 개발 검증을 요청했다. 아래 고정풍 완료 기록과 봉인은 보존하며 새 돌풍 지원 완료로 확대하지 않는다. 새producer가 확정되기 전 AI/Data의 최종 봉인 인수는 대기한다.

현재 구현의 바람은3축 exact-discrete OU이며 축별sigma와 상관시간tau(기본2초), seed를 갖는다. 작업 가정은 **평균풍0~5m/s, 기존sigma0~0.4m/s** 지원이다. 순간 합성풍은5m/s를 넘을 수 있고 가우시안 모델의3sigma는 하드상한이 아니다. 평균풍/실제 순간풍/상대 대기속도 및 축별 변동·시간상관을 구분해 검사·기록한다. 순간풍을 강제로clipping하거나3sigma를 전구간 보장으로 보고하지 않는다. 기존 최대5m/s는 이 작업에서 평균풍 상한으로 유지한다고 사용자에게 명시했다.

고풍random의 기존 속도 여유식과 최소1m/s 제약은sigma0.4에서 충돌하므로 단순 허용치 삭제로 통과시키지 않는다. Sim이 근거 있는 최소 경로계획 변경과 필요한 새revision을 검토하고,7m/s 계획예산·8m/s 모델한계·유입비0.1 및 기존 추종/전압/SOC/duty/기록gate를 우선 유지한다. 배터리/추진계endpoint·센서잡음·raw/C9·AI19입력은 변경하지 않으며, 고정풍v3/v4·동결계획·성공/실패raw의 의미와 증거를 보존한다.

실제 검증은 작은gate 후9기동×평균풍0/2.5/5m/s의 상한sigma0.4 정상27조건(각480초)을 기본으로 고풍random의 중간sigma·소수 복수seed/풍향·고정풍 회귀 및5m/s 돌풍 열화5조건을 추가하는 유한 목록을 요청했다. 확정 목록·지원구간·실제 바람 실현/추종·물리범위·선택/주입/endpoint·source/runtime/catalog·실패/최대3attempt·자원/소유프로세스/잠금 반납은 실제 결과로 검산한다. seed를 바꿔 범위 이탈을 숨기거나 검사를 완화하지 않는다. 이번raw는 개발 참고로 표시하며 학습·예비45·최종 독립시험에 편입하지 않는다.

Sim의 초기 오프라인 계획gate5개가 통과했고, 진행 방향별 평균풍 상대속도 예산을 적용하는 새 `random-continuous-v5-ou`를 준비했다. 기존 경로revision과 돌풍 생성식은 보존하며 중간 정지 금지·최소 이동·상승/하강·공간 다양성을 검사한다. Sim은 실제 시험 목록을 **40조건: 상한sigma 정상27 + 중간sigma/추가seed·풍향5 + sigma0 회귀3 + 열화5**로 제시했다. 작은gate3조건을 먼저 확인하며, 이 문장은 시험 계획과 오프라인gate 결과를 뜻하고 실제40조건 통과를 뜻하지 않는다. 근거 경로는 `simulation/artifacts/wind-ou-support-20261107/validation-01/`이며 최종 결과는 별도 검산한다.

예비45·본수집 후보1,674·학습·보정·모델평가·GPU·서비스/예약 실행은 시작하지 않는다. 이번 추가 지시는 지원 개발이며 본수집 수량이나27운용조합의 축을 자동 증가시키지 않는다. 실제 수집의sigma/seed 배치는 지원 결과와 실행manifest 검토에서 고정한다. 사전 보호해시는 `docs/data/reports/wind5-gust-support-20261107/guardrails-before.json`에 남겼고, 기존AI·총괄 채팅에도 새producer 변경 및 실제 수집 대기를 전달했다. 공통계획은Data가 취합·갱신한다.

## 2026-11-07 완료: 고정풍 최대5m/s 지원·SITL 개발 검증

상태: **WIND5_VERIFIED_PILOT_NOT_STARTED**. 사용자 추가 지시에 따라 시뮬레이션 담당이 `x500-wind5-engineering-v1`을 구현했고, Data가 최종 증거를 검산했다. 정상9기동×고정풍0/2.5/5m/s의27조건과5m/s 열화5조건이 최종 producer에서 **32/32 통과**했다. 각 실제 임무는480.000~480.036초다. 최초hover2조건의 임무 밖 저회전 로터 적격성 판정 불일치를 수정하고 같은case/seed/group·parent를 유지한 새attempt에서 통과했다. 전체34회 실행이며 최초 실패2회의 raw·라벨·integrity·판정을 보존했다.

지원 범위는 평균풍 최대5m/s이며 **2m/s 초과는 gust sigma0의 고정풍**이다. 계획 상대 대기속도7m/s·모델 상대 대기속도8m/s·로터별 유입비0.1을 함께 검사한다. random v4의 경로속도는 풍속 여유로 제한하고 보존v3의 범위는 유지한다. 이번 유효범위 확장은 공학적 시뮬레이션 지원이며 실기체 정확도 인증이 아니다. 배터리/추진계endpoint·센서 잡음·raw/C9 schema·common19 입력은 유지됐다. 최종 단위시험205개·네이티브12개·Ruff·mypy, 오프라인108경로·정적40조건 검사도 통과했다. C9 전체 인수시험은 재실행하지 않았다.

채택32조건의 실제 임무 내 모델 범위 이탈은0이고, 임무 전/후를 포함한 전체 이탈40,879표본은 보존했다. 물리 정상 적격성은 **실제airborne~landing 명령 양끝 포함** 구간에서 재계산한다. 이는 AI의 **실제mission 시작 포함~landing 시작 제외** 입력 규칙과 구분하며 AI 입력 경계를 바꾸지 않는다. 부트clock0 기록 수집 순서도 수정하되 원래sample/available 인과 검사는 유지한다. Data는32개 고유run·계보·최종hash·보호파일·최대3시도·실패보존·소유프로세스 종료 및 잠금 반납을 검산했다.

최종 source SHA256 `7501105199fa452cade518f3accce08046e0fe263017eff77746a524da265d86`, runtime SHA256 `df31643f1f74ee185d7d0fedd2b6aaab79fa79653ae6e34107860322b7487a26`다. battery catalog `08ae8e7cabb835a5b1d59e053ad524c14629c047b942649e56583132c984719e`와 propulsion catalog `fc974a80265c6d32753d25a4dd63a322bb3b300f5d9d3a2f24181cf135b220c2`는 유지됐다. [Sim 보고서](/home/sa/CapstoneV3/docs/simulation/reports/wind5-support-20261107/REPORT.md), [Data 검산 보고서](/home/sa/CapstoneV3/docs/data/reports/wind5-support-20261107/REPORT.md), [검산JSON](/home/sa/CapstoneV3/docs/data/reports/wind5-support-20261107/completion-audit.json), [개발참고32run 목록](/home/sa/CapstoneV3/docs/data/reports/wind5-support-20261107/development-reference-runs.json)에 근거를 남겼다.

이번32개는 개발 검증 참고 자료로 고정하며 예비45·학습·최종 독립 시험에 편입하지 않는다. 예비45run·본수집 후보1,674run·학습·보정·모델 평가·GPU 작업은 시작하지 않았다. 실제 예비 수집 전에는 새producer에 맞춘 Data45case manifest/중앙계보/유한실행·재개/메모리 반환과 AI 두19입력 인수를 확인한다. 아래2m/s 초과 지원 대기 및 이전source/runtime에 관한 정정 이력은 **이번 변경 전 상태**로 보존하며 현재 지원/봉인으로 사용하지 않는다.

## 2026-11-07 추가 실행 지시: 5m/s 지원 구현·실제 검증

사용자가 Data 채팅에 "5m/s 지원하게 변경하고 테스트해"라고 직접 지시했다. 기존 담당별 사전 준비의 실제SITL 실행 대기 제한은 이번5m/s 지원 개발 검증에 한해 해제한다. 시뮬레이션 소유 담당인 「시뮬레이션 환경 개선」 채팅에 구현·빌드·단위/수치/관련 회귀·실제 PX4/Gazebo SITL 시험을 요청했다. 예비45run·본수집1,674run·학습·보정·모델 성능평가·GPU 작업은 계속 대기한다. 이번 시험 raw는 개발 검증 자료이며 최종 독립 시험이나 학습 자료에 편입하지 않는다.

범위는 풍속0~5m/s 지원,9기동의 경로 생성·지상속도/상대 대기속도·모델 유효범위·기록/평가 연결의 일관된 수정이다. 기존2m/s 입력 허용치·5m/s 상대 대기속도·random 속도 여유식의 충돌을 함께 검토한다. 필요한 유효범위/계획 여유 변경은 근거와 변경 전후를 명시하고 공학적 지원 확장과 실기체 정확도 인증을 구분한다. 시험 통과만을 위한 품질·추종·전압/SOC/duty 기준 완화, 기동 대체/누락, 배터리/추진계 열화 강도·잡음·신경망 특징·raw/C9 임의 변경은 하지 않는다.

유한 검증안은 먼저 수정/단위/수치 gate와 작은 무풍/5m/s hover gate를 확인한 뒤,9기동×풍속0/2.5/5m/s의27개 정상 조건(각480초),5m/s 통합추진계existing/step/ramp3조건·배터리existing1조건·배터리rpt_16+추진계existing1조건의 총32필수 조건을 검증하는 것이다. 작은 gate가 같은 최종revision의 같은 조건이라면 성공 결과를 재사용한다. 독립 인프라 실패는 동일case/seed/group·parent를 유지하며 최대3시도 안에서만 처리하고, 성공run 전체 재생성과 무한 재시도는 하지 않는다. 반복적 공유/물리/의미 오류는 원인과 최소 수정 근거를 남긴다.

최종 조건별 지원·실제길이·풍속/상대대기속도·추종·전압/SOC/duty·기록/인과순서·실제주입·선택/정상/열화 endpoint·source/runtime/catalog hash·실패/재시도·자원/시간·소유 프로세스/소켓/잠금 반납 결과를 `simulation/artifacts/wind5-support-20261107/`에 보존하고 Data가 검산·취합했다. 실행 결과와 범위는 위 완료 절을 따른다.

## 2026-11-07 전체 구성 재제안 — 27운용 조합·예비45·본수집 후보1,674run

상태: **WIND_SUPPORT_VERIFIED_PAIRED_INTAKE_PENDING_COLLECTION_NOT_STARTED**. 사용자는 개발 균형/최종 전수 배치를 이해한 뒤 예비 수집 전에 각 담당에게 사전 준비를 요청하도록 지시했다. 아래 구성을 사전 준비 기준으로 전달하되45run 실제 비행·수집·본수집·학습·GPU 실행은 자동 시작하지 않는다. 사용자가 지정한27운용 조합과 새 통합 추진계 열화를 반영한 계획이며 수량은 균형 배치와 최종 시험의 조합 누락 방지를 위한 예산 후보로서, 학습 충분성·검정력·탐지율 보장의 근거가 아니다. 바람 조건은 상단의 순간풍 상한/시간변동 정의를 따른다. 본수집 수량은 예비 결과 후 확정한다. 아래 이전 수량 제안은 현재 수량으로 사용하지 않는다.

### 예비 수집 전 담당별 요청 — 2026-11-07

이번 요청의 범위는 담당 영역의 준비·최소 연결 수정·기존 단위/수치·합성 자료 검증과 구체적인 실행 절차 마련이다. 실제 새 SITL 비행·예비45run·본수집·서비스/예약 실행·학습·보정·GPU/Slurm 작업은 이 요청으로 시작하지 않는다. 추가 SITL이 필요하면 명령·수량·검사/중지 조건과 필요 근거를 보고하고 실행 대기한다. 기존 실행/설치 잠금과 사용자 실행 선호, 보존·보안·자원 제한을 지킨다.

|담당 채팅|사전 준비 요청|완료 판단에 필요한 결과|
|---|---|---|
|시뮬레이션 환경 개선|27조합별 지원표,2m/s 초과/5m/s의 상대 대기속도·경로 생성·추종·추진 여유 검토, 새 통합 추진계/배터리 선택과 실제 주입/임무 metadata 확인. 필요한 최소 소유 영역 수정·단위/수치 검증. 상한만 올리거나 불가 조합 대체 금지.|지원/미지원/추가 SITL 필요의 구분, 근거·단위/수치 시험 로그·변경 목록, 새 producer/catalog/runtime 확정값 또는 미확정 사유. 추가 실제 시험이 필요하면 실행 계획만 제출.|
|AI|두19신호 입력과 mission-only 정책·단위·history/future·tail·결측/불가용, 새 추진계 metadata와 old producer 차단, train-only scaler 및 개발/최종 누출 방지·평가 규칙 점검. 모델/특징/threshold 설계 변경 없는 최소 reader/검산 수정·합성 시험.|Data에 필요한 manifest/metadata 항목, 두 reader의 사전 인수 조건·합성 시험 증거·남은 제약, per-run/bounded memory 인수 방법과 비용 측정 항목. 새 producer 최종 봉인 확인 전 완료 인수 주장 금지.|
|총괄|Sim/AI/Data 준비 결과를 합칠 실행 게이트, 자원·저장소·기존 소유 작업/잠금·중복 시작 방지·서비스/종료/재개 절차 검토. old19+6 수집기 대신 새45run 경로의 구체적인 실행/확인/중지 명령 준비.|담당별 READY/대기 항목과 근거, 실제 자원 점검·유한 실행 계획·실행/중지/상태 명령. 실제 비행/서비스/학습은 대기. Data 소유 수집 구현을 동시에 독자 수정하지 않음.|
|Data(이 채팅)|조건·이상9방식·45case 논리 목록, 새 seed/paired/retry/group/분할·중복/보호 목록, current19 수집/압축/추출/입력 생성·회차당 처리 제한의 준비. 새 source 확정 뒤 실행 manifest 동결.|Sim의 지원표와 AI reader 요구를 반영한 검토 가능한45case/실행 계획·누출/수량 검산·단일run 단위 메모리 반환 및 최대3시도 증거. 사용자 실행 전 자료 생성 상태로 대기.|

Sim의 producer 변경과 AI/Data의 hash 인수를 순서대로 처리한다. AI는 합성/소유 영역 검토를 먼저 진행할 수 있으나 Sim 최종 봉인과 대조한 뒤 새 producer 인수를 판단한다. 각 담당은 자기 영역에 보고서를 남기고 자기 채팅의 최종 답변으로 결과·경로·남은 항목을 보고한다. 공통 `EXPERIMENT_PLAN.md`는 Data가 결과를 취합해 갱신하며 동시에 각 채팅이 임의 수정하지 않는다. 자동 감시/회신/예약 후속 호출은 이번 요청 범위에 포함하지 않는다.

전달 기록: 2026-11-07 사용자 직접 요청으로 기존 「시뮬레이션 환경 개선」(`01a1b02a-df1e-7ab2-81db-a3f4ca33d6c6`)·「AI」(`01a1363e-e8ee-73d0-832f-ccf3257bf80a`)·「총괄」(`01a13555-5257-74f1-981d-a5a426b8026c`) 채팅에 각각 사전 준비 요청을 실제 전송했고 세 호출 모두 오류 없이 대상thread ID를 반환했다. 요청 전달 완료와 각 담당의 준비 완료는 구분하며, 현재는 결과 확인 전 상태다. 새 채팅·자동화·실제 수집/학습 작업은 이 전달에서 생성하지 않았다.

### 공통 조건

기동은 호버 / 상승·하강 / 직선 왕복·가감속 / 타원 / 8자 / 리사주 / 제자리yaw / 복합 / 연속 무작위의9항목이며 각각 수집 목록과 coverage를 갖는다. 복합에 포함될 가능성으로 기본 경로의 독립 quota를 대체하지 않는다. 바람은0 / 0초과~2.5 / 2.5초과~5m/s의3구간으로27운용 조합이다. 초기SOC0.90·초기 및 주변온도25°C·추가 탑재0kg·요청480초를 제안한다. 저SOC·중간SOC·저온·탑재 조건은 제외하되 비행 중 SOC·온도 변화는 기록한다.

현재 배치의 바람은 무풍0 / 0<순간풍속<=2.5 / 2.5<순간풍속<=5m/s의3환경이다. 비무풍은 세기와 방향이 시간에 따라 변하는bounded profile을 사용한다. 기본profile의 실제 생성 구간은 약풍0.25~2.25, 강풍2.75~4.75m/s이며 전체구간의 균등 표집을 뜻하지 않는다. 기준풍향·경로heading/위상·허용된 경로 진폭/주기와 무차원변동강도는 허용 범위 안에서seed별로 분산하고 정확한 예비45 배치는manifest에서 동결한다. 고풍random은v6의 가능한3차원 바람 전체에 대한 속도 예산을 따른다.9기동×3환경과 고정풍5m/s 회귀를 포함한 지원 검증은 최종producer에서 완료했지만 모든seed·paired 시각 정렬·두AI입력 인수 완료를 뜻하지 않는다. 각 새run의 기존 적격성 검사는 유지하며 수량을 맞추기 위한 대체 기동·검사 완화를 하지 않는다.

### 역할별 수량 후보

|역할|구성|run 수|
|---|---|---:|
|정상 train|27조합×20새seed|540|
|정상 validation|27조합×3새seed|81|
|정상 calibration|27조합×3새seed|81|
|개발 평가|별도 정상81+paired 정상81+이상81|243|
|최종 독립 시험|별도 정상243+paired 정상243+이상243|729|
|합계|두 모델 공용 원시 계보·common19|1,674|

각 run은 별도 reset으로 생성하되 pair 내부 상관을 유지하며 창/반복 관측을 독립 표본으로 세지 않는다. 위 수량은 성공 계획 case 수이고 실제 attempt 수·실패/불가용은 별도 보고한다. 명목 임무시간은223.2시뮬레이션시간이며 실제 wall time·기록/점수 유효시간과 다르다. 개발/최종 각각 별도 정상·paired 정상·이상 수량을 동일하게 두는 사용자 선호를 유지한다.

정상 train만 모델 학습과 입력 scaler 적합에 사용한다. 정상 validation은 모델 선택 및 AI가 명시한 점수 성분 scale에, 정상 calibration은 임계값에 사용한다. 개발 평가는 방법 검토용이고 최종은 모델·scaler·threshold·후처리·평가 규칙 동결 후 개봉한다. 기존19+6나 이전 추진계 producer 자료를 자동 변환·혼합하지 않는다.

### 이상9방식과 개발/최종 배치

|건강 상태 계열|처음부터 열화|비행 중 급격 전환|비행 중 점진 전환|개발 이상run|최종 이상run|
|---|---|---|---|---:|---:|
|배터리 단독|rpt_16 고정, 추진계정상|배터리rpt_00→16, 추진계정상|배터리rpt_00→16, 추진계정상|27|81|
|추진계 단독|통합추진계열화, 배터리rpt_00|통합추진계정상→열화, 배터리rpt_00|통합추진계정상→열화, 배터리rpt_00|27|81|
|복합|배터리rpt_16+통합추진계열화 고정|배터리rpt_16 고정+추진계정상→열화|배터리rpt_16 고정+추진계정상→열화|27|81|
|합계|계열3×발생방식3=9|기본step1초|기본ramp20초|81|243|

추진계의 저항/마찰/프로펠러 계열 효과는 같은 통합 프로파일에서 동시에 적용되며 옛 부품별 단독 고장·약중강을 다시 추가하지 않는다. 배터리도 정상rpt_00·열화rpt_16의 현재 정책을 유지한다. 전환 시간은 실제 노화 속도가 아닌 가속 시험 설정이다. 복합 동적 사례의 배터리는 시작부터 열화 상태이며 배터리·추진계 동시 전환 시험이라고 표시하지 않는다.

개발 평가는 방식당9run으로9기동을 각각 한 번 포함하고 각 바람구간을3회 포함한다. 균형 배치 예시는 기동m=0..8, 계열f=0..2, 방식t=0..2일 때 `wind_bin=(m+t+f)%3`이다. 각 운용 조합에는 이상3run이 배정된다. 별도 정상은27×3, 각 이상의 정상 대조81개는 별도 raw로 만든다. 이 개발 부분배치가 모든 방식×27조합을 검증한 것으로 해석하지 않는다.

최종은 이상9방식 각각27운용 조합을 모두 한 번씩 거치므로9×27=243run이다. 정상 대조243개를 각 이상에1:1로 붙이고, 별도 정상은27×9=243run이다. 따라서 최종은729run이다. 이는 앞선 작은 부분배치보다 최종의 방식별 기동/바람 누락을 줄이기 위한 제안이며 정밀한 성능 보증 표본수는 아니다. 필요한 정밀도에 따른 반복 증량은 test 점수 개봉 전에 결정한다.

급격/점진 주입은 요청 시점180/240/300초를 방식·기동·바람과 분리해 균형 분산하는 제안이다. 최종의 방식별27run은 각 시점9run, 개발의 방식별9run은 각 시점3run으로 배치한다. 요청 offset의 기존 시나리오 원점과 execution의 실제 임무 시작을 구분하며 실제 onset/endpoint truth로 평가 시각을 정한다. 복합 동적 사례는 배터리 이상이 처음부터 있으므로 추진계 주입 이전 경보를 정상 오경보로 세지 않는다. 배터리 기존 이상 인지와 추진계 추가 주입 전후 반응을 구분하고, 이미 경보 중인 사례의 추가 이상 지연을 억지로 정의하지 않는다.

pair는 목표 전체 경로·바람·seed·초기SOC·온도·탑재·요청 길이를 일치시키고 실제 부하/반응은 그대로 둔다. 같은SOC가 열화 배터리의 남은Ah까지 일치함을 뜻하지 않는다. 각 정상 대조는 해당 평가 cohort에만 속하며 학습/보정에 넣지 않는다.

### 먼저 확인할 예비45run

27운용 조합의 정상 각1run과,9이상 방식 각1run+정상 대조9run으로45개를 제안한다. 이상9개도9기동에 각각 하나씩 배치하고 바람3구간을3개씩 포함한다. 별도 강풍 지원·수치/기능 점검은 시뮬레이션 담당의 사전 증거로 남기며, 현재 지원 불가 요청을 제출해서 반복 실패시키지 않는다. 예비45개는 개발 참고 자료이며 본수집1,674개·최종 시험에 편입하지 않는다.

시뮬레이션 담당에게 실제 조건·목표 경로·seed/group/pair·producer/배터리/추진계 catalog·선택/초기/목표 상태·요청/실제 임무 길이·실제 onset/endpoint, 추종/대기속도/ESC/전압/SOC 여유, 생성wall time·peak메모리·raw/archive 바이트·실패/재시도를 요청한다. 최대3시도(최초1+재시도2), 동일case/seed/group, 모든 실패 증거 보존을 유지한다. 공유 자원 장애·사용자 중지를 무한 재시도로 처리하지 않는다.

AI 담당에게 두 입력의 가용/불가용·임무 경계·결측/값유지/공백·history100/future10 도착·종료 제외·초기100시점·tail, 전처리시간/메모리/저장량을 요청한다. 임무 내부 구간을 이어붙이거나 보간/0채움하지 않고 canonical에 이착륙과 품질 증거를 보존한다.19신호 입력에60초 context나 건강 truth를 추가하지 않는다. 온라인 점수 유효시간과 종료후 run모델의 점수 구간은 각각 산정하며 요청480초에서 일정 시간만 빼서 보고하지 않는다.

조건별 적격성/가용성·생성/정제 시간·원시/압축/두 입력 저장량·메모리를 보고 기존 자원 보호선 안에서 본수집 여부를 판단한다. 반복 실패나 입력 불가용이 생긴 조합은 원인과 보존 결과를 확인하며 조건을 숨기거나 정상 범위를 늘리지 않는다. 후속 승인된 학습에서는 정상train135→270→540(조건당5/10/20반복), 고정 정상validation81개로 두 모델의 조건별 loss/잔차와 추가 수집 효과를 비교한다. 추가 수집은 train/개발 증거로 판단하고 독립 최종 점수로 판단하지 않는다.

원비행·raw hash·재추출·재시도·parent/previous attempt·paired/group은 train/validation/calibration/개발/최종을 넘지 않게 중앙 등록 목록으로 검사한다. 정상/이상test 역할 외에도 개발/최종 cohort를 분리·봉인한다. 온라인은 예측 오차·오경보episodes/hour·사건 검출과 실제 onset 기준 지연, 종료후 run모델은 정상run 오경보·방식/조건별 검출·전압 잔차·불가용률을 보고한다. 두 모델이 이상 원인을 완전히 분리한다거나 실제 열화율/SOH를 추정한다고 가정하지 않는다. 최종의 불가용·미관측도 분모/사유에 남기고 불완전 조건 세트를 완전한 최종 시험으로 표시하지 않는다.

## 2026-11-07 최신 정정: 바람3구간·운용27조합·통합 추진계 열화

사용자는 바람을 무풍0 / 0초과~2.5 / 2.5초과~5m/s의3구간으로 지정했다. 경계2.5m/s는 두 번째 구간에만 포함하고0은 무풍에만 포함한다. 기본7경로+복합·무작위2모드의9수집 항목을 각각 유지하므로 **기동9×바람3=27운용 조합**이다. 이는 건강 상태·발생 방식·반복 횟수를 곱하기 전의 운용 조건 수이며 총run 수가 아니다. 이전4구간·36조합과6범주 기준 수량은 현 구성에 사용하지 않는다. 저SOC·중간SOC·저온·추가 탑재 제외는 유지하며, 초기SOC0.90·25°C·추가 탑재0kg과 요청480초는 기존 기준값 제안으로 둔다.

「시뮬레이션 환경 개선」 채팅의 최신 사용자 지시·최종 답변, `docs/simulation/BATTERY_AGING.md`의 현재 추진계 절, 실제 `simulation/profiles/propulsion-health.json` 및 `propulsion_health.py`/`runner.py`/`collection_config.py`를 대조했다. 새 추진계 정책은 `x500-propulsion-binary-v1`이며 정상·열화 두 프로파일만 사용한다. 열화 프로파일은 권선 저항 증가·점성마찰 증가·프로펠러 추력/부하 특성 변화를 동시에 포함한다. 현재 열화 endpoint는 정상 대비 저항1.04배, 점성마찰계수+2e-6 N·m·s/rad, 추력계수0.99배, 부하 토크계수1.01배다. 모든 로터가 같은 건강 파라미터를 사용하되 실제 로터별 명령·RPM·전류·온도는 별도로 계산한다. 이 endpoint는 실측 수명 단계·SOH가 아닌 시뮬레이션 담당이 선택한 공학적 통합 열화 설정이다. Data 담당은 계수·강도를 변경하지 않는다.

|추진계 시험 방식|현재 실행 시나리오|동작|
|---|---|---|
|처음부터 열화|propulsion_loss_existing|통합 열화 프로파일 고정|
|비행 중 급격 전환|propulsion_loss_step|정상→같은 열화 endpoint, 기본1초 선형 전환|
|비행 중 점진 전환|propulsion_loss_ramp|정상→같은 열화 endpoint, 기본20초 선형 전환|

세 계열의 건강 파라미터는 같은 전환 비율로 이동한다. 기존 권선/마찰/프로펠러 단독 고장·약중강·`combined_existing`은 새 실행 목록에서 제거됐다. 같은 이름 `propulsion_loss_*`라도 과거 마찰 단독 의미로 해석하지 않으며, 아래 과거 제안의 추진계5조건과 기존9조건 배치는 폐기된 구성 기록으로 읽는다.

|고정 건강 상태 조합|배터리|추진계|
|---|---|---|
|정상|rpt_00|normal|
|배터리만 열화|rpt_16|normal|
|추진계만 열화|rpt_00|abnormal|
|배터리+추진계 열화|rpt_16|abnormal|

위4개는 고정 상태 조합이며, existing/step/ramp 발생 방식은 별도 축으로 관리한다. 배터리 단독3방식과 추진계 단독3방식은 유지한다. 현재 복합 실행은 추진계 시나리오에 `--battery-stage rpt_16`을 주는 방식이다. 복합 step/ramp는 배터리가 처음부터rpt_16이고 추진계가 비행 중 전환하는 시험이며, 배터리·추진계가 동시에 정상에서 열화로 전환하는 시험으로 표시하지 않는다. 복합의 발생 방식별 수량과 전체 신규run 수는 이 구분으로 다시 제안해야 하며 이번 조건 정정만으로 자동 증액하지 않는다.

현재 추진계 catalog/profile SHA256 `fc974a80265c6d32753d25a4dd63a322bb3b300f5d9d3a2f24181cf135b220c2`, producer source SHA256 `a8c4ff306f66b0a24964057a7facf4da55ad521d32e9da2562093ddac9a7410e`, runtime SHA256 `b4a6625143824c7324801a50e8bccbe13b26d4d533d5c0dd8cb762297544950c`를 실제 파일에서 계산해 `simulation/artifacts/propulsion-profile-20261107-01/verification.json`의 값과 일치함을 확인했다. 해당 기존 보고서는16조건 중 첫 묶음13조건 통과, 실패3조건 재시험 후 조건별16/16통과·총19시도·`passed_with_retries`다. 이번 확인에서 그 비행/196개 시험을 다시 실행하지 않았으며 탐지 성능·5m/s 적격성·신규 Data/AI 전체 인수 통과로 확대하지 않는다.

새 수집의 raw/evaluation에서 추진계 catalog/selection·초기/목표 상태·전환 시간·실제 onset/endpoint·producer hash를 계보와 함께 보존하고 검증해야 한다. 추진계 건강·전환 fraction·실제 파라미터는 평가 metadata이며19신호 neural 입력에 추가하지 않는다. 과거 producer/마찰 단독 자료와 새 통합 열화 자료를 자동 혼합·재라벨하지 않는다. 새 producer에 대한 Data/AI 인수 증거는 별도 필요하다.

바람3구간은 사용자가 정한 목표 배치다. 현재 code/health policy는 평균풍2m/s까지 허용하므로 두 번째 구간의2초과~2.5와 세 번째 구간은 추가 지원·검증 대기다. 특히 무작위 경로의 기존 속도 여유식은5m/s 요청을 지원하지 않는다. 시뮬레이션 담당에게27조합별 지원 가능성과 상대 대기속도·추종·추진 여유·기록 품질의 확인을 요청하며, Data가 검사 상한이나 물리 모델을 독자적으로 바꾸지 않는다. 이번 반영은 계획 문서에 한정하고 수집·학습·시뮬레이션/AI 코드는 실행·수정하지 않았다.

## 2026-11-07 수집 조건 정정: SOC·온도·탑재 조건 축소, 바람 최대5m/s

사용자는 새 공통19 데이터셋에서 저SOC·중간SOC·저온·탑재 조건을 제외하고 바람 최대5m/s를 요청했다. 이는 운용 조건 정정이며 이번 대화에서 물리 모델·코드·품질 기준을 변경하거나 실제 수집을 실행하는 지시로 확대하지 않는다. 기준 설정은 초기SOC0.90·25°C·추가 탑재0kg을 제안하며, 비행 중 SOC·온도의 자연 변화는 그대로 기록한다.

기동은 아래 정정한 기본7경로와 복합·연속 무작위2모드를 각각 수집 항목으로 관리한다. 요청 길이는 정상·paired 정상·이상 모두480초, 두 모델은 같은 원시 비행과19신호·분할을 공유한다. 바람은 최신 사용자 지시의0 / 0초과~2.5 / 2.5초과~5m/s의3구간이다. 평균풍/돌풍의 최대값 정의·방향·난수 실현은 실행 전에 시뮬레이션 담당과 고정해야 한다. 기존 수량 제안은 기동·바람·새 추진계 상태에 맞춘 예산 재배치 전 확정 수량으로 사용하지 않는다.

현재 `normal_variation.validate`와 `health_policy.yaml`의 평균풍 허용 상한은2m/s다. 기체 모델의 상대 대기속도 한계는5m/s이고, 무작위 경로 계획은 `4.5 - 평균풍속 - 3*gust_sigma`의 속도 여유와 최소1m/s 경로속도를 요구한다. 따라서 바람5m/s 요청을 허용 숫자 변경만으로 정상 범위에 편입할 수 없다. 시뮬레이션 담당이 상대 대기속도·방향별 기동·경로 생성·추종·전압/SOC·ESC 여유와 기록 품질을 검증하고, 지원할 기동/바람 조합을 확인한 뒤 수량을 다시 제안한다. 기존 검사를 완화하거나 지원 불가 조합을 다른 기동으로 조용히 대체하지 않는다. 이번 정정에서 시뮬레이션 소스·생산자 봉인·AI 입력 계약은 수정하지 않았다.

## 2026-11-07 기동 목록 정정 — 기본7경로 + 복합·무작위2모드

사용자가 기존의 다른 기동이 사라진 이유를 지적했다. Data 담당이 현재 경로를6개 큰 범주로 요약하면서 타원·8자·리사주를 독립 수집 항목에서 누락한 계획 오류다. 시뮬레이션 코드에서 삭제한 것은 아니다. `flight_program.BASIC`에는 기본7경로가 있고 `normal_variation.KINDS`에는 mixed/random을 더한9개 항목이 있다.

|수집 항목|현재 kind|확인할 정상 응답|
|---|---|---|
|호버|hover|정위치 기준 부하|
|상승·하강|altitude|반복 수직 부하 변화|
|직선 왕복·가감속|line|직선 이동·감속·반전|
|타원 선회|ellipse|연속 곡선·방향 변화|
|8자|figure8|좌우 선회 전환|
|리사주 곡선|lissajous|축별 주기가 다른 복합 수평 곡선|
|제자리 yaw 왕복 회전|yaw|위치 이동과 구분한 yaw 응답|
|복합 기동|mixed|명시적으로 정한 기본 경로들의 구간 전환|
|연속 무작위 비행|random|수평·고도·속도·방향이 변화하는 연속 경로|

위9항목을 각각 수집 목록과 coverage 표에 명시하는 제안으로 정정한다. 기존6범주 안의 수평 경로 선택이나 복합 경로에 포함될 가능성만으로 기본 경로의 독립 수집을 보장했다고 하지 않는다. 현재 mixed는 한run에서 기본7종 중6개를 택할 수 있어 모든 기본 경로를 반드시 포함하는 것도 아니다. 기본 경로별 최소 quota와 복합의 실제 구성·순서·시간을 별도로 기록한다. 코드상 지원과 새 조건의 실제 SITL/품질 통과는 구분한다.

최신 바람3구간과의 목표 조합은27개이며, 기동별 고풍 지원·검증 대기는 그대로다. 아래24조합·1,056run·예비42run 표는6범주 기준의 이전 검토안으로 보존하며 현재 확정 수량으로 사용하지 않는다. 수집 항목 복원과 총예산 증액은 구분하고, 새로운 최소 quota와 총run 수는 예산 재배치 후 다시 제안한다. 이번 정정은 문서에 한정하며 수집/학습·시뮬레이션 코드 변경을 실행하지 않는다.

## 2026-11-07 이전 수집 구성 재제안 — 기동 정정 전 기록, 현재 수량으로 사용하지 않음

사용자의 저SOC·중간SOC·저온·탑재 제외 및 바람 최대5m/s 요청을 반영한 검토안이다. 다음 수량은 조건 균형과 단계적 학습량 비교를 위한 예산 후보이며 충분한 학습량·검정력·탐지 성능을 입증한 표본수가 아니다. 실제 코드·생산자 봉인·수집/학습을 이번 재제안에서 변경·실행하지 않는다.

### 조건·길이 후보

기동6종(호버, 상승/하강, 수평 가감속, yaw회전, 복합, 부드러운 무작위) × 바람4구간(무풍0, 약풍0초과~1, 중풍1초과~2, 강풍2초과~5m/s)의24조합을 목표로 한다. 기준값 제안은 초기SOC0.90·초기/주변온도25°C·추가 탑재0kg이다. 비행 중 잔량·온도의 자연 변화는 기록한다. 모든 역할의 요청 임무 길이는480초다. paired 양쪽은 같은 길이·목표 경로·초기 운용 조건·바람 설정/seed를 사용하되 실제 기체·전기 반응은 강제로 맞추지 않는다.

첫 바람 적격성 시험은 돌풍0의 고정풍으로 최대5m/s까지 단계적으로 확인한다. 돌풍을 본수집에 포함할지,5m/s가 평균풍/합성 순간풍 중 어떤 상한인지, 방향·실현 경로·강풍 구간 내3/4/5m/s의 배치는 시뮬레이션 담당의 확인 후 실행 명세에 고정한다. 풍속 구간 이름은 수집 층이며 기체의 안전 풍급 인증이 아니다.

현재2m/s 초과 구간은 지원·검증 대기다. 특히 무작위 경로의 속도 여유식상 고풍의 경로 생성은 현재 제한되므로24조합 전체 지원을 전제하여 바로 수집하지 않는다. 상대 대기속도·기동 추종·ESC/전압/SOC 여유·기록/물리 gate를 먼저 검증한다. 지원 불가 조합은 사유를 보고하고 계획을 개정하며, 허용 숫자/품질 기준 변경이나 다른 기동으로의 조용한 대체로 수량을 맞추지 않는다.

### 본수집 수량 후보

|역할|배치|목표 run 수|
|---|---|---:|
|정상 train|24조합×20새seed|480|
|정상 validation|24조합×3새seed|72|
|정상 calibration|24조합×3새seed|72|
|개발 평가|별도 정상72+paired 정상72+이상72|216|
|최종 독립 시험|별도 정상72+paired 정상72+이상72|216|
|합계|공통19 자료에서 두 입력 파생|1,056|

무풍~2m/s의18조합분792run과, 강풍6조합분264run을 나눠 관리한다.792run도 새 전체 조합에 대한 실제 적격성·저장량 검증이 끝났다는 뜻은 아니다.1,056은24조합 전체가 지원되는 경우의 목표 예산이며 새 최종 확정 수량이 아니다. 명목 임무시간은140.8시뮬레이션시간이고 실제 생성/후처리 wall time·유효 점수시간과 다르다.

각 평가 세트에서 별도 정상·paired 정상·이상을72개씩 유지한다. 이상은 배터리 단독3조건(existing/1초 step/20초 ramp), 추진계 단독5조건(마찰 existing/step/20초 ramp, 권선 저항 existing, 프로펠러 계수 existing), 복합1조건(rpt_16+마찰 existing)의9조건이다. 정상/추진계 단독 배터리는rpt_00, 배터리 열화 목표는rpt_16으로 현재 봉인 정책을 유지한다. 전환 시간은 가속 시험이며 실제 노화 속도가 아니다. 고장 강도는 현재 producer를 사용하고 임의로 늘리지 않는다.

각 이상 조건은 평가 세트당8run, 배터리24/추진40/복합8run이다. 기동번호m=0..5, 바람구간w=0..3, 이상조건번호f=0..8에 대해 `(m+w)%3 == f%3`인 조합을 한 번씩 선택하면 조건당8run·모든6기동 포함·바람구간별2run이며, 전체24조합마다 이상3run이 된다. 정상 대조는 이72개 각각에 별도 reset 비행으로 붙이고, 별도 정상은24조합×3이다. 개발·최종은 같은 배치 규칙의 별도 seed/group/원비행을 사용한다.8run/이상조건은 정밀한 유형별 성능 인증에는 작으므로 필요한 최종 정밀도·운영 목표에 따른 증량은 test 점수를 열기 전에 결정한다.

### 먼저 실행할 소규모 예비 수집 후보

최대42run:24조합의 정상 각1개와, 이상9조건 각1개+정상 대조9개다. 강풍을 포함한24조합 적격성 확인을 먼저 수행하며 현재 지원 불가 요청을 대량으로 제출하지 않는다. 추가 강풍 기능/수치/SITL 점검은 시뮬레이션 담당의 별도 증거로 기록한다. 예비42개는 개발 참고 자료이며 본수집 수량에 합산하거나 최종 독립 시험에 넣지 않는다. 정상30조합/예비48개/본수집1,320개였던 직전 제안은 이 안으로 대체해 검토한다.

시뮬레이션 담당에게 case/조건/path/바람/seed·pair·producer/catalog hash, 요청/실제 임무 길이, 실제 주입 truth 시각, 추종·대기속도·ESC/전압/SOC 여유, 완료/실패/재시도, wall time·최대 메모리·raw/archive 바이트를 요청한다. 재시도는 최대3시도(최초1+재시도2)이며 같은 case/seed/group을 유지하고 모든 실패 기록을 남긴다. 성공 계획 run 수와 실제 attempt 수를 구분한다.

AI 담당에게 같은 raw 계보에서 나온 온라인 창과 임무 run 입력의 품질·가용성, 실제 임무 경계, history100/future10 도착과 종료 제외, 초기100시점 및 run tail, reset/결측/불가용 사유, 전처리 시간·메모리·저장량을 요청한다. 현재19 경로에60초 문맥 대기를 추가하지 않는다. 임무 내부 결측을 채우거나 경계를 품질에 맞춰 옮기지 않는다. 요청480초를 단순히 일정 초 빼서 유효시간으로 보고하지 않는다.

예비에서 누락 조건·원인 미확인 실패·두 입력 불가용이 생기면 해당 조건의 원인을 확인한 뒤 본수집 진입을 판단한다. 실제 저장 증가량과 메모리 peak로 기존 저장소 보호선·동시 처리 예산을 확인한다. 후속 승인된 학습에서는 동일한 정상validation72개를 유지하고 정상train120→240→480의5/10/20반복 학습량 비교로 추가 수집 효과를 본다. 이상test·최종test 점수로 수량·범위·고장 강도·모델·scaler·threshold를 조정하지 않는다.

### 분할·평가 경계

전체 원비행·raw hash·재추출·재시도·parent/previous attempt·paired case·group이 train/validation/calibration/개발평가/최종시험 경계를 넘지 않게 중앙 등록 목록에서 검사한다. 현재 reader의 normal_test/anomaly_test 역할만으로 개발/최종을 구분했다고 하지 않으며 별도 cohort/봉인 목록을 유지해야 한다. 같은 pair는 같은 평가 cohort에 둔다. 창 수를 독립 비행 수로 세지 않는다.

두 모델은 같은19신호와 역할·계보를 공유하고 모델별 가용/불가용을 함께 보고한다. 정상 train만 학습·입력 scaler에, 정상 validation은 모델 선택/명시된 점수 성분 scale에, 정상 calibration은 임계값에 사용한다. 개발 평가는 방법 검토용이고 최종은 모델·scaler·threshold·평가 규칙 고정 후 개봉한다. 온라인 모델은 예측 오차·오경보 episodes/hour·실제 onset 기준 검출/지연을, 비행 종료 run 모델은 run 오경보·조건별 검출·전압 잔차·불가용률을 보고한다. 종료 후 모델에 실시간 탐지 지연을 부여하지 않는다. 두 모델이 배터리/추진 원인을 완전히 분리한다고 가정하지 않고9조건에 대한 반응을 모두 기록한다.

## 2026-11-07 최신 변경: 두 모델 공통19 데이터 준비

사용자는 배터리 열화용 run 모델과 비행 중 미래 예측 모델을 둘 다 새19신호 입력으로 학습할 계획이며, 기존19+6 신규 구성을 사용하지 않도록 정정했다. 새 준비의 기준은 **현재 CMU 배터리 생산자의 공통 물리19신호 데이터셋 하나**다. 같은 raw/run 계보와 분할에서 온라인 창과 전체 run 입력을 파생한다. 아래10월31일·11월3일의19+6 회차·완료 증거·최종 시험 자료는 역사 자료로 보존하며 이번 새 학습 자료로 자동 편입하지 않는다.

Data 연결 수정 뒤 총괄이 정상1run의 실제 비행·공통 전처리·두 입력 인수를 완료했다.
비행은 총1개이며 경로 오류와 임무 구간 정책 수정 후 같은 원자료를 재사용했다.
대규모 수집·학습·GPU·보정·탐지 평가는 시작하지 않았다. 새 데이터 최종 수량과 조건별 충분성은 아직 확정하지 않는다.

|공통 근거|모델별 파생 입력|관리 규칙|
|---|---|---|
|현재 X5004S4P12Ah, `cmu-vah01-rpt16-only/1`, 정상rpt_00|history19:100시점19신호,1초 cadence,10시점4target|AI 소유 causal 전처리·도착 target 결합, context6 없음|
|동일 canonical19 frame·품질·시각|run_health19:실제 임무 구간의 연속10Hz 전체[T,19],마지막 tail 보존|pack V/A 그대로 인계; AI 안에서 변환·초기100 이후 전압 masking|
|한 공통 역할·계보 목록|각 view의 유효/불가용 사유 별도|결측 구간 이어붙이기·보간·실패run 은폐 금지|

분할은 정상 train/validation/calibration/test 및 anomaly_test의 역할을 유지한다. 동일 raw/hash·재시도·parent/previous attempt·pair·group은 분할하지 않는다. paired normal_test/anomaly_test는 같은 test 분할에 둔다. 선택 목록 밖 기존 등록·보호 자료와의 중복 대조는 실제 수집 계획의 인수 조건이며 이번 합성시험이 전체 외부 데이터의 누출 부재를 인증하지 않는다.

정상 train만 학습·scaler 통계에 사용하고 validation은 모델 선택, calibration은 정상 임계값, test는 고정 후 평가에 사용한다. 이상 test를 학습·정규화·threshold 결정에 넣지 않는다. 두 모델에 공통 목록을 쓰더라도 한 view가 unavailable이면 사유·coverage·전체 분모에 남긴다. AI의 현재 run 로더에는 역할별 manifest를 주고 pair 관계는 공통 목록에 보존한다.

정상1run 제안은 hover_normal/90초/seed2026110701/instance0/요청2배속이다. 충분한 학습량이나 다양한 조건의 검증을 위한 길이가 아니다.10초 history와 future 도착·종료·결측을 실제 프레임으로 검사한다. 새19 경로에 구60초 context 대기를 강제로 적용하지 않는다. 독립validation이 없으므로 `NOT_FIT_READY`이고, 검토에 사용한 예비run을 최종 독립시험으로 재명명하지 않는다.

총괄의 실제1run에서 PX4의 지면 접촉 시 상승률 명령이 미정의가 되는 정상 동작을 확인했다.
사용자는 이착륙을 별도 구간으로 두고 임무만 run 모델 입력으로 쓰도록 승인했다.
현재 run 입력 정책은 `mission_only_actual_execution/1`이며 생산자 execution의 실제 `mission`
시각 이상, `landing` 시각 미만의10Hz 격자 전체다. 품질을 보고 경계를 옮기지 않는다.
canonical에는 이착륙과 결측도 모두 보존한다. 임무 내부 결측·공백·양끝 누락·segment 변경은
계속 unavailable이며 보간·결측 채움은 없다. 온라인 history19의 인과 창 생성 범위는 유지한다.
명시적 임무 정책이 없는 과거 전체 airborne run manifest는 새 run 소비자가 거절한다.

연결 코드는 `Data/generation/pretrain/ai19.py`, 진입점은 `./Data/generation/drone-data ai19`다.
초기 Data 합성 통합·회귀시험61개 이후, 총괄 수정의 Data34개·AI reader/기능12개가 통과했다.
실제 run `ba4b64e199d3426798321cdbe674934b`에서 온라인98창과 임무880frame을 인수했다.
마지막80frame 부분 block을 보존했고 단위/배열 검산과 기록 파일의 읽기 분할 비교는 오차0이다.
실제 생산자-live 연결 비교와 새 학습 bundle의 forward는 미실행이다. 한run으로 scaler나
train/validation을 만들지 않았으며 `NOT_FIT_READY`다.
[총괄 결과](../../ai/reports/ai19-battery-integration-20261107/attempt-03/REPORT.md)에 실행·종료 근거를 남겼다.
현재 CMU 생산자에서 기존19+6 신규 `create`는 차단하며 기존 자료·동결package·실패 증거를 보존한다.
대량 회차의 메모리·저장량과 기존 보호목록 연계는 별도 검증 조건이다.

[구현·시험·보존·총괄1run 요청](../data/reports/ai19-battery-integration-20261107/REPORT.md)에 exact 명령과 증거를 기록했다. 최신32병렬 시험은 비행27/32·자료22/32 통과로 실패했으며 이번 Data 합성시험으로 해결됐다고 하지 않는다. 원인 미확인 오류·조건별 가용성·실제 처리율을 먼저 보고 후속 수집량·병렬도·길이를 판단한다. 물리 계수·입력 의미·품질 gate를 파형 일치나 통과율을 위해 변경하지 않는다.

다음의 기존 회차 기록은 당시 상태와 승인 범위로 읽는다.


기준일: 2026-10-31 KST. 사용자는 남은 작업의 **1번 서버 인수 → 2번 자원·메모리·복원 검증 → 3번 본학습**까지 순서대로 진행하도록 승인했다. 상태는 `APPROVED_PREPARATION_AND_TRAINING_PENDING_GATES`다. 비교 구성은 `ROUND_001/R6_1200_8MODEL_S0`를 유지한다. 최신 자료의 인수·게이트·유한한 예산 확정이 통과해야 본학습하며, **4번 정상 보정·개발 추론·독립 평가와 추가 수집·다음 회차는 이번 승인에 포함되지 않는다.**

## 현재 데이터

로컬 기본 저장소의 `processed/2026-10-31/prepared-portable/`를 인수한다. 저장소 루트는 `/home/sa/CapstoneV3/Data/`다. 압축 데이터는 5,693,868,960바이트(약5.69GB)이며, 복원 크기는 79,996,912,784바이트(약80.00GB)다. 압축본 크기를 원시 비행로그 크기와 혼동하지 않는다.

분할은 정상 train720 / validation104 / calibration104 / 정상 개발 평가128 / 이상 개발 평가144run이다. 원래 run·그룹·수집 seed·분할을 보존한다. 정상 train만으로 만든 기존 공통 scaler를 사용하며, train 창 수는 최신 품질 기록상569,743개다. 재수집이나 다른 분할로의 이동은 승인 범위가 아니다.

[현재 데이터·모델 계획](../../ai/configs/comparison-plan.json)과 압축본 receipt가 정확한 데이터 해시의 기준이다. 실제 인수 시 archive/member 해시, 품질 증거, train-only scaler, 역할·lineage·NPZ를 검증한다. 잠근 최종 시험 자료를 인수하거나 성능 확인에 사용하지 않는다.

## 완료와 남은 검증

- **완료 — 로컬 데이터:** 최신 1,200run 수집·통합 전처리·train 전용 공통 scaler·독립 품질 검산·무손실 압축 검증·날짜 폴더 정리.
- **구현 완료 — AI 인수:** 1,200run 역할·계보·봉인·배열 형상 검사. 이전 104run 제한을 확장하는 코드는 이미 구현되어 있다.
- **구현 완료 — 비교 실행:** seed0 신경망 8종과 baseline 2종의 유한한 순차 실행, 구성별 메모리·저장·복원·export 게이트.
- **구현 완료 — 메모리 처리:** train/validation의 chunk 정규화 mmap과 작은 primal 행렬을 쓰는 ridge. CPU4·RAM64GiB 요청 계약도 구현되어 있다.
- **구현 완료 — 독립 실행·화면:** 서버 tmux/Slurm 실행기와 학습·GPU 진행 CLI. LLM 호출이나 자동 후속 회차는 없다.
- **아직 미확인 — 이번 서버 실행:** 최신 자료 전송·약80GB 복원·AI 인수, 현재 서버 자원·큐, 정확한 8개 구성의 batch256 메모리·복원 검증, 실제 처리율·예산 확정, 본학습. 구현 완료를 이 항목의 실행 통과로 표시하지 않는다.

과거 서버 기능시험은 2026-10-18 DIS01 job19348에서 저장·복원·할당 반납까지 통과했다. 과거104run의 작은 표본으로 총16 optimizer step을 수행한 시험이므로 최신1,200run이나 본학습8구성의 인수 증거로 대체하지 않는다.

## 비교 구성

LSTM과 Transformer 각각 hidden64·2층 /128·2층 /128·4층 /256·4층, 총8종이다. 층 수는 encoder와 decoder 각각의 수다. Transformer heads는4/4/4/8이다. persistence와 ridge_forecast를 더해 총10종의 결과를 비교한다. 전체 명세의 모델 비교 목표는 유지한다.

학습 seed0, batch256, FP32, AMP 비활성, 최대60epoch, patience10, dropout0.1, AdamW lr0.001·weight_decay0.00001로 고정한다. 각 신경망은 새 seed0 가중치에서 시작한다. 과거 회차·게이트·다른 구성의 가중치를 이어받지 않는다. 수집 seed는 바꾸지 않는다. 추가 seed·다른 구성은 후속 승인 대상이다.

각 정확한 구성의 실제 GPU 메모리와 별도 프로세스 복원·export 검사를 통과한 후 본학습한다. 실패하면 가중치·로그를 보존하고 전체 비교를 멈춘다. batch를 줄이거나 재시도하지 않는다. 짧은 메모리 게이트 속도를 전체 학습 처리율로 제시하지 않는다.

## 자원과 실행 경계

기존 승인된 `ssh knu` 접속과 mamba 환경을 사용한다. Slurm p02/DIS01/단일 노드/GPU1개, 프로젝트 동시 GPU 작업1개를 지킨다. 일반 RAM 요청은64GiB이며 실제 가용량·추정 피크와 GPU 조건을 할당 안에서 다시 검사한다. CUDA_VISIBLE_DEVICES를 바꾸거나 로그인 노드에서 GPU 연산하지 않는다.

실제 데이터 로더를 포함한 학습 step 처리율과 validation 시간을 구분해 측정하고 fit/queue 시간을 실제 제출 전 유한한 예산으로 동결한다. 안내문의48시간/12시간은 예시 상한이며 예상 소요시간이 아니다. 이번에는 보정·평가 예산을 배정하거나 해당 작업을 제출하지 않는다. 현재 서버 자원이 확보됐다고 주장하지 않는다. 추가 환경 설치·업그레이드는 하지 않는다.

신규 준비 작업은 선택한 저장소의 별도 폴더를 사용하며 기본은 프로젝트 내부 `Data/`다. 외장 저장소를 명시한 경우에는 마운트 유실·경로 이상에서 다른 디스크로 대체하지 않고 실패한다. 원래 압축 데이터와 기존 산출물은 변경하지 않는다.

이번 승인 절차는 서버 자료 인수 → 자원·처리율·구성별 게이트 →10종 순차 학습/fit → 모델·체크포인트·설정·학습 기록 저장 → 학습 할당 반납 → 정지다. validation으로 checkpoint를 선택한다. 실제 성공 종료·반납 확인 후 `TRAINING_READY_STOPPED_PENDING_EVALUATION_APPROVAL`로 표시하며 정상 보정·개발 추론·평가를 자동 시작하지 않는다.

후속 승인 대상인 전체 비교 목표는 정상 보정 → 개발 추론 봉인 → 독립 개발 평가 → 보고다. calibration은 정상 자료만 사용하고 개발 이상 정답은 evaluator만 결합한다. 전체 명세의 평가 목표는 유지한다.

후속 평가에서는 예측 오차, 정상 오경보 episodes/hour, 사건 검출·미관측/불가용 사례, onset/recognition delay, 공통 유효 origin의 예측 성능을 함께 보고한다. 낮은 성능 때문에 재학습·재보정·추가 데이터/seed를 자동 실행하지 않는다. 전체 평가와 실제 자원 반납·보고 검증을 완료한 뒤에만 `REPORT_READY_STOPPED`를 사용할 수 있으며 이번 학습만의 종료에는 사용하지 않는다.

## 현재 명령·상태·증거

- 수집·전처리: `./Data/generation/drone-pretrain`. 현재 기본은300run, 비행16·그 밖의 처리4병렬,2배속, 게이트 선택이다. 이번 점검은 수집 승인이 아니다.
- 새 비교: `./ai/drone-ai train --help`. [준비·전송·서버 CLI 순서](PORTABLE_SETUP.md)를 따른다.
- 수집 상태는 `Data/operations/pretrain/latest.json`과 해당 작업의 `operator/state.json`을 사용한다. 비교 실행 후에는 해당 작업의 `prepared.json`, `controller.json`, release/evaluation/completion 파일을 사용한다. 중복된 전역 상태 관리 체계를 만들지 않는다.
- [정리 기록](../archive/maintenance/workspace-cleanup-20261017-01/report.json), [추가 점검 기록](../archive/maintenance/workspace-cleanup-20261017-01/review-02/)를 보존한다. 코드·입출력 계약의 C9 연결은 이번 점검에서 변경하지 않았다.

프로그램의 유한한 반복과 CLI 화면 갱신은 LLM 없이 수행한다. 화면 닫기와 OS/서버 작업 종료는 다르다. 사용자의 결과 확인 요청 전까지 LLM 감시·예약 호출·완료 이벤트 자동 재개를 만들지 않는다. 다음 회차와 과거100GB/300GB 확장안은 현재 실행 범위가 아니다.

v4.2 명세, [AGENTS.md](../../AGENTS.md), [총괄 운영 지침](../operations/X500_EXPERIMENT_COORDINATOR_CODEX_PROMPT.md)을 따른다. 과거 계획 원문은 [이전 EXPERIMENT_PLAN.md](../archive/maintenance/storage-path-01/before/docs/EXPERIMENT_PLAN.md)에 그대로 보존했다.

저장 경로는 2026-10-17 사용자 지시에 따라 로컬 프로젝트 `Data/`를 기본값으로 변경했다. 기존 통합 자료의 5개 파일은 바이트/해시 검증 후 로컬로 이동했다. 사용자 승인에 따라 일치하는 외장 원본만 제거했으며 receipt의 원래 계보 기록과 이동 전 상태 증거는 보존했다. 저장 위치는 새 수집 CLI와 학습 준비 옵션에서 선택할 수 있다. 이번 변경은 수집·학습 실행 승인이 아니다.

현재 폴더 배치에 따라 수집 코드는 `Data/generation/`, 시뮬레이션은 `simulation/`, 명세·기록은 `docs/`의 실제 경로를 사용한다. 호환 링크는 제거했다. 최신 1,200run 데이터의 전처리·공통 scaler와 기존 모델 비교 설정은 유지한다. 현재 소스·runtime 봉인의 갱신은 새 수집이나 학습을 승인하지 않는다.

2026-10-31 정리: 이전 데이터 대신 오늘 완료한 1,200run 통합본을 날짜 폴더에 보관한다. 이전 비교 초안·문서는 [정리 전 보존본](../data/reports/dataset-cleanup-20261031/before/)에 남기며, 최신 압축본·receipt·동결 증거는 내용을 변경하지 않는다. 이 경로·해시 갱신 자체는 실행 승인이 아니었다. 이후 사용자의 “순서대로 3번까지 진행해” 지시에 따라 위 서버 준비·게이트·본학습 범위를 별도로 승인 기록했다. 변경 전 계획은 보존한다.

## 2026-11-03: 최종 평가 자료 수집 메뉴 준비

최신 로컬 기록상18모델 개발 평가가 완료됐으며 최종 시험은 미사용이다. 현재1,200run의 역할은 train720/validation104/calibration104/정상 개발128/이상 개발144다. 이 자료를 최종 test로 재명명하지 않는다. 최신 모델 실행 상태·증거는 기존 `ai/experiments/CURRENT_STATE.json`과 해당 평가 보고서가 기준이며 위10월31일 준비 상태와 구분한다.

사용자는 정상·paired 대조·이상 수량을 맞추고 길이도 비슷하게 구성한 뒤, **실행은 직접 할 테니 기존 수집기에 새 번호로 추가하라**고 지정했다. 이에 기존 `./Data/generation/drone-pretrain`의4번에 `final324` 구성을 준비한다. 실제 수집·서비스·학습·보정·최종 추론은 이번 코드 준비 작업에서 실행하지 않는다.

구성은 별도 정상108(9조건×12), paired 정상108, 이상108(배터리 저항/추진 손실 각각existing/step/ramp ×18), 합계324run이다. paired는 기존 기준 호버 조건에서 정상1개+이상1개의108쌍이며 각각 별도 초기화 run이다. 일반480초/저SOC 정상12run180초, paired 양쪽480초다. 전체 정상216/이상108인 점과 동일 pair 내부의 상관을 유지한다. 총216개 계보 그룹은 난수·reset 설계의 그룹 수이며 실측 통계적 독립성 인증이 아니다. 수량은 사용자 선택 구성이고 충분한 검정력이나 낮은 오경보율의 정밀 추정을 입증한 표본수가 아니다.

별도 정상은 기존 정상 범위와 기본 경로를 재사용한다. 기존180/900초 복합·무작위 경로 봉인을480초로 바꾸지 않는다. 이번 세트의 그 경로 coverage 한계를 기록한다. 새로운 난수·계보를 사용하며 과거 보호 목록/기존 데이터·계획된 수집과의 seed/group 및 수집 후 raw identity 중복을 검사한다. 임의의 열화 강도·잡음·정상범위 변경은 없다.

2026-11-03 사용자 정정: 최종 수집도 기존 수집기와 동일하게 각run최대3회(최초1회+재시도2회)를 적용한다. 초기에 최종 메뉴만1회로 제한했던 설정을 수정한다. 개별 실패3회 소진 후 다음run은 진행하되 불완전 최종 세트는 봉인하지 않는다. 공유 자원 장애·사용자 중지는 재시도하지 않는다. 모든attempt를 같은case/seed/group 계보로 보존한다. 현재 실행 작업은 사용자 요청으로 중지하고 원래 설정·로그를 보존하며, 수정 정책은 새로 생성한 작업부터 적용한다. 재실행은 사용자가 직접 한다. 원시 archive/source/canonical hash, 실제 조건·길이, 품질·문맥/target 제외·무관측을 봉인하고 학습용 fit 흐름과 분리한다. 정규화 전 canonical을 만들며 새 scaler를 fit하지 않는다. 60초 문맥·gap reset·종료 이후 실제 model-score 노출은 후속 evaluator에서 별도 산정한다. 무관측 run을 제거하거나 학습/보정 자료로 자동 편입하지 않는다.

최종 평가 개봉 전에 사용 모델·기존 train scaler·보정 임계값·평가 protocol과 생산자/물리 설정 비교를 별도 고정한다. 모델 점수를 보며 수집량·조건·강도를 조정하지 않는다. 결과를 보고 설계를 변경하면 이번 세트는 사용된 평가 이력으로 남기고 변경 모델의 독립 최종 시험을 새로 정한다.


## 2026-11-03 최종 평가 수집 설계 개정: final324-diverse-v2

사용자는 기동·환경을 확장한324run 제안의 코드 반영을 승인했다. 실제 수집 실행·재실행은 사용자가 한다. 당시 실행 중인 `final-20261103-162350`은 끝날 때까지 소스/설정/기록을 유지하고, 종료 후 새 코드만 적용한다. 기존 작업은 이전 호버판 설계이며 새 설계 자료로 간주하지 않는다. 이전 보고서와 model/scaler/threshold·원자료·실패 증거는 보존한다.

### 고정 구성과 명목 길이

| 군 | 구성 | 수량 | 길이 |
|---|---|---:|---|
| 별도 정상 |6기동×6환경×3새seed|108|일반90×480초+저SOC18×180초|
| paired 정상 대조 |각 이상과 seed·환경·전체 경로·길이 일치|108|동일|
| 이상 |기존6종×6기동×3환경|108|동일|

군별 명목 임무시간46,440초(12.9시간), 전체38.7시간이다. 이는 이륙·착륙을 제외한 요청 시간이며 유효 관측/점수시간이나 생성 wall time이 아니다. 정상은 전체216개, 이상108개다.108pair와 별도 정상108개의216lineage group이며 각 pair 안의 두 run은 상관된다. 창 개수나 시점 수를 독립 표본 수로 해석하지 않는다.

기동은 호버·상승/하강·수평 가감속·yaw회전·복합·제한된 부드러운 난수다. 기본 기동은 실제 경로 kind로 지정하며 이름만 바꾸지 않는다.480초 복합은 호버→상하→수평→회전→figure8→ellipse 각80초, 구간 양끝은 기존 부드러운 연결을 사용한다.180초 저SOC 복합은 상하→수평 각90초로 압축한다. 짧은 복합에6기동을 모두 넣었다고 주장하지 않는다. 난수는 기존 속도·고도·가속도·jerk·yaw bound와 바람 보정 속도 제한을 그대로 적용하고 전체 경로를 실행 전에 봉인한다.

| 환경 | 변경 변수 | 나머지 기준값 |
|---|---|---|
|기준|SOC .87–.95|25°C, 무풍, 무탑재|
|중간SOC|SOC .65–.75|동일|
|저SOC|SOC .40–.50,180초|동일|
|바람|평균1–2m/s, gust sigma .1–.3m/s|SOC .9,25°C, 무탑재|
|저온|10–18°C|SOC .9, 무풍, 무탑재|
|탑재량|중앙 탑재 .1–.2kg|SOC .9,25°C, 무풍|

별도 정상의3seed는 범위의 하·중·상3등분에서 각각 추출한다. paired는 아래 균형 부분배치를 각 고장종에 반복한다. 각 칸은 별도seed의1pair이며, 한 기동의3칸은 같은 조건3반복이 아니다. 각 환경은 고장종별3pair, 각 기동도3pair다. 완전한6기동×6환경 교차 시험이 아니다.

|기동|환경1|환경2|환경3|
|---|---|---|---|
|호버|기준|저SOC|저온|
|상하|기준|중간SOC|탑재량|
|수평|중간SOC|바람|탑재량|
|회전|기준|바람|저온|
|복합|중간SOC|저SOC|저온|
|난수|저SOC|바람|탑재량|

### 물리 이상과 시각

배터리 저항 증가·추진 손실 각각 existing/step/ramp의 기존6종을 유지한다. 건강 파라미터와 고장 강도는 기존 producer 정의에서 가져오며, 새로운 `operation_variation`은 SOC·온도·탑재·바람·경로만 허용한다. 정상용 `normal_variation`을 이상 런에 혼용하거나 정상 파라미터로 열화를 덮어쓰지 않는다. 공통 neural입력·raw schema·물리계수·학습 scaler/threshold는 변경하지 않는다.

step/ramp는 일반480초에서 경로 시작 후180/240/300초를 고장종별각5회, 저SOC180초에서100초에 요청한다. 시각을 특정 환경에 고정하지 않으며 ramp20초는 유지한다. 기존 boot기준`fault_at_s`와 새`fault_after_mission_s`는 동시에 사용할 수 없다. 새 기준0초는 실제`airborne`고도 통과 시각`reached_t`이며 현재 경로의 시간원점과 같다. mission상태 전이는 이보다 약2초 뒤일 수 있어 두 시각을 혼동하지 않는다.

실행 시 실제 시작시각에 offset을 더해 네이티브 고장 스케줄을 예약하고 ACK를 확인한다. evaluation metadata에 요청offset·원점·예약시각·실제적용시각을 남기며, 탐지 지연은 실제 truth 적용시각을 기준으로 계산한다. existing은 최초 유효 판단 이후 인지율/인지시간으로 별도 집계한다. 예약시각을 물리 onset 정답으로 대신 쓰지 않는다.

### 분할·인수·품질 및 평가 경계

- 새seed/group을 사용하고 기존 task·동결manifest의seed/group/rawID/hash와 겹치면 차단한다. 같은pair·원비행·재시도는 분할하지 않는다. 기존 자료의 등록 범위 밖 외부 데이터까지 검사했다고 주장하지 않는다.
- 모든 run 최대3시도(최초1+재시도2), 같은case/seed/group을 유지한다. 실패 이유와 각시도 계보를 보존한다. 공유 저장소·메모리 오류 및 사용자중지는 재시도하지 않는다.324run완료/품질인수가 안 되면 완전한 최종 세트로 봉인하지 않는다.
- 시뮬레이션 담당의 생성wall time/저장량/실패·재시도/실제길이/추적오차/고장 적용시각과 AI 담당의 결측/값유지/공백/60초 통계문맥/예측history/horizon/종료mask/accepted target을 함께 기록한다. 유효시간은 단순히 요청길이−60초로 계산하지 않는다. 결측 후 문맥 재형성, 원시/정제/점수 구간별 exposure를 구분한다.
- 저SOC100초 onset의 유효 정상 전구간과 ramp완료 뒤 관측길이는 실제 품질 요약으로 확인해야 한다. 목표 관측이 없는 run을 성능 계산에서 조용히 제외하지 않는다. 사유·분모·미관측률과 실패를 별도 보고한다.
- 수집은 기존 causal전처리와 독립 품질 검산 후 정규화 전canonical을 봉인하고 정지한다. 학습·scaler fit·보정·탐지 점수 계산은 실행하지 않는다. 최종평가에서는 이미 동결한 모델/scaler/threshold/후처리/평가규칙을 사용한다.
- 조건별 정상 falsealarm episodes/hour, 이상 검출/미검출, step/ramp지연, existing인지, 관측coverage, paired대조를 따로 보고한다. 전체시간/평균 하나로 저SOC180초의 짧은관측을 숨기지 않는다. 신뢰구간은 최소pair/group단위를 보존한다.
- 수집 가능성·유효 exposure는 검증 전 제안값이다. 새 조건이 실제 비행/품질검사를 통과하는지는 사용자의 실제 수집에서 확인한다. 모델 점수를 보고 설계·threshold·모델을 바꾸면 해당 자료는 개발용으로 재분류하고 새 최종 독립자료를 봉인한다. 예비/기존호버 자료를 최종독립 검증이라고 재표시하지 않는다.

실제 로그 참고 원칙은 앞 절을 유지한다. 이번 구현에서 새 실측로그를 읽거나 물리 정확도를 실측 검증한 것으로 주장하지 않는다. 다른 기체 파형에 맞추어 물리계수·잡음·정상범위·열화강도를 조정하지 않았다.


### 구현 중 확인한 기존 수집 후처리 오류

기존 작업`final-20261103-162350`의ledger에서324run이`ARCHIVED_PENDING_QUALITY`였고, final-admission을 통과한 뒤 final-extract에서`consumer/final-test/<task>`부모 폴더 누락으로`FileNotFoundError`가 발생한 것을 상태/단위로그로 확인했다. 이는 비행 생성 실패와 구분한다. 운영 소스를 실행 중 수정하지 않았고, 새 코드에서 저장루트·symlink검사를 유지하며 부모 폴더를 독점 생성한 뒤 기존 추출기를 호출하도록 수정했다. 작은 합성 압축본으로 동일 오류를 재현하고 실제 추출기·truth제외·기존 파일 보존을 검증한다. 기존 작업의 실패/원자료를 삭제·덮어쓰기하거나 후처리를 자동 재실행하지 않는다.


### 수집 실행 환경의 미해결 시험 이력

이번 수정 검증에서도 수집용Python3.12.3프로세스의`SIGSEGV`와 정상적인`zip`순회에서의 비정상 unpack 오류가 각각 관찰됐다. 과거`docs/data/reports/capsule-recovery-20261031/allocator-diagnosis.json`에도 같은 종류의 미확인 메모리/객체 오류가 기록돼 있다. sandbox밖에서도 객체 오류가 발생했으므로 sandbox원인으로 단정하지 않는다. 독립적인 생성/수치 시험 통과는 이 문제의 해결 증거가 아니다.

해당 실패를 삭제하거나 반복 통과로 대체하지 않는다. 새 설계 코드의 검증과 수집용Python3.12장시간실행 안정성은 구분한다. 경로 수식·물리계수·Python환경을 추측으로 변경하지 않았으며, 실제 대규모 수집의 안정성을 보증하지 않는다. 이번 작업에서 수집·후처리·학습을 자동 시작하거나 실패 작업을 재실행하지 않는다.


## 2026-11-03 실제 final324-diverse-v2 수집·정제 종료

회차`final-20261103-185815`는21:23 KST에`FINAL_DATASET_SEALED_STOPPED`로 종료됐다. 정상108·paired대조108·이상108의324run,108pair·216lineage group이며, 실행 전 계획과 봉인 목록의 조건/역할/seed/group/길이 일치를 확인했다. 각 군의6기동·6환경별18run과 요청 임무시간46,440초가 유지됐다. 전체 재수집·계획 변경 없이 수집→인수→추출→전처리→독립 품질검산이 각각324개 완료됐다.

봉인 자료는`Data/processed/final-test/final-20261103-185815/sealed-manifest.json`이며`seal.json`의manifest SHA256와 일치한다. accepted target은 별도 정상43,074·paired대조43,057·이상43,100개이고 target0개run은 없다. 이는 전처리 품질검산의 관측 적격성 결과이며, 모델 점수/경보 exposure·탐지 성능·표본 독립성·학습 데이터 충분성의 검증이 아니다. 겹치는창/target개수를 독립표본으로 세지 않는다.

비행3case가2번째시도에서 완료됐다(recording_stalled1개, sample after available시각검증2개). 전처리5run도2번째시도에서 완료됐다(SIGSEGV4개, unknownopcode1개). 실패 로그·부분 산출물·계보를 보존했으며 최대3시도 한도를 소진한 단위는 없다. Python3.12실행 환경 문제와 시각 오류의 근본 원인은 미해결로 남긴다. 재시도 성공과 최종 검산 통과를 원인 해결로 보고하지 않는다.

기존 학습/평가자료·모델·scaler·threshold는 보존했고 이번 회차에서는scaler fit·학습·보정·모델 평가를 실행하지 않았다. 새 자료는 성능 튜닝에 사용하지 않은 채 봉인했다. 향후 모델/scaler/threshold/후처리/평가규칙을 동결한 별도 최종 평가 인수 절차를 따른다. 이번 감시는 데이터 봉인 완료로 종료한다.

사용자의 이전 자료 폐기 승인에 따라 이전 중단회차`final-20261103-162350`의 성공 원본324개만 압축본과 완전한 내용·해시 일치 검증 후 정리해121.09GB를 회수했다. 학습/평가 등록자료·이전 압축본27.19GB·실패 시도/기록·현재 자료는 보존했다. 근거는`Data/operations/pretrain/monitoring/final-20261103-185815/completion-report.json`과`cleanup-completion-report.json`에 있다.


## 2026-11-07 FINAL16 신규16편 연결 점검 승인

총괄 승인 `ai/reports/final16-20261107/attempt-01/authorization.json`(SHA256 `05017f2adc41caf14906ecab4ce82e02d9313b097da8be4609e18a3b9ca3041b`)에 따라 신규16편만 별도 회차로 준비한다. 이는 본학습1,674런/LR108 준비와 별개인 180초 입력·수집 연결 점검이며 독립 탐지 성능평가 자료로 부르지 않는다. 본학습/LR/예비 메뉴는 변경하거나 자동 연결하지 않는다.

- 기존 short180 정의의 pair01..08만 사용한다. 정상8+이상8, 독립실행8쌍이며 standalone/pair00은 제외한다. root가 동결한 원정의와 cohort만 바꾼 예상 manifest를 모두 보존하고 대조한다.
- seed_base `2938306477`은 총괄이 새 OS 난수로 정해 승인에 동결했다. namespace `final16-20261107-01`과 새 원장으로 시작하고, 이전45편·재시도·과거품질거절·원시/전처리 재사용은 허용하지 않는다.
- 요청 비행길이180초, 비행16동시, 요청RTF2, 비비행 전체최대4개이며 모든 비행 종료 뒤 처리한다. 각 case 물리시도1회·총최대16회다. 첫 실패에서 신규 투입과 후속작업을 멈추고 이미 실행 중인 작업만 종료시킨다. 자동 재비행·추가case·다음회차는 없다.
- purpose/cohort `final16_development_reference`, usage `final16_input_check` 및 전용 manifest/ledger/인수 schema를 사용한다. fit/calibration/final_test 사용을 영구 제외하며 19입력·단위·전처리 계산·전체mission tail·기존 물리/품질 기준은 그대로다.
- 실제 수집→원시평가→압축/추출검증→기존 두19전처리→독립검산→실제8pair→실제AI16개 인수→서비스종료/잠금·포트반환까지만 수행한다. 학습/모델forward/scaler/threshold/GPU/탐지 성능평가는 수행하지 않는다.
- 준비 task는 `Data/operations/pretrain/tasks/final16-20261107-01-16`, 부모 유한 flow는 `final16-20261107-01`이다. 실제 manifest/AI prerequisites와 전체16 등록경로를 먼저 점검하고 총괄 실행 전 최종 확인 뒤 Data만 한 번 제출한다.

원시·실패·과거 승인/원장/결과는 보존한다. 정적검사 통과는 실제16편·물리 정확도·학습 충분성·탐지 성능 통과로 확대하지 않는다. 검증과 제출 상태는 `docs/data/reports/final16-20261107/attempt-01/`의 준비·실행 인계에 기록한다. 총괄만 CURRENT_STATE를 수정한다.
