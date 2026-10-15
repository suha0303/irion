# X500 현재 실험 계획

기준일: 2026-10-17 KST. 현재 대상은 `ROUND_001/R6_1200_8MODEL_S0`의 구현·준비 상태 점검이다. **본학습·GPU 게이트·추가 수집은 시작하지 않았다.** 실행 승인은 준비 결과와 유한한 GPU 예산을 확인한 후 별도로 받는다.

## 현재 데이터

로컬 기본 저장소의 `processed/pretrain/merge-1200-20261017-01/prepared-portable/`를 인수한다. 저장소 루트는 `/home/sa/CapstoneV3/Data/`다. 압축 데이터는 5,597,050,336바이트이며, 복원 크기는 약79GB다. 5.6GB 압축본 크기를 원시 비행로그 크기와 혼동하지 않는다.

분할은 정상 train720 / validation104 / calibration104 / 정상 개발 평가128 / 이상 개발 평가144run이다. 원래 run·그룹·수집 seed·분할을 보존한다. 정상 train만으로 만든 기존 공통 scaler를 사용하며, train 창 수는 보존된 품질 기록상560,885개다. 재수집이나 다른 분할로의 이동은 승인 범위가 아니다.

[고정 데이터·모델 계획](../../ai/experiments/ROUND_001/compare-8-seed0-01/plan.json)과 압축본 receipt가 정확한 데이터 해시의 기준이다. 실제 인수 시 archive/member 해시, 품질 증거, train-only scaler, 역할·lineage·NPZ를 검증한다. 잠근 최종 시험 자료를 인수하거나 성능 확인에 사용하지 않는다.

## 비교 구성

LSTM과 Transformer 각각 hidden64·2층 /128·2층 /128·4층 /256·4층, 총8종이다. 층 수는 encoder와 decoder 각각의 수다. Transformer heads는4/4/4/8이다. persistence와 ridge_forecast를 더해 총10종의 결과를 비교한다. 전체 명세의 모델 비교 목표는 유지한다.

학습 seed0, batch256, FP32, AMP 비활성, 최대60epoch, patience10, dropout0.1, AdamW lr0.001·weight_decay0.00001로 고정한다. 각 신경망은 새 seed0 가중치에서 시작한다. 과거 회차·게이트·다른 구성의 가중치를 이어받지 않는다. 수집 seed는 바꾸지 않는다. 추가 seed·다른 구성은 후속 승인 대상이다.

각 정확한 구성의 실제 GPU 메모리와 별도 프로세스 복원·export 검사를 통과한 후 본학습한다. 실패하면 가중치·로그를 보존하고 전체 비교를 멈춘다. batch를 줄이거나 재시도하지 않는다. 짧은 메모리 게이트 속도를 전체 학습 처리율로 제시하지 않는다.

## 자원과 실행 경계

기존 승인된 `ssh knu` 접속과 mamba 환경을 사용한다. Slurm p02/DIS01/단일 노드/GPU1개, 프로젝트 동시 GPU 작업1개를 지킨다. 일반 RAM 요청은64GiB이며 실제 가용량·추정 피크와 GPU 조건을 할당 안에서 다시 검사한다. CUDA_VISIBLE_DEVICES를 바꾸거나 로그인 노드에서 GPU 연산하지 않는다.

fit/evaluation/queue 시간은 실제 제출 전 유한한 예산으로 동결한다. 안내문의48시간/12시간은 예시 상한이며 예상 소요시간이나 실행 승인이 아니다. 현재 서버 자원이 확보됐다고 주장하지 않는다. 추가 환경 설치·업그레이드는 하지 않는다.

신규 준비 작업은 선택한 저장소의 별도 폴더를 사용하며 기본은 프로젝트 내부 `Data/`다. 외장 저장소를 명시한 경우에는 마운트 유실·경로 이상에서 다른 디스크로 대체하지 않고 실패한다. 원래 압축 데이터와 기존 산출물은 변경하지 않는다.

승인된 절차는 구성별 게이트 →10종 순차 학습/fit → 정상 보정 → 개발 추론 봉인 → 학습 할당 반납 → 독립 개발 평가 → 평가 할당 반납 → 보고 후 정지다. validation으로 checkpoint를 선택하고, calibration은 정상 자료만 사용한다. 개발 이상 정답은 봉인된 추론과 성공적인 학습 할당 반납 확인 뒤 evaluator만 결합한다.

예측 오차, 정상 오경보 episodes/hour, 사건 검출·미관측/불가용 사례, onset/recognition delay, 공통 유효 origin의 예측 성능을 함께 보고한다. 낮은 성능 때문에 재학습·재보정·추가 데이터/seed를 자동 실행하지 않는다. 두 할당의 실제 종료와 보고 검증 후에만 REPORT_READY_STOPPED로 표시한다.

## 현재 명령·상태·증거

- 수집·전처리: `./Data/generation/drone-pretrain`. 현재 기본은300run, 비행16·그 밖의 처리4병렬,2배속, 게이트 선택이다. 이번 점검은 수집 승인이 아니다.
- 새 비교: `./ai/x500-compare --help`. [준비·전송·서버 CLI 순서](../../ai/gpu_train_next/README.md)를 따른다.
- 수집 상태는 `Data/operations/pretrain/latest.json`과 해당 작업의 `operator/state.json`을 사용한다. 비교 실행 후에는 해당 작업의 `prepared.json`, `controller.json`, release/evaluation/completion 파일을 사용한다. 중복된 전역 상태 관리 체계를 만들지 않는다.
- [정리 기록](../archive/maintenance/workspace-cleanup-20261017-01/report.json), [추가 점검 기록](../archive/maintenance/workspace-cleanup-20261017-01/review-02/)를 보존한다. 코드·입출력 계약의 C9 연결은 이번 점검에서 변경하지 않았다.

프로그램의 유한한 반복과 CLI 화면 갱신은 LLM 없이 수행한다. 화면 닫기와 OS/서버 작업 종료는 다르다. 사용자의 결과 확인 요청 전까지 LLM 감시·예약 호출·완료 이벤트 자동 재개를 만들지 않는다. 다음 회차와 과거100GB/300GB 확장안은 현재 실행 범위가 아니다.

v4.2 명세, [AGENTS.md](../../AGENTS.md), [총괄 운영 지침](../operations/X500_EXPERIMENT_COORDINATOR_CODEX_PROMPT.md)을 따른다. 과거 계획 원문은 [이전 EXPERIMENT_PLAN.md](../archive/maintenance/storage-path-01/before/docs/EXPERIMENT_PLAN.md)에 그대로 보존했다.

저장 경로는 2026-10-17 사용자 지시에 따라 로컬 프로젝트 `Data/`를 기본값으로 변경했다. 기존 통합 자료의 5개 파일은 바이트/해시 검증 후 로컬로 이동했다. 사용자 승인에 따라 일치하는 외장 원본만 제거했으며 receipt의 원래 계보 기록과 이동 전 상태 증거는 보존했다. 저장 위치는 새 수집 CLI와 학습 준비 옵션에서 선택할 수 있다. 이번 변경은 수집·학습 실행 승인이 아니다.

현재 폴더 배치에 따라 수집 코드는 `Data/generation/`, 시뮬레이션은 `simulation/`, 명세·기록은 `docs/`의 실제 경로를 사용한다. 호환 링크는 제거했다. 기존 1,200run 데이터·전처리·공통 scaler와 모델 비교 설정은 유지한다. 현재 소스·runtime 봉인의 갱신은 새 수집이나 학습을 승인하지 않는다.
