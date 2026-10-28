# AI 학습·보정·평가

사용자 명령은 프로젝트 루트의 `./ai/drone-ai`다. 서버에서 클론한 뒤 서버 정보·기존 Python·데이터·출력 경로를 직접 설정한다. 도움말·설정·작업 화면은 시스템 Python3으로 시작하며 기존 `ai/.venv`를 요구하지 않는다.

```bash
./ai/drone-ai configure
./ai/drone-ai setup --profile default
./ai/drone-ai doctor --profile default
./ai/drone-ai smoke --profile default --root /absolute/output/smoke-01
./ai/drone-ai watch --root /absolute/output/smoke-01
```

`setup`과 `doctor`는 기존 환경을 읽기 전용으로 검사한다. 공유 환경 설치나 GPU 제출은 없다. `smoke`만 명시적으로 짧은 GPU 학습·체크포인트 저장·별도 프로세스 복원을 시험한다. 본학습·탐지 성능 평가가 아니며 시험 가중치를 본학습에 이어 쓰지 않는다.

| 경로 | 목적 |
|---|---|
| `console/` | 프로필·진단·짧은 GPU 시험·통합 진행 화면 |
| `learning/` | LSTM·Transformer·학습·저장/복원·모델 비교·GPU 기록 |
| `execution/` | 공통 경로/해시·상태·baseline·보정·개발 평가 |
| `pipeline/` | 데이터/역할 계약·GPU 할당과 환경 검증 |
| `src/capstone_ai/` | 기존 데이터 처리·예측·점수·독립 evaluator |
| `configs/`, `remote/`, `handoff/` | 비교 계약·환경 검증·보존 인수 패키지 |
| `tests/`, `artifacts/`, `experiments/`, `reports/` | 회귀시험·모델/증거·실행 기록 |
| `.local/profiles/` | Git 제외 컴퓨터별 설정; 인증 비밀 저장 금지 |

현재 본학습 비교는 1,200run, seed0, LSTM·Transformer 각각 hidden64·2층 /128·2층 /128·4층 /256·4층과 persistence/ridge_forecast, 총10종이다. 별도 회차 승인이 필요하다. 기존 모델·데이터·증거를 보존하며 가중치를 자동 누적하지 않는다.

명령과 정지 규칙은 [서버 사용 안내](../docs/ai/PORTABLE_SETUP.md), 비교 범위는 [현재 실험 계획](../docs/ai/EXPERIMENT_PLAN.md)을 따른다. 데이터 생성은 [Data/generation](../Data/generation/README.md)의 별도 실행기다.
