# AI 운영 문서

서버에서 클론한 뒤 `./ai/drone-ai`로 서버·환경·데이터 경로를 설정한다. [서버 사용 안내](PORTABLE_SETUP.md)에 환경 준비·진단·짧은 GPU 학습/저장/복원 시험·진행 화면과 별도 본학습 절차를 정리했다.

[현재 코드](../../ai/README.md), [학습 실행기](../../ai/learning/README.md), [현재 실험 계획](EXPERIMENT_PLAN.md)을 확인한다. 본학습 비교 범위는 1,200run·seed0의 신경망8종과 baseline2종이다. 짧은 기능시험은 본학습·성능 검증을 대신하지 않는다.

기존 모델·데이터·통과/실패 증거를 보존한다. 변경 전 안내문은 `docs/archive/maintenance/ai-portability-20261018-01/`에 보존하며 실제 GPU 결과는 해당 실행의 상태·결과·Slurm 반납 기록으로 확인한다.
