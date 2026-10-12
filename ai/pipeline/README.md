# 모델·데이터·평가 연결

학습·추론·독립 평가가 공유하는 역할 판정·고정 데이터 계약·backend·GPU 할당 검사를 제공한다. 사용자 명령은 `./ai/drone-ai`다.

`runtime.py`는 `DRONE_AI_PROFILE`로 전달한 프로필의 Python·패키지와 실제 Slurm 소유권·노드·가시 GPU1개를 확인한다. 검증 전에 GPU 학습을 시작하지 않는다. 로그인 노드 CUDA 계산, `CUDA_VISIBLE_DEVICES` 변경, 공유 환경 자동 설치는 허용하지 않는다.

`control.py`의 과거 회차 승인 hash와 schema는 보존 증거 검증용이며 새 작업 승인에 쓰지 않는다. 개발 평가와 잠긴 최종 시험 자료를 분리하고 truth를 모델 입력에 넣지 않는다. 새 실행은 [서버 사용 안내](../../docs/ai/PORTABLE_SETUP.md)의 명시적인 준비·시험·승인 절차를 따른다.
