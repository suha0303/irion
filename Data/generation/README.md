# 데이터 생성 코드

- `drone-data`: 환경 설치·진단·소스 export·수집 메뉴의 공통 진입점.
- `drone-pretrain`: 설치 후 같은 수집·전처리 메뉴를 직접 여는 명령.
- `pretrain/`: 구성·seed·선택 게이트·수집·추출·전처리·공통 scaler 집계·독립 검산·압축·진행 화면.
- `tools/`: 기존 finite collector와 공유 프로세스·압축·검증 도구.
- `pretrain/resources/`: 현재 수집 템플릿·건강 조건·동결 전처리 패키지.
- `.venv/`: 설치 명령으로 생성하는 Python 3.12 CPU 처리 환경. Git에 넣지 않는다.

```bash
./Data/generation/drone-data setup --simulation
./Data/generation/drone-data doctor
./Data/generation/drone-data menu
```

프로젝트 루트에서 위 명령을 실행한다. 스크립트가 실제 설치 위치에서 프로젝트를 찾으므로 사용자 이름이나 `/home/sa/CapstoneV3` 경로를 맞출 필요가 없다. 실행에 필요한 Python 경로는 자동 설정한다. 공통 전처리 알고리즘과 모델 입력 계약은 `../../ai/src/capstone_ai/`를 재사용하고 학습용 GPU 환경은 변경하지 않는다. 자세한 사전 조건은 [설치 안내](../../docs/data/PORTABLE_SETUP.md)를 따른다.

작업 상태·로그는 프로젝트 `Data/operations/pretrain/`에 저장하고 최종 데이터는 메뉴에서 선택한 저장소에 둔다. 세부 설정과 실패·재시도 정책은 [수집 안내](pretrain/README.md)를 따른다. 게이트는 선택 사항이고 비행16/나머지4가 기본이다. 장시간 반복은 Python/셸 실행기가 맡으며 LLM 호출은 없다.
