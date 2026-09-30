# 데이터 생성·학습 전 처리·데이터셋

| 경로 | 목적 |
|---|---|
| [generation/](generation/) | 수집·전처리·검산 콘솔과 실행 코드 |
| `processed/pretrain/<task>/prepared-portable/` | 검증된 최종 무손실 압축 데이터셋 |
| `archives/`, `consumer/`, `streamed/` | 선택한 작업의 압축 원시·추출·묶음 처리 공간 |
| `gpu1200/`, `speed-tests/` | 사용자가 실행한 학습 준비 복원 공간·배속 시험 결과 |

최종 데이터셋은 `dataset.tar.xz`, 평가자 전용 정답 압축본, receipt와 보존 기록으로 구성된다. 데이터셋은 개인 산출물이며 소스 클론에 포함하지 않는다. 이 컴퓨터의 기존 1,200run 통합 자료와 split·scaler·hash는 보존한다. 압축본과 실제 복원 용량은 다르므로 복원 전에 receipt의 크기와 저장 공간을 확인한다.

```bash
./Data/generation/drone-data setup --simulation
./Data/generation/drone-data doctor
./Data/generation/drone-data menu
```

저장 폴더는 메뉴에서 선택한다. 기본값은 이 `Data/` 폴더이며 `generation/` 소스와 데이터 출력 하위 폴더를 구분한다. `generation/.venv/`는 정제 전용 CPU 환경이고 `operations/pretrain/`은 로컬 작업 설정·상태·로그·소유권 기록이다. 다른 컴퓨터에는 설치 환경이나 실행 중 작업 상태를 복제하지 않고 새 작업을 만든다.

기본 stream_compact는 성공 산출물의 복원 가능성을 검증한 뒤 작업 소유 원시·중간 파일만 정리한다. 실패 증거·잠금·claim은 임의 삭제하지 않는다. 수집 중 화면을 닫아도 user systemd 작업은 유지되며 PC 종료·로그아웃의 영향은 [운영 안내](../docs/data/PORTABLE_SETUP.md)를 확인한다.

시뮬레이터 내부 임시 원시는 `../simulation/runs/`, 배터리 공개 참조자료는 `../simulation/data/public/`다. 프로그램은 LLM 감시 없이 수집·전처리·검산 후 정지하며 본학습은 하지 않는다. [설치 안내](../docs/data/PORTABLE_SETUP.md), [수집 안내](generation/pretrain/README.md)를 따른다.
