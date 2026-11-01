# 설계 자산·문서 안내

현재 구현을 설명할 때는 [README의 구조도·사용법](../README.md)과 [v0.4 구조](architecture.md)를 사용합니다. 아래 PNG는 설계 변경 이력이며 현재 구현이나 실제 모델 검증의 증거가 아닙니다.

## 인계 시 읽을 문서

| 확인할 내용 | 기준 문서 |
| --- | --- |
| 실행·재생·실시간 연결·NPZ 내보내기 | [README](../README.md) |
| 외부 전처리 → 공통 프레임 → 화면·NPZ·모델 입력 참조 | [현재 구조](architecture.md) |
| 특징·단위·mask·관측 나이·저장 파일·추가 API | [전처리·NPZ 계약](processed-npz-contract.md) |
| LSTM·Transformer 입력 참조와 독립적인 결과 검증 | [입력·분석 계약 v2](v2-contracts.md) |
| 단위와 미지원값 해석 | [데이터 사전](data-dictionary.md) |
| 시험 입력·재생·저장·복구·화면 검증 범위 | [전처리·파일 보관 검증 기록](validation-v3.md) |

실제 공용 전처리기와 모델 서비스는 인계 후 계약을 맞춰 연결합니다. 시험 API 응답과 합성 관측으로 확인한 화면 동작을 실제 모델 정확도·실기체 비행 검증으로 해석하지 않습니다. 학습은 외부에서 수행합니다.

## 이전 구조도와 생성 프롬프트

| 자산 | 설계 당시 내용·현재와의 차이 | 생성 기록 |
| --- | --- | --- |
| [DB 사용 초기 구조도](arion-final-target-architecture.png) | SQLite 메타데이터·결과 저장을 포함했던 초기안입니다. 현재 파일 저장 구조로 대체되었습니다. | [원본 프롬프트](arion-final-target-architecture.prompt.txt) |
| [DB 제거 구조도](arion-final-target-architecture-no-db.png) | JSON·JSONL 파일 저장으로 변경한 중간안입니다. NPZ는 외부 학습에만 표시되어 있어, 이후 구현한 실시간 전처리 프레임·60초 NPZ 보관·내보내기 경로가 빠져 있습니다. | [수정 프롬프트](arion-final-target-architecture-no-db.prompt.txt) |

파일명에 `final-target`이 있어도 최신 설계를 뜻하지 않습니다. 두 이미지는 AI로 생성한 발표용 초안으로, 화살표·범례 등 세부 표현은 현재 구조 문서와 다를 수 있습니다. 프롬프트는 생성 당시의 요청을 보존한 기록이며 현재 개발 지시나 지원 기능 명세가 아닙니다.

## 화면 자료

| 자산 | 용도 |
| --- | --- |
| [균형형 워크스페이스](dashboard-balanced-preview.jpg) | 3D와 LSTM·Transformer 비교 배치 참고. [검증 기록](validation-balanced.md)과 함께 봅니다. |
| [전처리 실시간 화면](dashboard-processed-live.jpg) | 시험 전처리 입력의 관측·저장 상태 표시 참고. |
| [NPZ 재생 화면](dashboard-processed-npz.jpg) | 저장된 전처리 기록의 재생·비교 화면 참고. |

화면 자료는 촬영 시점의 시험 상태입니다. 실제 연결 상태·값·제공자 응답은 실행 환경에 따라 달라지며, 최신 검증 범위와 제한은 각 검증 기록을 확인합니다. 원본 비행 로그·학습 NPZ·로컬 DB와 `.data` 기록은 공개 저장소에 포함하지 않습니다.
