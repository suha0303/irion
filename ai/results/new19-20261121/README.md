# V0.2 모델 최종 결과 — 2026-11-21

완료된 8종 모델의 선택 근거·학습 결과·보정 기준·정상 250런 및 이상 200런 평가를 보존한다. 원본 JSON의 내용과 해시는 변경하지 않았다.

| 모델 | 정상 경보 / 250 | 이상 탐지 / 200 |
|---|---:|---:|
| inflight-lstm-128x4 | 30 | 184 |
| inflight-lstm-64x2 | 27 | 160 |
| inflight-transformer-128x4 | 24 | 152 |
| inflight-transformer-64x2 | 26 | 159 |
| whole-run-lstm-128x4 | 13 | 200 |
| whole-run-lstm-64x2 | 9 | 200 |
| whole-run-transformer-128x4 | 1 | 200 |
| whole-run-transformer-64x2 | 7 | 193 |

모델 선택은 구조·크기별 정상 검증 손실을 기준으로 했다. 경보 수는 런 단위이며 프레임 정확도와 다르다. 강풍 조건의 오경보와 추진계 단독 이상 미탐은 [조건별 결과](live/new19-inflight-live-verification-20261121/final-results-by-condition.json)에 남아 있다.

- [모델 선택과 해시](models/manifest.json): 각 모델의 `fit`, `calibration`, `final` JSON.
- [실시간·32회 앱 검증](live/README.md): 실시간 입력·예측·재생 검증과 정상 16회/이상 16회 평가.
- [실기체 보정·한계](real-flight/README.md): 기체별 정상 로그 보정 및 별도 이상 반응 감사.

이 디렉터리는 결과 증거 패키지다. 가중치, 원시 비행 로그, 학습 배열, 체크포인트, 서버 로그는 포함하지 않는다. JSON의 원래 실행 경로는 출처를 나타내며 이 저장소의 다운로드 경로가 아니다. 모델 매니페스트가 가리키는 가중치는 별도로 준비해야 실행할 수 있다. 각 하위 `SOURCE_HASHES.json`은 게시 파일과 원본 경로·SHA-256을 연결한다.
