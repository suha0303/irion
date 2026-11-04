# 최종 평가 데이터 수집·정제 결과

2026-11-03 완료한 `final324-diverse-v2` 회차 `final-20261103-185815`의 보존 결과다.

| 구분 | run | accepted target | target 없는 run |
|---|---:|---:|---:|
| 독립 정상 | 108 | 43,074 | 0 |
| paired 정상 대조 | 108 | 43,057 | 0 |
| 이상 | 108 | 43,100 | 0 |

108쌍·216개 계보 그룹이며, 각 군의 6기동·6환경에 각각 18run을 배치했다. paired 정상과 이상은 서로 의존한다. accepted target 수를 독립 표본 수로 해석하지 않는다.

- [completion-report.json](completion-report.json): 조건별 구성, 수집·추출·전처리·품질 완료 상태와 재시도 기록.
- [sealed-manifest.json.gz](sealed-manifest.json.gz): run·계보·소스·canonical 파일 해시를 포함한 원본 manifest의 무손실 압축본.
- [seal.json](seal.json): 압축 해제한 manifest의 SHA-256과 봉인 상태.
- [AI 최종 평가](../../../ai/results/final-evaluation-20261103/README.md): 동결 모델과 임계값을 사용한 후속 평가 결과.

비행 3case와 전처리 5run은 두 번째 시도에서 완료했다. 당시 시각 오류와 Python 실행 환경 오류의 근본 원인이 해결됐다는 뜻은 아니다. 이전 실패와 재시도 계보는 완료 보고서에 보존되어 있다. 데이터 봉인은 수집·전처리 무결성 확인이며 탐지 성능 인증은 아니다.

원시 비행·canonical 데이터 자체는 포함하지 않는다. JSON의 절대 경로는 원래 실행 당시의 출처다. 이 저장소에서는 경로를 실행하거나 원자료 존재를 가정하지 않는다. 복사 파일의 출처와 해시는 [manifest.json](manifest.json)에 있다.

```bash
gzip -dc sealed-manifest.json.gz | sha256sum
```

결과를 `seal.json`의 `manifest_sha256`과 비교한다.
