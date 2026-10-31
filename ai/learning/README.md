# 모델 학습·비교

사용자 명령은 `./ai/drone-ai`다. 직접 설정한 서버 프로필의 Python·경로를 사용하고 기존 모델·학습·평가 수학을 유지한다.

| 파일 | 역할 |
|---|---|
| `models.py`, `training.py` | LSTM/Transformer·실제 로더·학습·체크포인트 저장/복원 |
| `comparison.py`, `prepare_compare.py` | 1,200run 인수·메모리 매핑·비교 준비/승인 |
| `lifecycle.py`, `operator.py` | 게이트→학습→정상 보정→개발 평가→반납/정지 |
| `telemetry.py` | 검증된 할당 GPU의 사용률·VRAM 기록 |

고정 비교는 LSTM·Transformer 각각 hidden64·2층 /128·2층 /128·4층 /256·4층과 persistence/ridge_forecast, 총10종이다. seed0·batch256·FP32·최대60epoch·patience10·dropout0.1을 검사한다. 기존 revision/schema는 과거 데이터·증거를 검증하는 계약이므로 보존한다.

```bash
./ai/drone-ai train --profile default prepare --help
./ai/drone-ai watch --root /absolute/output/comparison-01
```

`prepare`는 본학습 승인이 없는 상태를 만든다. 별도 사용자 승인 뒤에만 `authorize --approve-one-comparison`과 `start`를 실행한다. 짧은 GPU 시험 통과는 본학습 승인이 아니다. 각 구성의 새 출력 폴더에 `model.safetensors`, `best.pt`, `last.pt`와 구조·설정을 저장하며 이전 회차에 자동 누적하지 않는다.

실패하면 증거를 남기고 중지한다. 성공하면 정상 보정·개발 평가·실제 할당 반납을 확인하고 정지한다. 최종 시험·추가 seed·자동 재학습·다음 회차는 별도 승인 대상이다. [명령과 인수 조건](../../docs/ai/PORTABLE_SETUP.md)을 따른다.
