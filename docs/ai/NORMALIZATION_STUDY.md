# LayerNorm 및 최적화 방법 비교

새 비교는 LSTM·Transformer 각 네 크기에 바닐라 SGD·Adam을 적용한16개다.
크기는64·2층,128·2층,128·4층,256·4층이며 층수는 인코더·디코더 각각의 수다.
Transformer는 Post-LN만, LSTM은 층 사이 LayerNorm만 사용한다.
현재 계획과 승인 상태는 `ai/configs/normalization-study.json`에 기록한다.

## 모델과 초기화

새 모델은 `ai/learning/models.py`에 정의하며 Dropout 층·옵션·제거 함수를 만들지 않는다.
과거 모델을 읽는 정확한 구현은 `ai/learning/legacy_models.py`에 보존한다.
새 모델은 `StudySpec`, 과거 모델은 기존 `ModelSpec`과 명시적 revision으로 구분한다.
기존 가중치나 결과를 새 모델로 덮어쓰지 않는다.

LSTM은 native LSTM 층 사이에 LayerNorm을 적용한다. 마지막 층 뒤에는 추가하지 않는다.
입력 행렬은 게이트별 Xavier uniform, 순환 행렬은 게이트별 orthogonal로 초기화한다.
입력측 forget bias만1, 나머지 두 bias 벡터는0으로 두어 합산된 forget offset이1이 된다.
미래 시점 embedding은 평균0·표준편차1의 정규분포다.

Transformer의 모든 행렬과 query embedding은 Xavier uniform gain1, bias는0이다.
LayerNorm은 gamma1·beta0·epsilon1e-5다. Linear 출력층도 Xavier uniform·bias0이다.
모델은 seed0으로 새로 생성하고 초기 가중치 해시를 보존한다.
같은 크기·종류의 SGD/Adam과 학습률 후보는 같은 초기 가중치에서 출발한다.

## 최적화와 학습률 선택

바닐라 SGD는 momentum0·dampening0·Nesterov 비활성이다.
Adam은 beta=(0.9,0.999)·epsilon1e-8이다. 둘 모두 weight decay0이다.
전체 최대60epoch에 해당하는 optimizer step 수를 먼저 고정한다.
앞5% step에서 최대 학습률의10%부터 최대값까지 선형 warmup하고,
남은 step에서 cosine으로 최대값의1%까지 감소한다. 재시작은 없다.
아주 짧은 기능시험에서는 warmup 최소2step을 사용한다.
warmup을 끝낸 다음 epoch부터 patience10을 계산한다.
최저 정상 validation 손실의 가중치를 최종 모델로 내보낸다.
checkpoint에는 optimizer·step 위치·조기 종료 상태·RNG·초기화 해시를 저장한다.

후보는 SGD=(0.001,0.01,0.03), Adam=(0.0001,0.0003,0.001)이다.
각16개 구성에서 후보3개를3epoch씩, 합계48회로 한정한다.
정상 train 최대8192창·validation 최대4096창을 각 역할 전체에서 등간격으로 선택한다.
역할·원본 인덱스·정규화된 입력 해시를 기록하고 사건 정답·개발평가 자료는 읽지 않는다.
동일한 학습량 뒤 마지막 validation 손실이 최소인 후보를 선택한다.
동률이면 선언된 후보 순서를 사용하며, 선택 기록은 데이터·scaler·소스에 결속한다.
이는 짧은 대리 시험이므로 전체 학습에서 최적의 학습률이라는 보장은 없다.
오류·비유한 값은 후보를 건너뛰거나 자동 대체하지 않고 중지한다.

사전시험 가중치는 본학습에 재사용하지 않는다. 선택 후에도 본학습은 seed0부터 시작한다.
실제 GPU 메모리·복원 및 처리율 검사는 기존 lifecycle이 수행한다.
처리율은 선택된 optimizer와 실제 loader의500개 학습 step, 별도의50step 워밍업,
전체 validation 측정을 사용한다. LR 후보의 짧은 시험 속도를 전체 처리율로 쓰지 않는다.

## 준비와 실행 경계

기존 `drone-ai configure`에서 해당 서버의 Python·데이터·새 출력 경로를 지정한다.
`train prepare`의 `--normalization-study --training-only`가 새 구성을 선택한다.
예산은 `--wall-hours`로 명시해야 하며, 현재48시간은 제출하지 않은 제안 상한이다.
`--reuse-from`은 기존에 실패·반납된 작업의 검증된 upstream을 읽기 전용 재사용한다.
모델이나 과거 평가 결과를 재사용하는 옵션이 아니다.

준비 명령은 GPU를 제출하지 않는다. 기존 명시적 `train authorize`와 `train start` 뒤에만
서버 tmux·단일 Slurm 작업이 정상자료 사전시험→검사→본학습·저장을 유한하게 수행한다.
자원은 p02/DIS01·단일 노드·GPU1개·CPU4·RAM64GiB로 고정한다.
진행 화면은 기존 `drone-ai watch --root ...`를 사용한다.
학습 종료와 GPU 반납 후 중지하며, 보정·개발 평가·최종 시험·다음 회차는 실행하지 않는다.
LLM 감시·자동 재시도·자동 후속 호출을 만들지 않는다.

## 실측에 따른 실행 비용 절감

정규화는 기존 float64 계산→float32 저장 순서와512창 블록을 유지한다.
메모리·시간 검사는 모든 블록에서 수행하며, 진행 JSON은2초 간격과 각 배열의
시작·flush 완료 시점에 저장한다. 정상화 결과의 수치나 입력 범위는 바꾸지 않는다.

학습은 loss의 비유한 값 검사와 clipping 전 전체 L2 norm의 비유한 값 검사를
매 step 수행한다. 이 norm에 모든 gradient가 포함되므로 파라미터마다 중복하던
`isfinite/all` 연산은 제거했다. NaN·Inf·norm overflow와 gradient 부재는
optimizer 실행 전에 중지한다. clipping 한도1.0과foreach 경로는 유지한다.
진단 profiler도 같은 검사 함수를 사용한다.

결정적 알고리즘과 미초기화 메모리 채움 설정, FP32·batch·모델 구조·학습률 정책은
변경하지 않았다. 이번 CPU 동등성 시험을 GPU 속도 개선율이나 탐지 성능 검증으로
해석하지 않는다. 기존 서버의 봉인 소스·checkpoint·실측 증거는 그대로 보존한다.
