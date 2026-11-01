> 이 문서는 v0.1 기록입니다. 현재 기능과 계약은 [v2 계약](v2-contracts.md), 검증은 [v2 검증 기록](validation-v2.md)을 참조하세요.

# AI팀 → 대시보드 결과 계약 v1

대시보드는 AI 모델과 전처리 파이프라인을 소유하지 않습니다. 동일 원본 비행을 가진 AI 서비스가 결과를 반환합니다. 데이터 준비 여부를 이상 여부로 대체하지 않습니다.

## HTTP 전송

연결 화면에서 지정한 로컬 HTTP 주소에 500ms 간격으로 GET 요청합니다. query는 `run_id`, `input_sha256`입니다. 예: `/results?run_id=run-...&input_sha256=...`. 요청에는 로그 내용·NPZ·학습 정보가 포함되지 않습니다. AI팀은 manifest와 원본 파일을 별도 인계받습니다.

응답 형식은 `arion-ai-results-v1`입니다. 정상 HTTP 응답은 다음 envelope를 반환합니다.

```json
{
  "schema_version": "arion-ai-results-v1",
  "run_id": "run-<입력 SHA의 앞 16자리>",
  "input_sha256": "<원본 파일 전체 SHA-256>",
  "state": "ready",
  "results": []
}
```

`state`는 `ready`, `warming_up`, `unavailable` 중 하나입니다. 빈 결과는 정상 판정을 뜻하지 않습니다. 서버 오류, 연결 중단, 스키마 오류는 별도 `error` 상태로 표시합니다. 오류 응답으로 이전 판정을 정상처럼 유지하지 않습니다.

## DetectionResult

정확한 타입·필수 값·enum은 [JSON Schema](ai-response.schema.json)를 사용합니다.

| 필드 | 계약 |
|---|---|
| event_id | 모델 버전 내 안정적인 결과 ID. 반복 폴링에 같은 ID 사용 |
| run_id / input_sha256 | 선택한 원본과 정확히 일치해야 함 |
| epoch | 대시보드 manifest의 epoch 번호 |
| start_ms / end_ms | 판단에 사용한 근거 구간, 같은 epoch 내 상대 시각 |
| decided_at_ms | 결과가 실제로 이용 가능해진 비행 시간축상의 시각. 단순한 구간 시작 시각 아님 |
| valid_until_ms | 이 판정이 현재 상태로 유효한 마지막 시각. epoch/비행 경계 내 |
| model_version | 모델·scaler·경보 정책을 재현할 수 있는 버전 식별자 |
| status | normal / suspect / warming_up / unavailable |
| target | battery / powertrain / motor_1 / motor_2 / motor_3 / motor_4 |
| score / threshold | normal·suspect이면 둘 다 필수. 유한 숫자. 건강도%·확률로 취급하지 않음 |
| title / description | 사람이 읽는 판단·근거 설명 |
| evidence | 선택적 신호별 observed / predicted / residual / unit / note. 제공된 값만 표시 |
| origin | 실제 모델은 model, 인공 UI 응답은 fixture |

시간은 `0 ≤ start_ms ≤ end_ms ≤ decided_at_ms ≤ valid_until_ms`이며 선택한 비행·epoch 내에 있어야 합니다. 단위 변환과 신호 정렬은 AI팀이 책임집니다. 19특징 NPZ 구조에 새 모델을 억지로 맞추지 않으며, 실제 입력 특징·scaler·window 변경은 AI 모델 버전에 반영합니다.

MAVLink의 `receive_monotonic_ns`를 기준으로 첫 유효 입력부터 상대 t_ms를 사용합니다. 실제 원본 timestamp는 UI의 반올림 문자열이 아닌 manifest/원본에 맞춥니다. epoch 경계를 넘어 근거 창을 만들지 않습니다. 정상 운용의 SOC 감소, 높은 전류, 모터 간 차이 자체를 고장 라벨로 사용하지 않습니다.

## 처리·저장

- envelope와 모든 이벤트의 비행·해시·시각·epoch를 검사하고 하나라도 실패하면 전체 응답을 저장하지 않습니다.
- `run_id + event_id + model_version + origin`을 키로 upsert합니다. envelope의 results는 누적 또는 증분 모두 허용하며 생략된 과거 이벤트는 삭제되지 않습니다.
- 현재 재생 시각에 도달한 이벤트만 화면에 공개합니다. 이전 epoch의 이벤트는 이력에 남지만 현재 판정으로 사용하지 않습니다.
- 유효기간이 지난 결과는 이력에 남고 현재 상태는 판정 대기입니다. API가 warming_up/unavailable/error이면 이전 점수로 정상 상태를 유지하지 않습니다.
- origin별 저장 공간을 분리하고 fixture 모드에는 명시적인 배너를 표시합니다. 시연 완료 후 시험 모드를 끕니다.

## 측정

`GET /api/metrics`는 두 지표를 분리합니다. `render_ack`는 WS 전송→React commit 후 requestAnimationFrame→서버 ACK의 왕복 상한값입니다. `result_receipt_to_render_ack`는 이미 재생 시각에 도달한 새로운 HTTP 결과의 서버 수신→화면 ACK입니다. 미래 결과를 의도적으로 대기시킨 시간, 모델 추론 시간, 500ms HTTP 폴링 대기는 제외합니다. API 결과 표본이 0이면 미측정으로 표시합니다.

실제 AI 성능은 이 지표로 평가하지 않습니다. 오탐·미탐·검출 지연·대상 식별은 독립 평가 데이터로 AI팀이 별도 검증합니다.
