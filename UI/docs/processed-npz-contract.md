# 실시간 전처리·NPZ 계약 v1

외부 공용 전처리기가 MAVLink UDP 관측을 전처리해 로컬 대시보드에 전달합니다. 대시보드는 모델 학습·scaler 적용·재전처리를 수행하지 않습니다. 관측 물리값·mask·관측 나이·계약 버전을 복구용 JSONL에 먼저 fsync하고, 같은 수락 행을 화면·NPZ·모델 입력 참조에 사용합니다. 원본 수신률은 전처리 입력만으로 추측하지 않으며, 실제 전달 수신률·생산자 시간축의 출력률·표시 최대 10Hz를 별도로 기록합니다. 추론 주기와 API 확인 주기도 구분합니다.

## 등록과 전달

1. `POST /api/profiles`에 `InputProfile`을 등록합니다. `profile_id`는 불변입니다. 특징·단위·주기를 변경하려면 새 ID를 사용합니다. `feature_order`에 모든 신호를 중복 없이 지정합니다. `signals`에는 label/unit/ttl_ms/message를 지정하며, 동일 단위의 기존 화면 항목은 `display_key`로 연결할 수 있습니다. 단위 변환은 생산 측에서 수행합니다.
2. `POST /api/sessions`에 `kind=normalized`, `profile_id`, `name`을 전달합니다. UI의 **실시간 연결 → 공용 수집기 / 전처리 스트림**도 같습니다.
3. 반환된 `run_id`를 `session_id`에 넣고 `/api/sessions/{id}/frames`에 POST합니다. `GET /api/contracts/processed-frame`으로 기계 판독 JSON Schema를 확인할 수 있습니다.

```json
{
  "schema_version": "arion-processed-v1",
  "session_id": "live-발급된ID",
  "profile_id": "normalized_flight_v1",
  "sequence": 1,
  "epoch": 0,
  "source_time_ns": 1000000000000000123,
  "relative_time_ns": 0,
  "preprocess_version": "normalized-v1",
  "sample_period_ms": 100,
  "values": {
    "battery_current_a": {"value": 0, "unit": "A", "mask": true, "status": "valid", "age_ns": 0}
  },
  "armed": false,
  "mode": "관측 모드",
  "flight_state": "지상"
}
```

시각·순서는 int64이며 나노초 정밀도는 서버와 파일에서 유지합니다. 브라우저는 상대 ms만 사용합니다. `sequence`와 `epoch`는 생산자의 식별자입니다. 같은 세션·생산자 epoch·sequence의 동일 재전송은 `duplicate=true`로 응답하고 새 행을 만들지 않습니다. 내용이 달라지거나 알 수 없는 이전 sequence면 422와 제외 진단을 기록합니다.

`relative_time_ns`는 생산자 상대 시각입니다. 대시보드는 재연결·clock 역행·긴 공백을 표시용 epoch로 분리하고, 수락 행의 상대 시각과 epoch를 공통 시간축으로 확정합니다. 원래 epoch/시각은 저널의 `raw` 및 프레임의 `producer_epoch`에 보존합니다. 이후 `processed_frames`와 NPZ의 epoch/tick은 **수락한 공통 시간축**을 사용합니다. 수신 sequence 누락도 별도로 집계합니다.

값을 생략하면 missing입니다. mask=true에는 유한한 값·valid 상태·관측 나이가 필요합니다. TTL을 넘은 값은 stale/mask=false로 확정합니다. invalid/stale의 실제 값이 있으면 NPZ X에 보존하되 화면의 숫자는 숨깁니다. 값이 없는 셀은 NaN이며 0을 결측으로 사용하지 않습니다. 이전 값의 반복 전달은 생산 측이 관측 나이를 증가시켜야 합니다.

기존 `arion-normalized-v2`의 valid/age_ms 입력은 같은 공통 행으로 변환합니다. 새 프레임과 기존 어댑터 모두 추가 리샘플링·정규화는 하지 않습니다.

## NPZ와 파일 보관

60초의 공통 시간 구간을 저장하며 epoch 변경·최대 60,000행·압축 해제 예상 크기 48MiB·수집 종료에서도 마감합니다. 압축 작업자는 하나이고 저널 byte 범위를 읽어 압축한 뒤 임시 파일을 원자적으로 교체합니다. 압축이 실패해도 입력은 저널에 남으며 **저장 재시도** 또는 재시작 복구로 다시 생성합니다. 현재 구간은 다음 마감까지 pending입니다.

- `X[N,F]`: float64 물리값, 실제 결측 NaN.
- `mask/status/age_ns[N,F]`: 유효성, 상태 코드(0 missing / 1 valid / 2 stale / 3 invalid), int64 ns(-1 미관측).
- `tick_ns/source_time_ns/sequence/producer_sequence/epoch[N]`: 공통 상대 ns·원본 ns·수락 행 순서·생산자 순서·공통 epoch.
- `row_id/row_sha256`: 수락 행의 식별자와 canonical processed frame 해시.
- `rows`: JSON 문자열 배열. 원본 입력·수락된 표시 프레임·공통 프레임을 보존하며 구간별 재생과 참조 검증에 사용합니다.
- `metadata`: JSON 문자열. 물리 단위/특징 순서, 프로필 스냅샷, 전처리 버전, 출처, 구간 범위, 완료된 기록의 해시를 담습니다. object/pickle은 사용하지 않습니다.

파일은 `.data/catalog.json`, `profiles.json`, `providers.json`, `sessions/{id}.jsonl`, `sessions/{id}/npz/`, `records/{id}/manifest.json`, `records/{id}/analysis.jsonl` 등에 저장합니다. 원본 JSONL/NPZ는 `logs/`에 보존합니다. SQLite는 최초 이관 시 읽기 전용으로만 열고 `.data/backups/before-file-storage.sqlite3`와 `file-migration.json`을 남깁니다. 이후 런타임 저장은 파일만 사용합니다. 서버는 **단일 worker**로 실행합니다. 별도 CLI로 가져오기·검사할 때는 서버를 중지해 동시 카탈로그 수정을 피합니다.

활성 표시 버퍼는 최근 5분/최대 60,000행입니다. 완료된 전처리 세션은 NPZ 두 구간 캐시로 재생합니다. 차트는 최대 6,000개 포인트로 제한하고 시점 이동에 따라 최근 5분을 갱신합니다. 기존 원본 MAVLink JSONL 파서는 유지합니다.

## 모델과 내보내기 API

`GET /api/runs/{id}/input-segment?start_sequence=1&end_sequence=20`은 수락 행 순서를 사용합니다. `frames`는 기존 표시용 프레임, `processed_frames`는 물리 특징/mask/age를 가진 공통 입력, `rows`는 식별자·공통 시각·epoch입니다. `input_ref.segment_sha256`은 processed_frames를 키 정렬한 compact JSON으로 인코딩하고 LF로 연결한 SHA-256입니다. 원본 MAVLink 세션은 기존 표시 프레임 해시를 유지합니다. 서로 다른 epoch 구간은 참조할 수 없습니다.

LSTM/Transformer는 이 입력 참조와 모델별 점수·임계값을 반환합니다. 진행 중에는 file hash 대신 stream ref를 사용합니다. 다른 세션·구간·전처리 버전·해시 결과는 결합하지 않습니다. 종료된 세션에서도 같은 stream ref를 검증할 수 있으며 file ref는 최종 저널 해시를 사용합니다. 가져온 NPZ는 원본 NPZ 파일 해시의 독립 비행입니다.

| API | 동작 |
|---|---|
| `GET /api/runs/{id}/npz/chunks` | 저장 상태와 구간 목록 |
| `POST /api/runs/{id}/npz/retry` | 실패한 저장 작업 재시도 |
| `POST /api/runs/{id}/npz/export` | 종료된 비행 전체 NPZ 생성 작업 |
| `GET /api/exports/{job_id}` | queued/running/ready/error와 다운로드 주소 |
| `GET /api/exports/{job_id}/download` | 생성 완료 파일 다운로드 |
| `POST /api/runs/import` | JSONL 또는 물리 단위 NPZ 가져오기 |

전체 내보내기는 디스크 memmap과 ZIP 스트리밍을 사용합니다. 전체 비행 배열의 RAM 복사본을 만들지 않습니다. 서버 재시작 시 중단된 내보내기는 오류로 표시하며 다시 요청할 수 있습니다.

가져오기는 현재 압축 파일 64MiB / 압축 해제 128MiB 이내입니다. 학습용 NPZ 중 normalization이 identity 계열이고 특징·단위·시간축·mask·age가 확인되는 파일은 독립 관측 재생을 지원합니다. `age_messages`의 메시지 그룹 나이는 특징에 매핑합니다. `missing_fill=0`은 mask=false에서 실제 0으로 표시하지 않습니다. 모델 준비 여부를 정상 판정으로 사용하지 않습니다. `real-training.npz`에는 위치와 yaw 각도가 없으므로 3D 자세·궤적을 생성하지 않습니다. 기존 CLI 품질 검사는 유지합니다.
