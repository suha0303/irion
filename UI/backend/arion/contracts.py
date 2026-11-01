"""Versioned input and provider contracts. Normalized data is never resampled here."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, model_validator
from .telemetry import LABELS, MAX_AGE, RATE_TARGETS

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class SignalDefinition(Contract):
    label: str = Field(min_length=1, max_length=100)
    unit: str = Field(max_length=30)
    ttl_ms: FiniteFloat = Field(gt=0, le=60000)
    message: str = Field(default='NORMALIZED', max_length=100)

class InputProfile(Contract):
    schema_version: Literal['arion-profile-v2'] = 'arion-profile-v2'
    profile_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,80}$')
    input_format: Literal['mavlink', 'normalized']
    preprocess_version: str = Field(min_length=1, max_length=100)
    sample_period_ms: FiniteFloat | None = Field(default=None, gt=0, le=60000)
    gap_ms: FiniteFloat = Field(default=1000, gt=0, le=60000)
    signals: dict[str, SignalDefinition] = Field(min_length=1, max_length=256)
    expected_rates_hz: dict[str, FiniteFloat] = Field(default_factory=dict)
    voltage_encoding: Literal['mavlink', 'arion_4s'] = 'mavlink'
    missing_current_raw: list[FiniteFloat] = Field(default_factory=lambda:[-1], max_length=8)

    @model_validator(mode='after')
    def canonical_units(self):
        for key,spec in self.signals.items():
            if key in LABELS and spec.unit!=LABELS[key][1]:
                raise ValueError(f'{key} 표시 단위는 {LABELS[key][1]}여야 합니다. 생산 측에서 명시적으로 변환하세요.')
            if self.input_format=='mavlink' and (key not in LABELS or spec.message!=LABELS[key][2]):
                raise ValueError('MAVLink 프로필은 지원되는 디코더 신호에만 연결할 수 있습니다.')
        if self.input_format=='mavlink' and any(k not in MAX_AGE for k in self.expected_rates_hz):raise ValueError('지원되지 않는 MAVLink 스트림 이름')
        if any(v<=0 for v in self.expected_rates_hz.values()):raise ValueError('기대 수신률은 양수여야 합니다.')
        return self

class SourceConfig(Contract):
    name: str = Field(default='실시간 비행', min_length=1, max_length=120)
    kind: Literal['udp', 'serial', 'normalized'] = 'udp'
    profile_id: str = 'mavlink_common_v1'
    host: Literal['127.0.0.1', 'localhost'] = '127.0.0.1'
    port: int = Field(default=14560, ge=1024, le=65535)
    device: str = Field(default='', max_length=256)
    baud: int = Field(default=115200, ge=1200, le=3000000)
    source_system: int | None = Field(default=None, ge=1, le=255)
    source_component: int | None = Field(default=None, ge=0, le=255)

class IncomingSignal(Contract):
    value: FiniteFloat | None
    unit: str = Field(max_length=30)
    valid: bool
    age_ms: FiniteFloat = Field(ge=0)

class NormalizedInput(Contract):
    schema_version: Literal['arion-normalized-v2']
    session_id: str
    sequence: StrictInt = Field(ge=0)
    epoch: StrictInt = Field(ge=0)
    source_time_ns: StrictInt = Field(ge=0, le=2**63-1)
    preprocess_version: str
    sample_period_ms: FiniteFloat = Field(gt=0)
    values: dict[str, IncomingSignal] = Field(max_length=256)
    armed: bool | None = None
    mode: str = Field(default='미확인', max_length=80)
    flight_state: str = Field(default='미확인', max_length=80)

class InputRequirement(Contract):
    signals: dict[str, str] = Field(default_factory=dict)
    sample_period_ms: FiniteFloat | None = Field(default=None, gt=0)
    preprocess_version: str | None = None

class Provider(Contract):
    provider_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,80}$')
    name: str = Field(min_length=1, max_length=80)
    kind: Literal['detector', 'explanation'] = 'detector'
    model_version: str = Field(default='unconfigured', max_length=100)
    requirements: InputRequirement = Field(default_factory=InputRequirement)
    poll_interval_ms: int = Field(default=500, ge=100, le=60000)

class InputReference(Contract):
    kind: Literal['file', 'stream']
    sha256: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    session_id: str | None = None
    start_sequence: int | None = Field(default=None, ge=1)
    end_sequence: int | None = Field(default=None, ge=1)
    segment_sha256: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    preprocess_version: str | None = None

class EvidenceV2(Contract):
    signal: str = Field(min_length=1, max_length=100)
    unit: str = Field(max_length=30)
    observed: FiniteFloat | None = None
    predicted: FiniteFloat | None = None
    residual: FiniteFloat | None = None
    note: str = Field(default='', max_length=500)

class AnalysisResult(Contract):
    schema_version: Literal['arion-analysis-v2'] = 'arion-analysis-v2'
    event_id: str = Field(min_length=1, max_length=120)
    provider_id: str
    run_id: str
    input_ref: InputReference
    epoch: int = Field(ge=0)
    start_ms: FiniteFloat = Field(ge=0)
    end_ms: FiniteFloat = Field(ge=0)
    decided_at_ms: FiniteFloat = Field(ge=0)
    valid_until_ms: FiniteFloat = Field(ge=0)
    model_version: str = Field(min_length=1, max_length=100)
    kind: Literal['detector', 'explanation'] = 'detector'
    status: Literal['normal', 'suspect', 'unavailable', 'warming_up', 'explanation']
    target: str = Field(default='powertrain', max_length=80)
    score: FiniteFloat | None = None
    threshold: FiniteFloat | None = None
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default='', max_length=6000)
    evidence: list[EvidenceV2] = Field(default_factory=list, max_length=64)
    related_event_ids: list[str] = Field(default_factory=list, max_length=64)
    origin: Literal['model', 'fixture'] = 'model'

    @model_validator(mode='after')
    def validate_result(self):
        if not self.start_ms <= self.end_ms <= self.decided_at_ms <= self.valid_until_ms:
            raise ValueError('근거·판정·유효기간 순서 오류')
        if self.kind == 'detector' and self.status in ('normal', 'suspect') and (self.score is None or self.threshold is None):
            raise ValueError('수치 판정에는 점수와 임계값이 필요합니다.')
        if self.kind == 'explanation' and (self.status != 'explanation' or self.score is not None or self.threshold is not None):
            raise ValueError('설명 결과는 수치 판정으로 사용할 수 없습니다.')
        if self.kind == 'explanation' and not (self.evidence or self.related_event_ids):
            raise ValueError('설명에는 근거 신호 또는 관련 이벤트 참조가 필요합니다.')
        return self

class AnalysisResponse(Contract):
    schema_version: Literal['arion-provider-results-v2']
    provider_id: str
    run_id: str
    state: Literal['ready', 'warming_up', 'unavailable']
    results: list[AnalysisResult] = Field(default_factory=list, max_length=2000)


def builtin_profiles():
    definitions = {k: SignalDefinition(label=v[0], unit=v[1], message=v[2], ttl_ms=MAX_AGE[v[2]]) for k,v in LABELS.items()}
    return [InputProfile(profile_id=p, input_format='mavlink', preprocess_version='raw-v1', signals=definitions,
                         expected_rates_hz=RATE_TARGETS, voltage_encoding='arion_4s' if p=='arion_cmu_v1' else 'mavlink',
                         missing_current_raw=[-1,-100] if p in ('arion_cmu_v1','px4_sih_v1') else [-1]) for p in ('arion_cmu_v1', 'mavlink_common_v1','px4_sih_v1')] + [
        InputProfile(profile_id='normalized_flight_v1', input_format='normalized', preprocess_version='normalized-v1',
                     sample_period_ms=100, signals=definitions, expected_rates_hz={'NORMALIZED':10})]
