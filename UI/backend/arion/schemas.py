from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    signal: str = Field(min_length=1, max_length=100)
    unit: str = Field(max_length=30)
    observed: float | None = None
    predicted: float | None = None
    residual: float | None = None
    note: str = Field(default="", max_length=500)


class DetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    event_id: str = Field(min_length=1, max_length=120)
    run_id: str
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    epoch: int = Field(ge=0)
    start_ms: float = Field(ge=0)
    end_ms: float = Field(ge=0)
    decided_at_ms: float = Field(ge=0)
    valid_until_ms: float = Field(ge=0)
    model_version: str = Field(min_length=1, max_length=100)
    status: Literal["normal", "suspect", "unavailable", "warming_up"]
    target: Literal["battery", "powertrain", "motor_1", "motor_2", "motor_3", "motor_4"]
    score: float | None = None
    threshold: float | None = None
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)
    evidence: list[Evidence] = Field(default_factory=list, max_length=30)
    origin: Literal["model", "fixture"] = "model"

    @model_validator(mode="after")
    def ordered_times(self):
        if not self.start_ms <= self.end_ms <= self.decided_at_ms <= self.valid_until_ms:
            raise ValueError("start ≤ end ≤ decided_at ≤ valid_until 순서가 필요합니다.")
        if self.status in ("normal", "suspect") and (self.score is None or self.threshold is None):
            raise ValueError("판정에는 점수와 임계값이 필요합니다.")
        return self


class AIResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["arion-ai-results-v1"]
    run_id: str
    input_sha256: str
    state: Literal["ready", "warming_up", "unavailable"]
    results: list[DetectionResult] = Field(default_factory=list, max_length=10000)


class AIConnection(BaseModel):
    url: str = Field(max_length=2048)


class DemoRequest(BaseModel):
    enabled: bool
