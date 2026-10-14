"""
Canonical, versioned event and prediction contracts (schema_version "1.0").

Design rules
* Only attributes that actually exist in each dataset are carried; dataset-specific
  payloads are explicit (``EllipticAttributes`` vs ``ZenodoAttributes``) and selected by
  ``source_dataset`` (a discriminated union), so an Elliptic payload can never be routed to
  the Zenodo detector or vice versa.
* Neither dataset has wall-clock transaction timestamps. ``event_time`` is therefore an
  ordinal (``elliptic_time_step``: 1..49, ~2-week buckets; ``block_height``: Bitcoin block
  number). Wall-clock fields (``produced_at``, ``processing_time``, ``scored_at``) are
  system times of this pipeline, not blockchain times.
* Elliptic graph context: each event lists the transaction ids it pays to
  (``out_neighbors``) and the size of its time-step snapshot. A graph-dependent model is only
  invoked on a complete snapshot; otherwise the result status is
  ``rejected_incomplete_graph_context``. Zenodo Parquet data has no address/edge data, so
  ``graph`` is null for Zenodo events.
"""
from datetime import datetime, timezone
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventTime(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["elliptic_time_step", "block_height"]
    value: int = Field(ge=0)


class EllipticGraphRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str = Field(description="elliptic:step:<t>")
    snapshot_size: int = Field(gt=0, description="number of transactions in this time-step snapshot")
    out_neighbors: List[str] = Field(default_factory=list, description="tx ids this transaction pays to")


class DataQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "invalid"] = "ok"
    issues: List[str] = Field(default_factory=list)


class EllipticAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: List[float] = Field(min_length=165, max_length=165,
                                  description="raw (unscaled) Elliptic features f1..f165, anonymised by Elliptic")


class ZenodoAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_number: int = Field(ge=0)
    tx_index: int = Field(ge=0)
    num_inputs: int = Field(ge=0)
    num_outputs: int = Field(ge=0)
    total_input_satoshi: int
    total_output_satoshi: int
    fee_satoshi: int
    fee_rate: float = Field(description="fee / (num_inputs + num_outputs); NOT satoshi per byte")
    is_coinbase: int = Field(ge=0, le=1)
    avg_input_val: float
    avg_output_val: float
    input_output_count_ratio: float
    input_output_val_ratio: float
    max_input_prior_tx_count: int
    avg_input_prior_tx_count: float
    max_input_prior_volume_satoshi: int
    new_address_fraction: float


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    event_id: str = Field(description="stable id: '<source_dataset>:<transaction_id>'")
    transaction_id: str
    event_time: EventTime
    produced_at: str = Field(description="producer wall-clock time (UTC ISO-8601)")
    replay_sequence: Optional[int] = Field(default=None, ge=0, description="deterministic replay order")
    data_quality: DataQuality = Field(default_factory=DataQuality)

    @model_validator(mode="after")
    def _stable_id(self):
        expected = f"{self.source_dataset}:{self.transaction_id}"
        if self.event_id != expected:
            raise ValueError(f"event_id must be '{expected}'")
        return self


class EllipticEvent(_EventBase):
    source_dataset: Literal["elliptic"]
    attributes: EllipticAttributes
    graph: EllipticGraphRef

    @field_validator("event_time")
    @classmethod
    def _kind(cls, v):
        if v.kind != "elliptic_time_step" or not 1 <= v.value <= 49:
            raise ValueError("elliptic events need event_time.kind='elliptic_time_step' with value 1..49")
        return v


class ZenodoEvent(_EventBase):
    source_dataset: Literal["zenodo"]
    tx_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    attributes: ZenodoAttributes
    graph: None = None

    @field_validator("event_time")
    @classmethod
    def _kind(cls, v):
        if v.kind != "block_height":
            raise ValueError("zenodo events need event_time.kind='block_height'")
        return v


TransactionEvent = Union[EllipticEvent, ZenodoEvent]


class EventEnvelope(BaseModel):
    """Wrapper used to parse any event with discriminated routing on source_dataset."""
    event: TransactionEvent = Field(discriminator="source_dataset")


ResultStatus = Literal["scored", "skipped_coinbase", "skipped_invalid_input",
                       "rejected_incomplete_graph_context", "rejected_invalid_event", "error"]


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    event_id: Optional[str] = None
    transaction_id: Optional[str] = None
    source_dataset: Optional[Literal["elliptic", "zenodo"]] = None
    event_time: Optional[EventTime] = None
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    score: Optional[float] = None
    score_type: Optional[Literal["illicit_probability", "isolation_forest_anomaly_score"]] = None
    score_percentile: Optional[float] = Field(default=None, description="Zenodo only: rank within fit window")
    threshold: Optional[float] = None
    flagged: Optional[bool] = None
    reason_codes: List[str] = Field(default_factory=list)
    graph_context: Optional[Literal["complete", "partial", "not_required", "not_available"]] = None
    status: ResultStatus
    error: Optional[str] = None
    produced_at: Optional[str] = None
    processing_time: Optional[str] = Field(default=None, description="when the consumer started processing")
    scored_at: str = Field(default_factory=utc_now_iso)


def event_json_schema() -> dict:
    return EventEnvelope.model_json_schema()


def result_json_schema() -> dict:
    return PredictionResult.model_json_schema()
