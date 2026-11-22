from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .event_rules import ActiveClose, CycleRules, PrimitiveRules, default_primitive_rules


class VideoSource(BaseModel):
    type: Literal["video"]
    path: str = Field(min_length=1, max_length=1024)


class CameraSource(BaseModel):
    type: Literal["camera"]
    index: int = Field(ge=0)


class RunRequest(BaseModel):
    source: Annotated[VideoSource | CameraSource, Field(discriminator="type")]
    sop_version: str | None = Field(default=None, min_length=1, max_length=128)
    confidence: float = Field(default=0.25, gt=0, le=1)
    cycle_exit_x: float | None = Field(default=None, ge=0, lt=0.30)
    cycle_rules: CycleRules = Field(default_factory=CycleRules)
    primitive_rules: PrimitiveRules | None = Field(default_factory=default_primitive_rules)


RunState = Literal["starting", "running", "stopping", "stopped", "completed", "failed"]
TERMINAL_STATES = frozenset({"stopped", "completed", "failed"})


class RunStatus(BaseModel):
    id: str
    status: RunState
    source: VideoSource | CameraSource
    sop_version: str | None
    confidence: float
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    frames_processed: int
    last_frame_id: int | None
    timestamp_ms: int | None
    error: str | None
    cycle_exit_x: float
    active_cycle_id: str | None
    completed_cycles: int


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    track_id: int | None = None


class DetectionFrame(BaseModel):
    type: Literal["detection.frame"] = "detection.frame"
    run_id: str
    frame_id: int
    timestamp_ms: int
    width: int
    height: int
    detections: list[Detection]
    active_close: list[ActiveClose] = Field(default_factory=list)


class WorkpieceCycleEvent(BaseModel):
    type: Literal["sop.workpiece_cycle"] = "sop.workpiece_cycle"
    phase: Literal["started", "ended"]
    run_id: str
    cycle_id: str
    start_ms: int | None
    start_frame_id: int | None
    end_ms: int | None = None
    end_frame_id: int | None = None
    confirmed_ms: int
    confirmed_frame_id: int
    last_b_seen_ms: int | None = None
    last_b_seen_frame_id: int | None = None
    partial: bool = False
    a_track_id: int | None = None
    b_track_id: int | None = None
    empty_ms: int | None = None
    b_move_confirmed_ms: int | None = None
