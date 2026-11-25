"""JSON configuration and wire models for the four primitive events."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SizeRange(RuleModel):
    min: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bounds(self) -> "SizeRange":
        if self.min is None and self.max is None:
            raise ValueError("Set at least one size bound")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Minimum size must not exceed maximum size")
        return self

    def contains(self, value: float) -> bool:
        return value > 0 and (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


class BoxSize(SizeRange):
    metric: Literal["height_ratio", "width_ratio", "area_ratio"] = "height_ratio"
    min: float | None = Field(default=None, gt=0, le=1)
    max: float | None = Field(default=None, gt=0, le=1)


class Target(RuleModel):
    class_id: Literal[0, 1, 2, 3]
    bbox_size: BoxSize | None = None


class Direction(RuleModel):
    angle_deg: float = Field(ge=0, lt=360)
    tolerance_deg: float = Field(default=30, ge=0, le=180)


class EventRule(RuleModel):
    id: str = Field(min_length=1, max_length=128)
    min_confidence: float = Field(default=0.25, gt=0, le=1)
    max_gap_ms: int = Field(default=100, ge=0, le=5000)


class SingleTargetRule(EventRule):
    target: str = Field(min_length=1)


class MoveRule(SingleTargetRule):
    type: Literal["Move"]
    window_ms: int = Field(default=300, gt=0, le=5000)
    min_displacement_ratio: float = Field(default=0.035, gt=0, le=1)
    direction: Direction | None = None


Edge = Literal["left", "right", "top", "bottom"]


class RelativePosition(RuleModel):
    reference: str = Field(min_length=1)
    anchor: Literal["center", "left", "right", "top", "bottom"] = "center"
    axis: Literal["x", "y"]
    min: float | None = Field(default=None, allow_inf_nan=False)
    max: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bounds(self) -> "RelativePosition":
        if self.min is None and self.max is None:
            raise ValueError("Set at least one relative position bound")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Minimum relative position must not exceed maximum")
        return self

    def contains(self, value: float) -> bool:
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


class CloseRule(EventRule):
    type: Literal["Close"]
    targets: tuple[str, str]
    distance_metric: Literal["bbox_gap", "center_distance", "bottom_edge_gap"] = "bbox_gap"
    edge: Edge | None = None
    target_edge: Edge | None = None
    enter_distance_ratio: float = Field(default=0.015, ge=0, le=1)
    exit_distance_ratio: float = Field(default=0.025, gt=0, le=1)
    hold_ms: int = Field(default=300, ge=0, le=30000)
    bbox_height_ratio: SizeRange | None = None
    bbox_sizes: dict[str, BoxSize] = Field(default_factory=dict)
    min_overlap_ratio: float | None = Field(default=None, gt=0, le=1)
    union_with: str | None = Field(default=None, min_length=1)
    relative_position: RelativePosition | None = None

    @property
    def subject_aliases(self) -> tuple[str, ...]:
        return (*self.targets, self.union_with) if self.union_with else self.targets

    @model_validator(mode="after")
    def validate_pair(self) -> "CloseRule":
        if self.targets[0] == self.targets[1]:
            raise ValueError("Close requires two distinct target aliases")
        if self.edge is not None and self.distance_metric != "bbox_gap":
            raise ValueError("Close edge requires distance_metric='bbox_gap'")
        if self.target_edge is not None and self.distance_metric == "center_distance":
            raise ValueError("Close target_edge requires a gap distance_metric")
        if self.union_with in self.targets:
            raise ValueError("Close union_with must be distinct from the two distance targets")
        if self.relative_position:
            reference = self.relative_position.reference
            if reference != self.targets[0] and (reference != "union" or self.union_with is None):
                raise ValueError("Close relative_position.reference must be the first target or 'union' with union_with")
        if self.exit_distance_ratio <= self.enter_distance_ratio:
            raise ValueError("Close exit_distance_ratio must exceed enter_distance_ratio")
        if self.bbox_sizes.keys() - set(self.subject_aliases):
            raise ValueError("Close bbox_sizes must reference the rule's target aliases")
        return self


class InRule(SingleTargetRule):
    type: Literal["In"]
    confirm_ms: int = Field(default=200, ge=0, le=30000)


class OutRule(SingleTargetRule):
    type: Literal["Out"]
    exit_window_ms: int = Field(default=300, gt=0, le=5000)
    missing_ms: int = Field(default=500, gt=0, le=5000)
    edge_distance_ratio: float = Field(default=0.03, ge=0, le=0.25)
    min_displacement_ratio: float = Field(default=0.03, gt=0, le=1)
    edge: Edge | None = None
    left_edge_x: float = Field(default=0, ge=0, lt=0.30)


PrimitiveRule = Annotated[MoveRule | CloseRule | InRule | OutRule, Field(discriminator="type")]


class EventReference(RuleModel):
    event: str
    bind: dict[str, str] = Field(default_factory=dict)


def resolve_subjects(
    terms: list[EventReference], same_tracks: list[str], available: dict[str, dict[str, int]],
) -> dict[str, int]:
    result = {}
    repeated = set()
    for term in terms:
        if term.event not in available:
            raise ValueError(f"Unknown event or forward reference: {term.event!r}")
        source = available[term.event]
        renamed = {term.bind.get(alias, alias): category for alias, category in source.items()}
        if term.bind.keys() - source.keys() or len(renamed) != len(source) or not all(renamed):
            raise ValueError(f"Invalid aliases for {term.event!r}")
        for alias, category in renamed.items():
            if alias in result:
                repeated.add(alias)
                if result[alias] != category or alias not in same_tracks:
                    raise ValueError(f"Shared alias {alias!r} requires same_tracks and the same YOLO class")
            result[alias] = category
    if len(set(same_tracks)) != len(same_tracks) or set(same_tracks) - repeated:
        raise ValueError("same_tracks must list distinct aliases shared by multiple branches or steps")
    return result


class ParallelRule(RuleModel):
    id: str = Field(min_length=1, max_length=128)
    op: Literal["parallel"] = "parallel"
    all: list[EventReference] = Field(min_length=1, max_length=8)
    when: list[EventReference] = Field(default_factory=list, max_length=7)
    same_tracks: list[str] = Field(default_factory=list)
    min_overlap_ms: int = Field(default=120, ge=0, le=30000)
    within_cycle: bool = True

    @property
    def terms(self) -> list[EventReference]:
        return [*self.all, *self.when]

    @model_validator(mode="after")
    def validate_terms(self) -> "ParallelRule":
        if not 2 <= len(self.terms) <= 8:
            raise ValueError("Parallel all and when must have 2 to 8 branches in total")
        return self


class CycleRules(RuleModel):
    empty_before_ms: int = Field(default=1200, gt=0, le=60000)
    empty_after_ms: int = Field(default=1200, gt=0, le=60000)
    a_wait_ms: int = Field(default=3000, gt=0, le=60000)
    move_to_out_ms: int = Field(default=2500, gt=0, le=60000)
    min_confidence: float = Field(default=0.25, gt=0, le=1)


class PrimitiveRules(RuleModel):
    schema_version: Literal[1] = 1
    targets: dict[str, Target]
    events: list[PrimitiveRule] = Field(max_length=256)
    parallel: list[ParallelRule] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def validate_references(self) -> "PrimitiveRules":
        seen = set()
        close: dict[str, CloseRule] = {}
        for rule in self.events:
            if rule.id in seen:
                raise ValueError(f"Duplicate event id: {rule.id}")
            seen.add(rule.id)
            targets = rule.subject_aliases if isinstance(rule, CloseRule) else (rule.target,)
            for target in targets:
                if target not in self.targets:
                    raise ValueError(f"Unknown target {target!r} in {rule.id!r}")
            if isinstance(rule, CloseRule):
                close[rule.id] = rule
        for rule in self.parallel:
            if rule.id in seen:
                raise ValueError(f"Duplicate event id: {rule.id}")
            seen.add(rule.id)
            self.validate_parallel(rule, close)
        return self

    def validate_parallel(self, rule: ParallelRule, close: dict[str, CloseRule]) -> dict[str, int]:
        terms = rule.terms
        if len({term.event for term in terms}) != len(terms):
            raise ValueError(f"Duplicate branch in {rule.id!r}")
        available = {}
        for term in terms:
            if term.event not in close:
                raise ValueError(f"Parallel branch {term.event!r} must name a Close rule")
            available[term.event] = {alias: self.targets[alias].class_id for alias in close[term.event].subject_aliases}
        return resolve_subjects(terms, rule.same_tracks, available)


def default_primitive_rules() -> PrimitiveRules:
    path = Path(__file__).with_name("primitive_rules.json")
    return PrimitiveRules.model_validate_json(path.read_bytes())


class TrackedTarget(BaseModel):
    track_id: int
    class_id: int


class PrimitiveEvent(BaseModel):
    type: Literal["sop.primitive_event"] = "sop.primitive_event"
    event: Literal["Move", "Close", "In", "Out"]
    phase: Literal["detected", "started", "ended"]
    run_id: str
    sequence: int
    rule_id: str
    occurrence_id: str
    subjects: dict[str, TrackedTarget]
    start_ms: int
    start_frame_id: int
    end_ms: int | None = None
    end_frame_id: int | None = None
    confirmed_ms: int
    confirmed_frame_id: int
    reason: Literal["separated", "unobserved", "stream_gap", "run_finished"] | None = None
    evidence: dict[str, int | float | str] = Field(default_factory=dict)


class ActiveClose(BaseModel):
    rule_id: str
    occurrence_id: str
    subjects: dict[str, TrackedTarget]
    evidence_start_ms: int


class CompositeEvent(BaseModel):
    type: Literal["sop.composite_event"] = "sop.composite_event"
    op: Literal["event", "parallel", "sequence"]
    run_id: str
    rule_id: str
    occurrence_id: str
    cycle_id: str | None
    subjects: dict[str, TrackedTarget]
    start_ms: int
    confirmed_ms: int
    confirmed_frame_id: int
    evidence_ids: list[str]


class ParallelEvent(CompositeEvent):
    op: Literal["parallel"] = "parallel"
