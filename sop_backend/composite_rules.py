"""Parse and resolve the versioned configuration for composed SOP events."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .event_rules import (
    CloseRule, CycleRules, Direction, EventReference, InRule, MoveRule, OutRule, ParallelRule,
    PrimitiveRules, RuleModel, Target, resolve_subjects,
)


class ClearCondition(RuleModel):
    class_ids: set[Literal[0, 1, 2, 3]] = Field(min_length=1)
    min_confidence: float = Field(default=0.25, gt=0, le=1)


class CycleStart(RuleModel):
    after_clear_ms: int = Field(gt=0, le=60000)
    event: str
    within_ms: int = Field(gt=0, le=60000)


class CycleEnd(RuleModel):
    move: str
    out: str
    within_ms: int = Field(gt=0, le=60000)
    clear_ms: int = Field(gt=0, le=60000)


class CycleDefinition(RuleModel):
    clear: ClearCondition
    start: CycleStart
    end: CycleEnd


class SequenceRule(RuleModel):
    id: str = Field(min_length=1, max_length=128)
    op: Literal["sequence"]
    steps: list[EventReference] = Field(min_length=2)
    within_ms: int = Field(gt=0, le=60000)
    same_tracks: list[str] = Field(default_factory=list)
    within_cycle: bool = True


class EventAliasRule(EventReference):
    id: str = Field(min_length=1, max_length=128)
    op: Literal["event"]
    within_cycle: bool = True


CompositeRule = Annotated[EventAliasRule | ParallelRule | SequenceRule, Field(discriminator="op")]


class CompositeRules(RuleModel):
    schema_version: Literal[1] = 1
    cycle: CycleDefinition
    events: list[CompositeRule] = Field(default_factory=list, max_length=64)


def default_composite_rules() -> CompositeRules:
    return CompositeRules.model_validate_json(Path(__file__).with_name("composite_rules.json").read_bytes())


@dataclass(frozen=True)
class ResolvedRules:
    primitives: PrimitiveRules
    cycle: CycleDefinition
    enter: InRule
    move: MoveRule
    exit: OutRule
    events: list[CompositeRule]


def _apply_cycle_overrides(cycle: CycleDefinition, overrides: CycleRules | None) -> CycleDefinition:
    if overrides is None:
        return cycle
    fields = overrides.model_fields_set
    return CycleDefinition(
        clear=cycle.clear.model_copy(update={
            "min_confidence": overrides.min_confidence,
        }) if "min_confidence" in fields else cycle.clear,
        start=CycleStart(
            event=cycle.start.event,
            after_clear_ms=overrides.empty_before_ms if "empty_before_ms" in fields else cycle.start.after_clear_ms,
            within_ms=overrides.a_wait_ms if "a_wait_ms" in fields else cycle.start.within_ms,
        ),
        end=CycleEnd(
            move=cycle.end.move, out=cycle.end.out,
            within_ms=overrides.move_to_out_ms if "move_to_out_ms" in fields else cycle.end.within_ms,
            clear_ms=overrides.empty_after_ms if "empty_after_ms" in fields else cycle.end.clear_ms,
        ),
    )


def resolve_rules(
    config: CompositeRules, primitives: PrimitiveRules | None, exit_x: float, overrides: CycleRules | None = None,
) -> ResolvedRules:
    cycle = _apply_cycle_overrides(config.cycle, overrides)
    base = primitives or PrimitiveRules(targets={}, events=[])
    used = set(base.targets) | {rule.id for rule in (*base.events, *base.parallel, *config.events)}
    prefix = "cycle."
    while any(name.startswith(prefix) for name in used):
        prefix = "_" + prefix
    seat_a, seat_b = prefix + "seat_a", prefix + "seat_b"
    confidence = cycle.clear.min_confidence
    supplied = {
        "cycle.a_in": InRule(id=prefix + "a_in", type="In", target=seat_a, min_confidence=confidence),
        "cycle.b_moves_left": MoveRule(
            id=prefix + "b_moves_left", type="Move", target=seat_b, min_confidence=confidence,
            window_ms=450, max_gap_ms=150, min_displacement_ratio=0.04,
            direction=Direction(angle_deg=180, tolerance_deg=35),
        ),
        "cycle.b_out_left": OutRule(
            id=prefix + "b_out_left", type="Out", target=seat_b, min_confidence=confidence,
            edge="left", left_edge_x=exit_x, edge_distance_ratio=0.018, exit_window_ms=500, max_gap_ms=200,
        ),
    }
    targets = {**base.targets, seat_a: Target(class_id=1), seat_b: Target(class_id=2)}
    primitives = base.model_copy(update={"targets": targets, "events": [*base.events, *supplied.values()]})
    events_by_id = {rule.id: rule for rule in primitives.events}
    enter = supplied.get(cycle.start.event) or events_by_id.get(cycle.start.event)
    move = supplied.get(cycle.end.move) or events_by_id.get(cycle.end.move)
    exit_rule = supplied.get(cycle.end.out) or events_by_id.get(cycle.end.out)
    if not isinstance(enter, InRule) or not isinstance(move, MoveRule) or not isinstance(exit_rule, OutRule):
        raise ValueError("Cycle start must reference In, and cycle end must reference Move followed by Out")
    if targets[move.target].class_id != targets[exit_rule.target].class_id:
        raise ValueError("Cycle Move and Out must refer to the same class")
    if not {targets[enter.target].class_id, targets[exit_rule.target].class_id} <= cycle.clear.class_ids:
        raise ValueError("Cycle clear.class_ids must cover the classes used by In and Out")

    composite_events = [*base.parallel, *config.events]
    subjects = {
        rule.id: {alias: targets[alias].class_id for alias in (
            rule.subject_aliases if isinstance(rule, CloseRule) else (rule.target,)
        )}
        for rule in primitives.events
    }
    close = {rule.id: rule for rule in base.events if isinstance(rule, CloseRule)}
    for rule in composite_events:
        if rule.id in subjects:
            raise ValueError(f"Duplicate event id: {rule.id}")
        if isinstance(rule, ParallelRule):
            subjects[rule.id] = primitives.validate_parallel(rule, close)
        elif isinstance(rule, SequenceRule):
            subjects[rule.id] = resolve_subjects(rule.steps, rule.same_tracks, subjects)
        else:
            subjects[rule.id] = resolve_subjects([rule], [], subjects)
    return ResolvedRules(primitives, cycle, enter, move, exit_rule, composite_events)
