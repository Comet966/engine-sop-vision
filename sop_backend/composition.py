"""Combine primitive event evidence into workpiece cycles and simultaneous actions."""

from collections import defaultdict
from dataclasses import dataclass

from .cycle import MAX_FRAME_GAP_MS, SEAT_A, SEAT_B, FrameTime
from .event_rules import (
    ActiveClose, CycleRules, Direction, InRule, MoveRule, OutRule, ParallelEvent, ParallelRule,
    PrimitiveEvent, PrimitiveRules, Target, TrackedTarget,
)
from .models import DetectionFrame, WorkpieceCycleEvent


@dataclass(frozen=True)
class Arrival:
    time: FrameTime
    empty_ms: int


@dataclass(frozen=True)
class Departure:
    motion: PrimitiveEvent
    exit: PrimitiveEvent


class WorkpieceCycleComposer:
    def __init__(self, rules: CycleRules, exit_x: float, primitives: PrimitiveRules | None) -> None:
        self.rules = rules
        self.exit_x = exit_x
        base = primitives or PrimitiveRules(targets={}, events=[])
        used = set(base.targets) | {rule.id for rule in (*base.events, *base.parallel)}
        prefix = "cycle."
        while any(name.startswith(prefix) for name in used):
            prefix = "_" + prefix
        self.seat_a = prefix + "seat_a"
        self.seat_b = prefix + "seat_b"
        self.in_rule = prefix + "a_in"
        self.move_rule = prefix + "b_moves_left"
        self.out_rule = prefix + "b_out_left"
        exit_rule = OutRule(
            id=self.out_rule, type="Out", target=self.seat_b, min_confidence=rules.min_confidence,
            edge="left", left_edge_x=exit_x, edge_distance_ratio=0.018,
            exit_window_ms=500, max_gap_ms=200,
        )
        self.out_confirmation_ms = exit_rule.missing_ms
        self.primitives = base.model_copy(update={
            "targets": {**base.targets, self.seat_a: Target(class_id=SEAT_A), self.seat_b: Target(class_id=SEAT_B)},
            "events": [
                *base.events,
                InRule(id=self.in_rule, type="In", target=self.seat_a, min_confidence=rules.min_confidence),
                MoveRule(
                    id=self.move_rule, type="Move", target=self.seat_b, min_confidence=rules.min_confidence,
                    window_ms=450, max_gap_ms=150, min_displacement_ratio=0.04,
                    direction=Direction(angle_deg=180, tolerance_deg=35),
                ),
                exit_rule,
            ],
        })
        self.active: WorkpieceCycleEvent | None = None
        self.completed = 0
        self.started = 0
        self.previous: FrameTime | None = None
        self.frame_size: tuple[int, int] | None = None
        self.empty_since: FrameTime | None = None
        self.arrival: Arrival | None = None
        self.movements: dict[int, PrimitiveEvent] = {}
        self.pending_exit: Departure | None = None

    def update(self, frame: DetectionFrame, events: list[PrimitiveEvent]) -> WorkpieceCycleEvent | None:
        now = FrameTime(frame.frame_id, frame.timestamp_ms)
        size = (frame.width, frame.height)
        if self.previous is not None:
            gap = now.timestamp_ms - self.previous.timestamp_ms
            if now.frame_id <= self.previous.frame_id or gap < 0:
                return None
            if now.frame_id != self.previous.frame_id + 1 or gap > MAX_FRAME_GAP_MS or size != self.frame_size:
                self.active = None
                self.empty_since = None
                self.arrival = None
                self.movements.clear()
                self.pending_exit = None
        self.previous = now
        self.frame_size = size

        visible = {box.class_id for box in frame.detections if box.confidence >= self.rules.min_confidence}
        if SEAT_A in visible or SEAT_B in visible:
            if self.active is None and self.empty_since is not None:
                duration = now.timestamp_ms - self.empty_since.timestamp_ms
                if duration >= self.rules.empty_before_ms:
                    self.arrival = Arrival(now, duration)
            self.empty_since = None
        else:
            self.empty_since = self.empty_since or now
        if self.arrival and now.timestamp_ms - self.arrival.time.timestamp_ms > self.rules.a_wait_ms:
            self.arrival = None

        # An Out follows a confirmation delay. Keep the Move until it can be checked against the first missing frame.
        cutoff = now.timestamp_ms - (self.rules.move_to_out_ms + self.out_confirmation_ms + MAX_FRAME_GAP_MS)
        self.movements = {key: event for key, event in self.movements.items() if event.confirmed_ms >= cutoff}
        for event in events:
            if event.rule_id == self.move_rule:
                self.movements[event.subjects[self.seat_b].track_id] = event
            elif event.rule_id == self.out_rule:
                self._record_exit(event)

        if SEAT_B in visible:
            self.pending_exit = None
        if self.pending_exit and self.empty_since:
            empty_ms = now.timestamp_ms - self.empty_since.timestamp_ms
            if empty_ms >= self.rules.empty_after_ms:
                return self._end(frame, empty_ms)
        if self.active is None and self.arrival is not None:
            for event in events:
                if event.rule_id == self.in_rule and event.start_frame_id >= self.arrival.time.frame_id:
                    return self._start(frame, event, self.arrival)
        return None

    def _record_exit(self, event: PrimitiveEvent) -> None:
        track = event.subjects[self.seat_b].track_id
        move = self.movements.get(track)
        if move is None or event.end_ms is None:
            return
        gap = event.end_ms - move.confirmed_ms
        if not 0 <= gap <= self.rules.move_to_out_ms:
            return
        if self.active and self.active.start_frame_id is not None and move.confirmed_frame_id < self.active.start_frame_id:
            return
        self.pending_exit = self.pending_exit or Departure(move, event)

    def _start(self, frame: DetectionFrame, event: PrimitiveEvent, arrival: Arrival) -> WorkpieceCycleEvent:
        self.started += 1
        self.active = WorkpieceCycleEvent(
            phase="started", run_id=frame.run_id, cycle_id=f"{frame.run_id}:{self.started}",
            start_ms=event.start_ms, start_frame_id=event.start_frame_id,
            confirmed_ms=frame.timestamp_ms, confirmed_frame_id=frame.frame_id,
            a_track_id=event.subjects[self.seat_a].track_id, empty_ms=arrival.empty_ms,
        )
        self.arrival = None
        return self.active

    def _end(self, frame: DetectionFrame, empty_ms: int) -> WorkpieceCycleEvent:
        departure = self.pending_exit
        assert departure is not None
        observed_start = self.active
        event = departure.exit
        ended = WorkpieceCycleEvent(
            phase="ended", run_id=frame.run_id,
            cycle_id=observed_start.cycle_id if observed_start else f"{frame.run_id}:partial:{event.end_frame_id}",
            start_ms=observed_start.start_ms if observed_start else None,
            start_frame_id=observed_start.start_frame_id if observed_start else None,
            a_track_id=observed_start.a_track_id if observed_start else None,
            end_ms=event.end_ms, end_frame_id=event.end_frame_id,
            confirmed_ms=frame.timestamp_ms, confirmed_frame_id=frame.frame_id,
            last_b_seen_ms=int(event.evidence["last_seen_ms"]),
            last_b_seen_frame_id=int(event.evidence["last_seen_frame_id"]),
            partial=observed_start is None, b_track_id=event.subjects[self.seat_b].track_id,
            empty_ms=empty_ms, b_move_confirmed_ms=departure.motion.confirmed_ms,
        )
        if observed_start:
            self.completed += 1
        self.active = None
        self.pending_exit = None
        self.arrival = None
        self.movements.clear()
        return ended


@dataclass(frozen=True)
class ParallelMatch:
    subjects: dict[str, TrackedTarget]
    start_ms: int
    evidence_ids: tuple[str, ...] = ()


class ParallelEventDetector:
    def __init__(self, rules: list[ParallelRule]) -> None:
        self.rules = rules
        self.fired: dict[str, set[tuple[str | None, tuple[tuple[str, int], ...]]]] = {}
        self.next_id = 1

    def update(self, frame: DetectionFrame, cycle: WorkpieceCycleEvent | None) -> list[ParallelEvent]:
        if not self.rules:
            return []
        active: dict[str, list[ActiveClose]] = defaultdict(list)
        for close in frame.active_close:
            active[close.rule_id].append(close)
        events = []
        for rule in self.rules:
            if rule.within_cycle and cycle is None:
                self.fired.pop(rule.id, None)
                continue
            scope = cycle if rule.within_cycle else None
            events.extend(self._match(frame, rule, active, scope))
        return events

    def _match(
        self, frame: DetectionFrame, rule: ParallelRule, active: dict[str, list[ActiveClose]],
        cycle: WorkpieceCycleEvent | None,
    ) -> list[ParallelEvent]:
        cycle_id = cycle.cycle_id if cycle else None
        matches = [ParallelMatch({}, (cycle.start_ms or 0) if cycle else 0)]
        for term in rule.all:
            joined = []
            for close in active[term.event]:
                subjects = {term.bind.get(alias, alias): target for alias, target in close.subjects.items()}
                for previous in matches:
                    if any(previous.subjects.get(alias, target) != target for alias, target in subjects.items()):
                        continue
                    joined.append(ParallelMatch(
                        previous.subjects | subjects, max(previous.start_ms, close.evidence_start_ms),
                        (*previous.evidence_ids, close.occurrence_id),
                    ))
            matches = joined
            if not matches:
                break

        fired = set()
        previous = self.fired.get(rule.id, set())
        events = []
        for match in matches:
            if frame.timestamp_ms - match.start_ms < rule.min_overlap_ms:
                continue
            key = (cycle_id, tuple(sorted((alias, target.track_id) for alias, target in match.subjects.items())))
            if key not in previous and key not in fired:
                events.append(ParallelEvent(
                    run_id=frame.run_id, rule_id=rule.id, occurrence_id=f"{frame.run_id}:parallel:{self.next_id}",
                    cycle_id=cycle_id, subjects=match.subjects, start_ms=match.start_ms,
                    confirmed_ms=frame.timestamp_ms, confirmed_frame_id=frame.frame_id,
                    evidence_ids=list(match.evidence_ids),
                ))
                self.next_id += 1
            fired.add(key)
        self.fired[rule.id] = fired
        return events
