"""Combine primitive event evidence into workpiece cycles and simultaneous actions."""

from collections import defaultdict
from dataclasses import dataclass

from .composite_rules import (
    CompositeRule, EventAliasRule, ResolvedRules, SequenceRule, default_composite_rules, resolve_rules,
)
from .cycle import MAX_FRAME_GAP_MS, FrameTime
from .event_rules import (
    ActiveClose, CompositeEvent, CycleRules, ParallelEvent, ParallelRule, PrimitiveEvent, PrimitiveRules, TrackedTarget,
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
    def __init__(
        self, rules: CycleRules | None, exit_x: float, primitives: PrimitiveRules | None,
        *, resolved: ResolvedRules | None = None,
    ) -> None:
        if resolved is None:
            cycle_config = default_composite_rules().model_copy(update={"events": []})
            resolved = resolve_rules(cycle_config, primitives, exit_x, rules)
        self.rules = resolved.cycle
        self.exit_x = resolved.exit.left_edge_x
        self.primitives = resolved.primitives
        self.in_rule, self.move_rule, self.out_rule = resolved.enter.id, resolved.move.id, resolved.exit.id
        self.seat_a, self.move_target, self.seat_b = resolved.enter.target, resolved.move.target, resolved.exit.target
        self.exit_class = self.primitives.targets[self.seat_b].class_id
        self.out_confirmation_ms = resolved.exit.missing_ms
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

        visible = {box.class_id for box in frame.detections if box.confidence >= self.rules.clear.min_confidence}
        if visible & self.rules.clear.class_ids:
            if self.active is None and self.empty_since is not None:
                duration = now.timestamp_ms - self.empty_since.timestamp_ms
                if duration >= self.rules.start.after_clear_ms:
                    self.arrival = Arrival(now, duration)
            self.empty_since = None
        else:
            self.empty_since = self.empty_since or now
        if self.arrival and now.timestamp_ms - self.arrival.time.timestamp_ms > self.rules.start.within_ms:
            self.arrival = None

        # An Out follows a confirmation delay. Keep the Move until it can be checked against the first missing frame.
        cutoff = now.timestamp_ms - (self.rules.end.within_ms + self.out_confirmation_ms + MAX_FRAME_GAP_MS)
        self.movements = {key: event for key, event in self.movements.items() if event.confirmed_ms >= cutoff}
        for event in events:
            if event.rule_id == self.move_rule:
                self.movements[event.subjects[self.move_target].track_id] = event
            elif event.rule_id == self.out_rule:
                self._record_exit(event)

        if self.exit_class in visible:
            self.pending_exit = None
        if self.pending_exit and self.empty_since:
            empty_ms = now.timestamp_ms - self.empty_since.timestamp_ms
            if empty_ms >= self.rules.end.clear_ms:
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
        if not 0 <= gap <= self.rules.end.within_ms:
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
        for term in rule.terms:
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


@dataclass(frozen=True)
class SequenceAttempt:
    subjects: dict[str, TrackedTarget]
    first_ms: int
    evidence_ids: tuple[str, ...]


class CompositeEventDetector:
    def __init__(self, rules: list[CompositeRule]) -> None:
        self.parallel = ParallelEventDetector([rule for rule in rules if isinstance(rule, ParallelRule)])
        self.ordered_rules = [rule for rule in rules if not isinstance(rule, ParallelRule)]
        self.attempts: dict[str, tuple[str | None, list[SequenceAttempt]]] = {}
        self.previous: DetectionFrame | None = None
        self.next_id = 1

    def update(
        self, frame: DetectionFrame, cycle: WorkpieceCycleEvent | None, primitives: list[PrimitiveEvent],
    ) -> list[CompositeEvent]:
        previous = self.previous
        if previous is not None:
            gap = frame.timestamp_ms - previous.timestamp_ms
            if frame.frame_id <= previous.frame_id or gap < 0:
                return []
            if (frame.frame_id != previous.frame_id + 1 or gap > MAX_FRAME_GAP_MS
                    or (frame.width, frame.height) != (previous.width, previous.height)):
                self.attempts.clear()
                self.parallel.fired.clear()
        self.previous = frame

        events: list[CompositeEvent] = list(self.parallel.update(frame, cycle))
        if not self.ordered_rules:
            return events
        observed: dict[str, list[PrimitiveEvent | CompositeEvent]] = defaultdict(list)
        for event in primitives:
            if event.phase != "ended":
                observed[event.rule_id].append(event)
        for event in events:
            observed[event.rule_id].append(event)
        for rule in self.ordered_rules:
            if rule.within_cycle and cycle is None:
                self.attempts.pop(rule.id, None)
                continue
            cycle_id = cycle.cycle_id if rule.within_cycle and cycle else None
            if isinstance(rule, EventAliasRule):
                matches = [CompositeEvent(
                    op="event", run_id=frame.run_id, rule_id=rule.id,
                    occurrence_id=f"{frame.run_id}:event:{rule.id}:{event.occurrence_id}",
                    cycle_id=cycle_id,
                    subjects={rule.bind.get(alias, alias): subject for alias, subject in event.subjects.items()},
                    start_ms=max(event.start_ms, (cycle.start_ms or 0) if cycle_id and cycle else 0),
                    confirmed_ms=event.confirmed_ms, confirmed_frame_id=event.confirmed_frame_id,
                    evidence_ids=[event.occurrence_id],
                ) for event in observed[rule.event]]
            else:
                matches = self._sequence(frame, rule, cycle_id, observed)
            events.extend(matches)
            observed[rule.id].extend(matches)
        return events

    def _sequence(
        self, frame: DetectionFrame, rule: SequenceRule, cycle_id: str | None,
        observed: dict[str, list[PrimitiveEvent | CompositeEvent]],
    ) -> list[CompositeEvent]:
        old_cycle_id, previous = self.attempts.get(rule.id, (None, []))
        previous = [
            attempt for attempt in previous if cycle_id == old_cycle_id
            and frame.timestamp_ms - attempt.first_ms <= rule.within_ms
        ]
        # Existing attempts represent earlier frames. New evidence can advance each by one step.
        initial = SequenceAttempt({}, frame.timestamp_ms, ())
        future = previous.copy()
        completed = {}
        consumed = set()
        for attempt in (*previous, initial):
            term = rule.steps[len(attempt.evidence_ids)]
            for evidence in observed[term.event]:
                subjects = {term.bind.get(alias, alias): target for alias, target in evidence.subjects.items()}
                if any(attempt.subjects.get(alias, target) != target for alias, target in subjects.items()):
                    continue
                updated = SequenceAttempt(
                    attempt.subjects | subjects, attempt.first_ms, (*attempt.evidence_ids, evidence.occurrence_id),
                )
                if len(updated.evidence_ids) == len(rule.steps):
                    key = tuple(sorted((alias, target.track_id) for alias, target in updated.subjects.items()))
                    completed[key] = updated
                    consumed.add(updated.evidence_ids[0])
                else:
                    future.append(updated)

        # The newest candidate for the same bindings and next step has the longest remaining window.
        pending = {}
        for attempt in future:
            if attempt.evidence_ids[0] in consumed:
                continue
            key = (len(attempt.evidence_ids), tuple(sorted(
                (alias, target.track_id) for alias, target in attempt.subjects.items()
            )))
            if key not in pending or attempt.first_ms >= pending[key].first_ms:
                pending[key] = attempt
        self.attempts[rule.id] = (cycle_id, list(pending.values()))

        events = []
        for match in completed.values():
            events.append(CompositeEvent(
                op="sequence", run_id=frame.run_id, rule_id=rule.id,
                occurrence_id=f"{frame.run_id}:sequence:{self.next_id}",
                cycle_id=cycle_id, subjects=match.subjects, start_ms=match.first_ms,
                confirmed_ms=frame.timestamp_ms, confirmed_frame_id=frame.frame_id,
                evidence_ids=list(match.evidence_ids),
            ))
            self.next_id += 1
        return events
