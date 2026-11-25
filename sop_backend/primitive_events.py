"""Detect motion, proximity, entry and exit from sequential YOLO frames."""

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product
from math import atan2, degrees, dist, hypot
from typing import Literal

from .cycle import MAX_FRAME_GAP_MS, FrameTime
from .event_rules import (
    ActiveClose, BoxSize, CloseRule, Edge, InRule, MoveRule, OutRule, PrimitiveEvent, PrimitiveRule,
    PrimitiveRules, TrackedTarget,
)
from .models import Detection, DetectionFrame
from .tracking import BoxTracker, area, center


CloseReason = Literal["separated", "unobserved", "stream_gap", "run_finished"]
Point = tuple[float, float]
Box = tuple[int, int, int, int]


@dataclass
class Dwell:
    start: FrameTime
    last: FrameTime
    segment: FrameTime
    observed_ms: int = 0
    continuous: bool = True

    @classmethod
    def begin(cls, now: FrameTime) -> "Dwell":
        return cls(now, now, now)

    def see(self, now: FrameTime) -> None:
        if self.continuous:
            self.observed_ms += now.timestamp_ms - self.last.timestamp_ms
        else:
            self.segment = now
        self.last = now
        self.continuous = True


@dataclass
class Motion:
    raw: deque[Point] = field(default_factory=lambda: deque(maxlen=3))
    history: deque[tuple[FrameTime, Point]] = field(default_factory=deque)
    fired: bool = False


@dataclass
class Departure:
    history: deque[tuple[FrameTime, Box]] = field(default_factory=deque)
    first_missing: FrameTime | None = None
    fired: bool = False
    last_size_match_frame_id: int = -1


@dataclass
class Proximity:
    dwell: Dwell
    started: PrimitiveEvent | None = None


def bbox_gap(a: Box, b: Box) -> float:
    return hypot(max(0, a[0] - b[2], b[0] - a[2]), max(0, a[1] - b[3], b[1] - a[3]))


def close_distance(a: Box, b: Box, metric: str) -> float:
    if metric == "center_distance":
        return dist(center(a), center(b))
    if metric == "bottom_edge_gap":
        a = (a[0], a[3], a[2], a[3])
    return bbox_gap(a, b)


def enclosing_box(a: Box, b: Box) -> Box:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def bbox_overlap_ratio(a: Box, b: Box) -> float:
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height / min(area(a), area(b))


def matches_box_size(size: BoxSize, box: Box, width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    width_ratio, height_ratio = (x2 - x1) / width, (y2 - y1) / height
    if width_ratio <= 0 or height_ratio <= 0:
        return False
    if size.metric == "height_ratio":
        return size.contains(height_ratio)
    if size.metric == "width_ratio":
        return size.contains(width_ratio)
    return size.contains(width_ratio * height_ratio)


def gap_exceeded(last: FrameTime, now: FrameTime, max_gap_ms: int, visible: bool) -> bool:
    if visible and now.frame_id == last.frame_id + 1:
        return False
    return now.timestamp_ms - last.timestamp_ms > max_gap_ms


def outward_motion(
    history: deque[tuple[FrameTime, Box]], rule: OutRule, width: int, height: int,
) -> tuple[Edge, float] | None:
    if len(history) < 2:
        return None
    diagonal = hypot(width, height)
    _, first = history[0]
    _, last = history[-1]
    x0, y0 = center(first)
    x, y = center(last)
    dx, dy = (x - x0) / diagonal, (y - y0) / diagonal
    edges: tuple[tuple[Edge, float, float], ...] = (
        ("left", last[0] - width * rule.left_edge_x, -dx), ("right", width - last[2], dx),
        ("top", last[1], -dy), ("bottom", height - last[3], dy),
    )
    candidates = (
        (edge, delta) for edge, gap, delta in edges
        if (rule.edge is None or edge == rule.edge)
        and gap / diagonal <= rule.edge_distance_ratio and delta >= rule.min_displacement_ratio
    )
    return max(candidates, key=lambda candidate: candidate[1], default=None)


class PrimitiveEventDetector:
    """One instance per run. Call update once for every decoded frame, including empty ones."""

    def __init__(self, config: PrimitiveRules) -> None:
        self.config = config
        self.tracker = BoxTracker(max(
            [700] + [rule.max_gap_ms + MAX_FRAME_GAP_MS for rule in config.events]
            + [rule.missing_ms + 2 * MAX_FRAME_GAP_MS for rule in config.events if isinstance(rule, OutRule)],
        ))
        self.sequence = 0
        self.previous_frame: DetectionFrame | None = None
        self.motion: dict[str, dict[int, Motion]] = defaultdict(dict)
        self.entries: dict[str, dict[int, Dwell]] = defaultdict(dict)
        self.entered: dict[str, set[int]] = defaultdict(set)
        self.departures: dict[str, dict[int, Departure]] = defaultdict(dict)
        self.proximity: dict[str, dict[tuple[int, ...], Proximity]] = defaultdict(dict)

    def update(self, frame: DetectionFrame) -> list[PrimitiveEvent]:
        events = []
        now = FrameTime(frame.frame_id, frame.timestamp_ms)
        previous = self.previous_frame
        if previous is not None:
            gap = now.timestamp_ms - previous.timestamp_ms
            if now.frame_id <= previous.frame_id or gap < 0:
                return []
            if (now.frame_id != previous.frame_id + 1 or gap > MAX_FRAME_GAP_MS
                    or (frame.width, frame.height) != (previous.width, previous.height)):
                events.extend(self.finish("stream_gap", now=now))
        self.previous_frame = frame
        self.tracker.update(frame)
        visible = defaultdict(dict)
        for detection in frame.detections:
            key = detection.track_id
            assert key is not None
            visible[detection.class_id][key] = detection

        frame.active_close.clear()
        for rule in self.config.events:
            if isinstance(rule, CloseRule):
                first, second = (
                    self._size_matches(target, self._eligible(rule, target, visible), frame, rule.bbox_sizes.get(target))
                    for target in rule.targets
                )
                if rule.union_with:
                    union = rule.union_with
                    partners = self._size_matches(
                        union, self._eligible(rule, union, visible), frame, rule.bbox_sizes.get(union),
                    )
                    unions = {
                        (a, partner): enclosing_box(box.bbox_xyxy, added.bbox_xyxy)
                        for (a, box), (partner, added) in product(first.items(), partners.items())
                        if a != partner
                    }
                    candidates = (
                        ((a, b, partner), box)
                        for ((a, partner), box), b in product(unions.items(), second) if b != partner
                    )
                else:
                    candidates = (((a, b), first[a].bbox_xyxy) for a, b in product(first, second))
                events.extend(self._close(rule, first, second, candidates, frame, now))
            else:
                boxes = self._eligible(rule, rule.target, visible)
                sized = self._size_matches(rule.target, boxes, frame)
                if isinstance(rule, MoveRule):
                    events.extend(self._move(rule, sized, frame, now))
                elif isinstance(rule, InRule):
                    events.extend(self._in(rule, sized, frame, now))
                else:
                    events.extend(self._out(rule, boxes, sized, frame, now))
        return events

    def _eligible(
        self, rule: PrimitiveRule, target: str, visible: dict[int, dict[int, Detection]],
    ) -> dict[int, Detection]:
        category = self.config.targets[target].class_id
        return {key: box for key, box in visible[category].items() if box.confidence >= rule.min_confidence}

    def _size_matches(
        self, target: str, boxes: dict[int, Detection], frame: DetectionFrame, override: BoxSize | None = None,
    ) -> dict[int, Detection]:
        size = override or self.config.targets[target].bbox_size
        if size is None:
            return boxes
        return {
            key: box for key, box in boxes.items()
            if matches_box_size(size, box.bbox_xyxy, frame.width, frame.height)
        }

    def _event(
        self, rule: PrimitiveRule, frame: DetectionFrame, keys: tuple[int, ...], start: FrameTime,
        now: FrameTime, *, end: FrameTime | None = None, evidence: dict[str, int | float | str] | None = None,
    ) -> PrimitiveEvent:
        self.sequence += 1
        targets = rule.subject_aliases if isinstance(rule, CloseRule) else (rule.target,)
        return PrimitiveEvent(
            event=rule.type, phase="started" if end is None else "detected", run_id=frame.run_id,
            sequence=self.sequence, rule_id=rule.id, occurrence_id=f"{frame.run_id}:{self.sequence}",
            subjects={
                target: TrackedTarget(track_id=key, class_id=self.config.targets[target].class_id)
                for target, key in zip(targets, keys)
            },
            start_ms=start.timestamp_ms, start_frame_id=start.frame_id,
            end_ms=end.timestamp_ms if end else None, end_frame_id=end.frame_id if end else None,
            confirmed_ms=now.timestamp_ms, confirmed_frame_id=now.frame_id,
            evidence=evidence or {},
        )

    def _move(
        self, rule: MoveRule, boxes: dict[int, Detection], frame: DetectionFrame, now: FrameTime,
    ) -> list[PrimitiveEvent]:
        states = self.motion[rule.id]
        for key in list(states):
            if gap_exceeded(states[key].history[-1][0], now, rule.max_gap_ms, key in boxes):
                del states[key]

        events = []
        diagonal = hypot(frame.width, frame.height)
        for key, box in boxes.items():
            state = states.setdefault(key, Motion())
            state.raw.append(center(box.bbox_xyxy))
            xs, ys = zip(*state.raw)
            current = (sum(xs) / len(xs), sum(ys) / len(ys))
            history = state.history
            history.append((now, current))
            while now.timestamp_ms - history[0][0].timestamp_ms > rule.window_ms:
                history.popleft()

            origin, (x, y) = history[0]
            if state.fired:
                stable = max(dist(current, point) for _, point in history) / diagonal
                if now.timestamp_ms - origin.timestamp_ms >= 0.8 * rule.window_ms:
                    state.fired = stable > 0.4 * rule.min_displacement_ratio
                continue
            if len(history) < 3:
                continue
            dx, dy = current[0] - x, current[1] - y
            displacement = hypot(dx, dy) / diagonal
            angle = degrees(atan2(dy, dx)) % 360
            direction = rule.direction
            if direction and abs((angle - direction.angle_deg + 180) % 360 - 180) > direction.tolerance_deg:
                continue
            if displacement >= rule.min_displacement_ratio:
                state.fired = True
                events.append(self._event(
                    rule, frame, (key,), origin, now, end=now,
                    evidence={"displacement_ratio": displacement, "angle_deg": angle},
                ))
        return events

    def _in(
        self, rule: InRule, boxes: dict[int, Detection], frame: DetectionFrame, now: FrameTime,
    ) -> list[PrimitiveEvent]:
        entered = self.entered[rule.id]
        entered.intersection_update(self.tracker.tracks)
        states = self.entries[rule.id]
        for key, dwell in list(states.items()):
            if gap_exceeded(dwell.last, now, rule.max_gap_ms, key in boxes):
                del states[key]
            elif key not in boxes:
                dwell.continuous = False

        events = []
        for key in boxes:
            if key in entered:
                continue
            dwell = states.setdefault(key, Dwell.begin(now))
            dwell.see(now)
            if dwell.observed_ms >= rule.confirm_ms:
                entered.add(key)
                del states[key]
                events.append(self._event(rule, frame, (key,), dwell.start, now, end=now))
        return events

    def _out(
        self, rule: OutRule, boxes: dict[int, Detection], sized: dict[int, Detection],
        frame: DetectionFrame, now: FrameTime,
    ) -> list[PrimitiveEvent]:
        states = self.departures[rule.id]
        for key, box in boxes.items():
            state = states.setdefault(key, Departure())
            if state.fired or (state.history and gap_exceeded(state.history[-1][0], now, rule.max_gap_ms, True)):
                state = states[key] = Departure()
            state.first_missing = None
            # A partially clipped box is still visible. Size qualifies the trajectory, not absence.
            if key in sized:
                state.last_size_match_frame_id = now.frame_id
            state.history.append((now, box.bbox_xyxy))
            while now.timestamp_ms - state.history[0][0].timestamp_ms > rule.exit_window_ms:
                state.history.popleft()

        events = []
        for key in states.keys() - boxes.keys():
            if key not in self.tracker.tracks:
                del states[key]
                continue
            state = states[key]
            if state.fired:
                continue
            state.first_missing = state.first_missing or now
            if now.timestamp_ms - state.first_missing.timestamp_ms < rule.missing_ms:
                continue
            state.fired = True
            if state.last_size_match_frame_id < state.history[0][0].frame_id:
                continue
            motion = outward_motion(state.history, rule, frame.width, frame.height)
            if motion:
                self.tracker.tracks[key].matchable = False
                edge, displacement = motion
                events.append(self._event(
                    rule, frame, (key,), state.history[0][0], now, end=state.first_missing,
                    evidence={"edge": edge, "outward_displacement_ratio": displacement,
                              "last_seen_ms": state.history[-1][0].timestamp_ms,
                              "last_seen_frame_id": state.history[-1][0].frame_id},
                ))
        return events

    def _close(
        self, rule: CloseRule, first: dict[int, Detection], second: dict[int, Detection],
        candidates: Iterable[tuple[tuple[int, ...], Box]],
        frame: DetectionFrame, now: FrameTime,
    ) -> list[PrimitiveEvent]:
        states = self.proximity[rule.id]
        left, right = (self.config.targets[alias] for alias in rule.targets)
        symmetric = (
            left == right and rule.bbox_height_ratio is None and not rule.bbox_sizes
            and rule.distance_metric != "bottom_edge_gap" and rule.union_with is None
        )
        diagonal = hypot(frame.width, frame.height)
        min_overlap = rule.min_overlap_ratio or 0
        distances: dict[tuple[int, ...], tuple[float, float | None, float | None]] = {}
        for pair, distance_box in candidates:
            a, b = pair[:2]
            if a == b or (symmetric and a > b):
                continue
            box_a, box_b = first[a].bbox_xyxy, second[b].bbox_xyxy
            height_ratio = None
            if rule.bbox_height_ratio is not None:
                height_b = box_b[3] - box_b[1]
                if height_b <= 0:
                    continue
                height_ratio = (box_a[3] - box_a[1]) / height_b
                if not rule.bbox_height_ratio.contains(height_ratio):
                    continue
            overlap = bbox_overlap_ratio(distance_box, box_b) if min_overlap else None
            distance = close_distance(distance_box, box_b, rule.distance_metric) / diagonal
            distances[pair] = (distance, height_ratio, overlap)
        events = []
        for pair, state in list(states.items()):
            match = distances.get(pair)
            reason: CloseReason | None = None
            if gap_exceeded(state.dwell.last, now, rule.max_gap_ms, match is not None):
                reason = "unobserved"
            elif match is not None and (
                match[0] > rule.exit_distance_ratio or (match[2] is not None and match[2] < min_overlap)
            ):
                reason = "separated"
            elif match is None:
                state.dwell.continuous = False
            if reason is not None:
                if state.started:
                    events.append(self._end_close(state, now, reason))
                del states[pair]

        for pair, (distance, height_ratio, overlap) in distances.items():
            if overlap is not None and overlap < min_overlap:
                continue
            state = states.get(pair)
            if state is None:
                if distance > rule.enter_distance_ratio:
                    continue
                state = states[pair] = Proximity(Dwell.begin(now))
            state.dwell.see(now)
            if state.started is None and state.dwell.observed_ms >= rule.hold_ms:
                evidence = {"distance_ratio": distance, "observed_ms": state.dwell.observed_ms}
                if height_ratio is not None:
                    evidence["bbox_height_ratio"] = height_ratio
                if overlap is not None:
                    evidence["bbox_overlap_ratio"] = overlap
                state.started = self._event(
                    rule, frame, pair, state.dwell.start, now, evidence=evidence,
                )
                events.append(state.started)
            if state.started:
                frame.active_close.append(ActiveClose(
                    rule_id=rule.id, occurrence_id=state.started.occurrence_id, subjects=state.started.subjects,
                    evidence_start_ms=state.dwell.segment.timestamp_ms,
                ))
        return events

    def _end_close(self, state: Proximity, now: FrameTime, reason: CloseReason) -> PrimitiveEvent:
        assert state.started is not None
        self.sequence += 1
        return state.started.model_copy(update={
            "phase": "ended", "sequence": self.sequence, "reason": reason,
            "end_ms": state.dwell.last.timestamp_ms, "end_frame_id": state.dwell.last.frame_id,
            "confirmed_ms": now.timestamp_ms, "confirmed_frame_id": now.frame_id,
            "evidence": {"observed_ms": state.dwell.observed_ms},
        })

    def finish(self, reason: CloseReason = "run_finished", *, now: FrameTime | None = None) -> list[PrimitiveEvent]:
        if self.previous_frame is None:
            return []
        now = now or FrameTime(self.previous_frame.frame_id, self.previous_frame.timestamp_ms)
        events = [
            self._end_close(state, now, reason)
            for states in self.proximity.values() for state in states.values() if state.started
        ]
        self.tracker.tracks.clear()
        self.motion.clear()
        self.entries.clear()
        self.entered.clear()
        self.departures.clear()
        self.proximity.clear()
        return events
