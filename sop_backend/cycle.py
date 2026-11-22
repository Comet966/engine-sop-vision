"""Detect one workpiece cycle from the station camera."""

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .models import DetectionFrame, WorkpieceCycleEvent


BELT_TOP = 0.61
BELT_BOTTOM = 0.93
ARM_MS = 167
MOTION_MS = 400
MISSING_MS = 400
SEAT_A = 1
SEAT_B = 2
A_STATION = (0.09, 0.30, 0.36, 0.65)
A_STABLE_MS = 150
A_MAX_GAP_MS = 100
MAX_FRAME_GAP_MS = 250


@dataclass(frozen=True)
class FrameTime:
    frame_id: int
    timestamp_ms: int


@dataclass(frozen=True)
class Observation:
    time: FrameTime
    x: float
    y: float
    left: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class ExitEvidence:
    frame_id: int
    last_seen_ms: int
    first_missing_ms: int
    confirmed_ms: int
    last_seen_frame_id: int
    first_missing_frame_id: int


class CycleEndDetector:
    def __init__(self, exit_x: float) -> None:
        self.exit_x = exit_x
        self.reset()

    def reset(self) -> None:
        self.history: deque[Observation] = deque()
        self.last: Observation | None = None
        self.first: Observation | None = None
        self.first_missing: FrameTime | None = None
        self.armed = False
        self.reached_exit = False
        self.state = "WAITING"

    def match(self, candidates: list[Observation], timestamp_ms: int) -> Observation | None:
        last = self.last
        if last is None:
            in_station = [point for point in candidates if 0.36 <= point.x <= 0.66]
            return min(in_station, key=lambda point: abs(point.x - 0.48), default=None)

        gap = max(0, timestamp_ms - last.time.timestamp_ms)
        max_x_gap = min(0.09, 0.035 + gap * 0.00045)
        nearby = [
            point for point in candidates
            if abs(point.x - last.x) <= max_x_gap and abs(point.y - last.y) <= 0.075
        ]
        return min(nearby, key=lambda point: abs(point.x - last.x) + abs(point.y - last.y), default=None)

    def observe(
        self,
        frame_id: int,
        timestamp_ms: int,
        seat_boxes: Iterable[Sequence[int]],
        width: int,
        height: int,
    ) -> tuple[Observation | None, ExitEvidence | None]:
        time = FrameTime(frame_id, timestamp_ms)
        points = []
        for x1, y1, x2, y2 in seat_boxes:
            y = (y1 + y2) / (2 * height)
            if BELT_TOP <= y <= BELT_BOTTOM:
                points.append(Observation(time, (x1 + x2) / (2 * width), y, x1 / width, (x1, y1, x2, y2)))

        point = self.match(points, timestamp_ms)
        if point is None:
            return None, self.on_missing(time)

        self.first_missing = None
        if self.last is not None and timestamp_ms - self.last.time.timestamp_ms > MAX_FRAME_GAP_MS:
            self.history.clear()
        self.last = point
        self.first = self.first or point
        self.history.append(point)
        while self.history[0].time.timestamp_ms < timestamp_ms - MOTION_MS:
            self.history.popleft()

        if not self.armed:
            if abs(point.x - self.first.x) > 0.06:
                self.first = point
            self.armed = (
                timestamp_ms - self.first.time.timestamp_ms >= ARM_MS
                and 0.36 <= self.first.x <= 0.66
            )

        origin = self.history[0]
        moving_left = self.armed and len(self.history) >= 3 and origin.x - point.x > 0.05
        if self.reached_exit and point.x > self.exit_x + 0.08:
            self.reached_exit = False
        if moving_left and self.first.x - point.x >= 0.15:
            self.reached_exit |= point.left <= self.exit_x and point.x <= self.exit_x + 0.055
        self.state = "EXIT ZONE" if self.reached_exit else "MOVING LEFT" if moving_left else "TRACKING B"
        return point, None

    def on_missing(self, time: FrameTime) -> ExitEvidence | None:
        if self.last is None:
            return None
        if self.first_missing is None:
            self.first_missing = time
        if self.reached_exit:
            self.state = "CONFIRMING EXIT"
        elif self.state != "WAITING":
            self.state = "B NOT VISIBLE"
        if time.timestamp_ms - self.first_missing.timestamp_ms < MISSING_MS:
            return None

        event = None
        if self.reached_exit:
            event = ExitEvidence(
                frame_id=time.frame_id,
                last_seen_ms=self.last.time.timestamp_ms,
                first_missing_ms=self.first_missing.timestamp_ms,
                confirmed_ms=time.timestamp_ms,
                last_seen_frame_id=self.last.time.frame_id,
                first_missing_frame_id=self.first_missing.frame_id,
            )
        self.reset()
        return event


def seat_a_in_station(frame: DetectionFrame) -> bool:
    left, top, right, bottom = A_STATION
    for detection in frame.detections:
        if detection.class_id == SEAT_A:
            x1, y1, x2, y2 = detection.bbox_xyxy
            x = (x1 + x2) / (2 * frame.width)
            y = (y1 + y2) / (2 * frame.height)
            if left <= x <= right and top <= y <= bottom:
                return True
    return False


class WorkpieceCycleDetector:
    def __init__(self, exit_x: float) -> None:
        self.exit = CycleEndDetector(exit_x)
        self.active: WorkpieceCycleEvent | None = None
        self.completed = 0
        self.previous_frame: FrameTime | None = None
        self.first_a: FrameTime | None = None
        self.last_a: FrameTime | None = None
        self.require_clear_a = False

    def observe(self, frame: DetectionFrame) -> WorkpieceCycleEvent | None:
        now = FrameTime(frame.frame_id, frame.timestamp_ms)
        if self.previous_frame:
            gap = now.timestamp_ms - self.previous_frame.timestamp_ms
            if now.frame_id <= self.previous_frame.frame_id or gap < 0:
                return None
            if now.frame_id != self.previous_frame.frame_id + 1 or gap > MAX_FRAME_GAP_MS:
                self.first_a = None
                if self.require_clear_a:
                    self.last_a = now
                self.exit.reset()
        self.previous_frame = now

        previous_a = self.last_a
        a_visible = seat_a_in_station(frame)
        if a_visible:
            self.last_a = now
        elif self.require_clear_a and previous_a and now.timestamp_ms - previous_a.timestamp_ms >= A_STABLE_MS:
            self.require_clear_a = False

        if self.active is not None:
            return self._check_exit(frame)
        if not a_visible or self.require_clear_a:
            return None

        if self.first_a is None or previous_a is None or now.timestamp_ms - previous_a.timestamp_ms > A_MAX_GAP_MS:
            self.first_a = now
        if now.timestamp_ms - self.first_a.timestamp_ms < A_STABLE_MS:
            return None

        self.active = WorkpieceCycleEvent(
            phase="started",
            run_id=frame.run_id,
            cycle_id=f"{frame.run_id}:{self.completed + 1}",
            start_ms=self.first_a.timestamp_ms,
            start_frame_id=self.first_a.frame_id,
            confirmed_ms=now.timestamp_ms,
            confirmed_frame_id=now.frame_id,
        )
        self.first_a = None
        return self.active

    def _check_exit(self, frame: DetectionFrame) -> WorkpieceCycleEvent | None:
        seat_boxes = (box.bbox_xyxy for box in frame.detections if box.class_id == SEAT_B)
        _, evidence = self.exit.observe(frame.frame_id, frame.timestamp_ms, seat_boxes, frame.width, frame.height)
        if evidence is None:
            return None

        assert self.active is not None
        ended = self.active.model_copy(update={
            "phase": "ended",
            "end_ms": evidence.first_missing_ms,
            "end_frame_id": evidence.first_missing_frame_id,
            "confirmed_ms": evidence.confirmed_ms,
            "confirmed_frame_id": evidence.frame_id,
            "last_b_seen_ms": evidence.last_seen_ms,
            "last_b_seen_frame_id": evidence.last_seen_frame_id,
        })
        self.active = None
        self.completed += 1
        self.require_clear_a = self.last_a is not None and frame.timestamp_ms - self.last_a.timestamp_ms < A_STABLE_MS
        return ended


