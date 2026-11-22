"""Associate YOLO boxes with run-local IDs for a fixed camera."""

from dataclasses import dataclass
from math import dist, hypot, log

from .models import Detection, DetectionFrame


def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(1, (x2 - x1) * (y2 - y1))


@dataclass
class Track:
    detection: Detection
    seen_ms: int
    velocity: tuple[float, float] = (0.0, 0.0)
    matchable: bool = True

    def match_cost(self, detection: Detection, now_ms: int, diagonal: float) -> float | None:
        elapsed = now_ms - self.seen_ms
        old_center = center(self.detection.bbox_xyxy)
        new_center = center(detection.bbox_xyxy)
        if dist(old_center, new_center) / diagonal > min(0.13, 0.037 + elapsed * 0.00025):
            return None
        predicted = tuple(p + v * min(elapsed, 100) for p, v in zip(old_center, self.velocity))
        scale_change = abs(log(area(detection.bbox_xyxy) / area(self.detection.bbox_xyxy)))
        return dist(predicted, new_center) / diagonal + min(scale_change, 2.0) * 0.012

    def observe(self, detection: Detection, now_ms: int) -> None:
        elapsed = now_ms - self.seen_ms
        if elapsed > 0:
            self.velocity = tuple(
                0.5 * (new - old) / elapsed + 0.5 * velocity
                for new, old, velocity in zip(
                    center(detection.bbox_xyxy), center(self.detection.bbox_xyxy), self.velocity,
                )
            )
        self.detection = detection
        self.seen_ms = now_ms


class BoxTracker:
    def __init__(self, retention_ms: int) -> None:
        self.retention_ms = retention_ms
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, frame: DetectionFrame) -> None:
        now_ms = frame.timestamp_ms
        diagonal = hypot(frame.width, frame.height)
        self.tracks = {
            key: track for key, track in self.tracks.items() if now_ms - track.seen_ms <= self.retention_ms
        }
        # Assign the closest plausible matches first, with at most one box per track.
        matches = []
        for index, box in enumerate(frame.detections):
            box.track_id = None
            for key, track in self.tracks.items():
                if track.matchable and box.class_id == track.detection.class_id:
                    cost = track.match_cost(box, now_ms, diagonal)
                    if cost is not None:
                        matches.append((cost, key, index))

        used = set()
        for _, key, index in sorted(matches):
            box = frame.detections[index]
            if key not in used and box.track_id is None:
                box.track_id = key
                self.tracks[key].observe(box, now_ms)
                used.add(key)

        for box in frame.detections:
            if box.track_id is None:
                box.track_id = self.next_id
                self.tracks[self.next_id] = Track(box, now_ms)
                self.next_id += 1
