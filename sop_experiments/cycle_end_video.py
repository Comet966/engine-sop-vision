"""Prototype: mark a cycle when the same Seat B track exits left."""

import argparse
from collections.abc import Iterator
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import cv2

from sop_backend.cycle import BELT_BOTTOM, BELT_TOP, SEAT_B, CycleEndDetector, ExitEvidence, Observation
from sop_backend.detection import FrameReader
from yolo_deployment.infer_video import decode, draw_detections, load_backend, prepare


def draw_status(frame, detector: CycleEndDetector, point: Observation | None, events: list[ExitEvidence], t: int) -> None:
    height, width = frame.shape[:2]
    x = round(detector.exit_x * width)
    y1, y2 = round(BELT_TOP * height), round(BELT_BOTTOM * height)
    yellow = (62, 219, 255)
    green = (123, 245, 140)
    for y in range(y1, y2, 14):
        cv2.line(frame, (x, y), (x, min(y + 8, y2)), yellow, 2)
    cv2.putText(frame, "EXIT", (x + 5, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, yellow, 1, cv2.LINE_AA)
    if point:
        x1, y1, x2, y2 = point.box
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), yellow, 2)

    last = events[-1] if events else None
    show_end = last is not None and t - last.confirmed_ms < 1800
    status = f"CYCLE END #{len(events)}" if show_end else detector.state
    scale = 0.62 if width >= 1000 else 0.51
    left = width - (295 if width >= 1000 else 244)
    lines = (
        (f"{t / 1000:06.2f}s     End count {len(events)}", (245, 245, 245)),
        (status, green if show_end else yellow),
        (f"Last exit {last.first_missing_ms / 1000:.2f}s" if last else "Waiting for workpiece exit", (210, 217, 226)),
    )
    cv2.rectangle(frame, (left, 8), (width - 8, 81), (28, 24, 20), -1)
    for index, (label, color) in enumerate(lines):
        cv2.putText(frame, label, (left + 10, 28 + 21 * index), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def read_cache(path: Path) -> Iterator[tuple[int, list]]:
    with path.open() as cache:
        for line in cache:
            yield json.loads(line)


def render(source: Path, output: Path, model: Path, exit_x: float, cache: Path | None) -> tuple[int, list[ExitEvidence]]:
    rate = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        text=True,
    ).strip()
    if Fraction(rate) <= 0:
        raise ValueError("Video frame rate must be positive")
    if output.exists():
        raise FileExistsError(output)

    cv2.setNumThreads(2)
    backend = load_backend(model) if cache is None else None
    cached = read_cache(cache) if cache else None
    detector = CycleEndDetector(exit_x)
    reader = FrameReader(source)
    events = []
    with TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "result.mp4"
        width = int(reader.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size",
            f"{width}x{height}", "-framerate", rate, "-i", "pipe:0", "-i", str(source), "-map", "0:v:0",
            "-map", "1:a:0?", "-c:v", "libx264", "-threads", "3", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(staged),
        ]
        try:
            with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
                assert encoder.stdin is not None
                while (sample := reader.read()) is not None:
                    frame_id, t, frame = sample
                    if cached is not None:
                        cached_t, detections = next(cached)
                        if cached_t != t:
                            raise ValueError(f"Cache timestamp mismatch at frame {frame_id}")
                    else:
                        tensor, *transform = prepare(frame)
                        assert backend is not None
                        detections = decode(backend(tensor), transform, frame.shape, 0.25)
                    seat_boxes = (box for category, _, box in detections if category == SEAT_B)
                    point, event = detector.observe(frame_id, t, seat_boxes, width, height)
                    if event is not None:
                        events.append(event)
                    draw_detections(frame, detections)
                    draw_status(frame, detector, point, events, t)
                    encoder.stdin.write(frame.tobytes())
                if cached is not None and next(cached, None) is not None:
                    raise ValueError("Detection cache has more frames than the source")
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError("FFmpeg could not encode the annotated video")
            staged.rename(output)
        finally:
            reader.close()
    return reader.frame_id, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Path("yolo_deployment/best.onnx"))
    parser.add_argument("--exit-x", type=float, required=True, help="Left edge of the visible belt, normalized by width")
    parser.add_argument("--detections", type=Path, help="Optional JSONL cache: [timestamp_ms, detections]")
    args = parser.parse_args()
    if not 0 <= args.exit_x < 0.30:
        parser.error("--exit-x must be between 0 and 0.30")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames, events = render(args.source, args.output, args.model, args.exit_x, args.detections)
    print(json.dumps({"frames": frames, "events": [asdict(event) for event in events]}, indent=2))


if __name__ == "__main__":
    main()
