"""Read video frames and adapt the deployed four-class detector for the API."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from yolo_deployment.infer_video import SIZE, TensorRT, decode, draw_detections, load_backend, prepare

from .models import Detection, DetectionFrame


LABELS = ("手套", "瓦片安装位置 A", "瓦片安装位置 B", "吸管")


class SourceError(RuntimeError):
    """A capture error safe to report to the client."""


@dataclass(frozen=True)
class DetectionPacket:
    result: DetectionFrame
    jpeg: bytes


class FrameReader:
    def __init__(self, source: Path | int) -> None:
        self.camera = isinstance(source, int)
        self.capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        self.frame_id = 0
        self.started_at = monotonic()
        self.last_timestamp = -1.0
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

        if not self.capture.isOpened():
            self.close()
            raise SourceError("无法打开视频源")
        if not self.camera and (not isfinite(self.fps) or self.fps <= 0):
            self.close()
            raise SourceError("无法读取录像帧率")

    def read(self) -> tuple[int, int, np.ndarray] | None:
        ok, image = self.capture.read()
        if not ok:
            if self.camera or self.frame_id == 0:
                raise SourceError("无法从视频源读取画面")
            return None

        if self.camera:
            timestamp = (monotonic() - self.started_at) * 1000
        else:
            timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC)
            if not isfinite(timestamp) or timestamp < 0 or timestamp <= self.last_timestamp:
                timestamp = 0.0 if self.frame_id == 0 else self.last_timestamp + 1000 / self.fps
        self.last_timestamp = timestamp
        frame_id = self.frame_id
        self.frame_id += 1
        return frame_id, round(timestamp), image

    def close(self) -> None:
        self.capture.release()


class Detector:
    def __init__(self, path: Path) -> None:
        cv2.setNumThreads(2)
        self.backend = load_backend(path)
        try:
            # Catch incompatible model exports at startup, before accepting a run.
            sample = self.backend(np.zeros((1, 3, SIZE, SIZE), dtype=np.float32))
            decode(sample, (1.0, 0, 0), (SIZE, SIZE), 0.25)
        except Exception:
            self.close()
            raise

    def read_frame(self, reader: FrameReader, run_id: str, confidence: float) -> DetectionPacket | None:
        frame = reader.read()
        if frame is None:
            return None

        frame_id, timestamp_ms, image = frame
        tensor, gain, left, top = prepare(image)
        boxes = decode(self.backend(tensor), (gain, left, top), image.shape, confidence)
        height, width = image.shape[:2]
        result = DetectionFrame(
            run_id=run_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            width=width,
            height=height,
            detections=[
                Detection(class_id=category, label=LABELS[category], confidence=score, bbox_xyxy=box)
                for category, score, box in boxes
            ],
        )
        draw_detections(image, boxes)
        encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not encoded:
            raise RuntimeError("Could not encode detection preview")
        return DetectionPacket(result, jpeg.tobytes())

    def close(self) -> None:
        if isinstance(self.backend, TensorRT):
            self.backend.close()
