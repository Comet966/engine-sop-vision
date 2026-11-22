import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .composition import ParallelEventDetector, WorkpieceCycleComposer
from .detection import DetectionPacket, Detector, FrameReader, SourceError
from .event_rules import ParallelEvent, PrimitiveEvent
from .models import RunRequest, RunState, RunStatus, TERMINAL_STATES, WorkpieceCycleEvent
from .primitive_events import PrimitiveEventDetector


logger = logging.getLogger(__name__)
PRIMITIVE_REPLAY_LIMIT = 4096


class DetectionRun:
    def __init__(self, request: RunRequest, cycle_exit_x: float) -> None:
        self.id = uuid4().hex
        self.request = request
        self.status: RunState = "starting"
        self.created_at = datetime.now(timezone.utc)
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.error: str | None = None
        self.latest: DetectionPacket | None = None
        self.cycle = WorkpieceCycleComposer(request.cycle_rules, cycle_exit_x, request.primitive_rules)
        self.parallel = ParallelEventDetector(self.cycle.primitives.parallel)
        self.composed_events: list[WorkpieceCycleEvent | ParallelEvent] = []
        self.primitives = PrimitiveEventDetector(self.cycle.primitives)
        self.primitive_events: deque[PrimitiveEvent] = deque(maxlen=PRIMITIVE_REPLAY_LIMIT)
        self.stop = asyncio.Event()
        self.changed = asyncio.Event()

    def snapshot(self) -> RunStatus:
        result = self.latest.result if self.latest else None
        return RunStatus(
            id=self.id,
            status=self.status,
            source=self.request.source,
            sop_version=self.request.sop_version,
            confidence=self.request.confidence,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            frames_processed=result.frame_id + 1 if result else 0,
            last_frame_id=result.frame_id if result else None,
            timestamp_ms=result.timestamp_ms if result else None,
            error=self.error,
            cycle_exit_x=self.cycle.exit_x,
            active_cycle_id=self.cycle.active.cycle_id if self.cycle.active else None,
            completed_cycles=self.cycle.completed,
        )

    def _notify(self) -> None:
        previous = self.changed
        self.changed = asyncio.Event()
        previous.set()

    def transition(self, status: RunState, error: str | None = None) -> None:
        self.status = status
        self.error = error
        if status == "running":
            self.started_at = datetime.now(timezone.utc)
        if status in TERMINAL_STATES:
            events = self.primitives.finish()
            if self.request.primitive_rules is not None:
                self.primitive_events.extend(events)
            self.finished_at = datetime.now(timezone.utc)
        self._notify()

    def publish(self, packet: DetectionPacket) -> None:
        self.latest = packet
        primitives = self.primitives.update(packet.result)
        if self.request.primitive_rules is not None:
            self.primitive_events.extend(primitives)
        event = self.cycle.update(packet.result, primitives)
        if event is not None:
            self.composed_events.append(event)
        self.composed_events.extend(self.parallel.update(packet.result, self.cycle.active))
        self._notify()

    def primitive_events_after(self, sequence: int) -> list[PrimitiveEvent]:
        events = self.primitive_events
        if not events or events[-1].sequence <= sequence:
            return []
        return [event for event in events if event.sequence > sequence]

    def request_stop(self) -> RunStatus:
        if self.status in {"starting", "running"}:
            self.stop.set()
            self.transition("stopping")
        return self.snapshot()

    async def updates(self) -> AsyncIterator[tuple[RunStatus, DetectionPacket | None]]:
        while True:
            changed, state = self.changed, self.snapshot()
            yield state, self.latest
            if state.status in TERMINAL_STATES:
                break
            await changed.wait()


class DetectionService:
    def __init__(self) -> None:
        # Keep capture and CUDA/TensorRT operations on the same thread.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sop-detector")
        self.detector: Detector | None = None
        self.runs: dict[str, DetectionRun] = {}
        self.active: DetectionRun | None = None
        self.task: asyncio.Task[None] | None = None

    async def load(self, model: Path) -> None:
        self.detector = await asyncio.get_running_loop().run_in_executor(self.executor, Detector, model)

    def start(self, request: RunRequest, source: Path | int) -> DetectionRun:
        exit_x = request.cycle_exit_x
        if exit_x is None:
            exit_x = 0.16 if isinstance(source, Path) and source.name == "瓦片安装2.mp4" else 0.03
        run = DetectionRun(request, exit_x)
        self.runs[run.id] = run
        self.active = run
        self.task = asyncio.create_task(self._run(run, source))
        return run

    async def _capture(self, run: DetectionRun, source: Path | int) -> None:
        loop = asyncio.get_running_loop()
        detector = self.detector
        assert detector is not None
        if run.stop.is_set():
            return

        reader = await loop.run_in_executor(self.executor, FrameReader, source)
        try:
            if run.stop.is_set():
                return
            run.transition("running")
            started = loop.time()
            while not run.stop.is_set():
                packet = await loop.run_in_executor(
                    self.executor, detector.read_frame, reader, run.id, run.request.confidence
                )
                if packet is None:
                    break
                if isinstance(source, Path):
                    delay = started + packet.result.timestamp_ms / 1000 - loop.time()
                    if delay > 0:
                        try:
                            await asyncio.wait_for(run.stop.wait(), timeout=delay)
                        except asyncio.TimeoutError:
                            pass
                if not run.stop.is_set():
                    run.publish(packet)
        finally:
            await loop.run_in_executor(self.executor, reader.close)

    async def _run(self, run: DetectionRun, source: Path | int) -> None:
        try:
            await self._capture(run, source)
        except SourceError as exc:
            run.transition("failed", str(exc))
        except Exception:
            logger.exception("Detection run %s failed", run.id)
            run.transition("failed", "推理或视频预览处理失败")
        else:
            run.transition("stopped" if run.stop.is_set() else "completed")
        finally:
            self.active = None

    async def close(self) -> None:
        try:
            if self.active:
                self.active.request_stop()
            if self.task:
                await self.task
            if self.detector:
                await asyncio.get_running_loop().run_in_executor(self.executor, self.detector.close)
        finally:
            self.executor.shutdown(wait=True)
