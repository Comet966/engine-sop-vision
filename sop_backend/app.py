import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .models import CameraSource, RunRequest, RunStatus
from .runs import DetectionRun, DetectionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = "frame"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model = Path(os.environ.get("SOP_MODEL_PATH", PROJECT_ROOT / "yolo_deployment/best.onnx")).resolve()
    video_dir = Path(os.environ.get("SOP_VIDEO_DIR", PROJECT_ROOT / "video")).resolve()
    if model.suffix.lower() not in {".onnx", ".engine"} or not model.is_file():
        raise RuntimeError("SOP_MODEL_PATH must point to a readable .onnx or .engine file")
    service = DetectionService()
    try:
        await service.load(model)
        app.state.service = service
        app.state.video_dir = video_dir
        yield
    finally:
        await service.close()


app = FastAPI(title="SOP 实时检测", lifespan=lifespan)


def find_run(request: Request, run_id: str) -> DetectionRun:
    run = request.app.state.service.runs.get(run_id)
    if run is None:
        raise HTTPException(404, "检测任务不存在")
    return run


@app.post("/api/runs", status_code=202)
async def start_run(payload: RunRequest, request: Request) -> RunStatus:
    service: DetectionService = request.app.state.service
    if service.active is not None:
        raise HTTPException(409, detail={"message": "已有检测任务运行中", "run_id": service.active.id})

    if isinstance(payload.source, CameraSource):
        source = payload.source.index
    else:
        relative_path = Path(payload.source.path)
        root = request.app.state.video_dir
        try:
            source = (root / relative_path).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(422, "视频路径无效") from exc
        if relative_path.is_absolute() or not source.is_relative_to(root):
            raise HTTPException(422, "视频路径必须在配置的录像目录内")
        if not source.is_file():
            raise HTTPException(404, "视频文件不存在")
    return service.start(payload, source).snapshot()


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> RunStatus:
    return find_run(request, run_id).snapshot()


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str, request: Request) -> RunStatus:
    return find_run(request, run_id).request_stop()


async def preview_stream(run: DetectionRun) -> AsyncIterator[bytes]:
    last_frame_id = None
    async for _, packet in run.updates():
        if packet is None or packet.result.frame_id == last_frame_id:
            continue
        last_frame_id = packet.result.frame_id
        header = (
            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(packet.jpeg)}\r\n"
            f"X-Frame-Id: {last_frame_id}\r\nX-Timestamp-Ms: {packet.result.timestamp_ms}\r\n\r\n"
        )
        yield header.encode("ascii") + packet.jpeg + b"\r\n"
    yield f"--{BOUNDARY}--\r\n".encode("ascii")


@app.get(
    "/api/runs/{run_id}/preview",
    response_class=StreamingResponse,
    responses={200: {"content": {"multipart/x-mixed-replace": {}}}},
)
async def preview_run(run_id: str, request: Request) -> StreamingResponse:
    return StreamingResponse(
        preview_stream(find_run(request, run_id)),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


async def send_events(websocket: WebSocket, run: DetectionRun) -> None:
    last_state = None
    last_frame_id = None
    composed_cursor = 0
    primitive_cursor = 0
    async for state, packet in run.updates():
        composed_events = run.composed_events[composed_cursor:]
        primitive_events = run.primitive_events_after(primitive_cursor)
        if state.status != last_state:
            await websocket.send_json({"type": "run.status", "run": state.model_dump(mode="json")})
            last_state = state.status
        if packet is not None and packet.result.frame_id != last_frame_id:
            await websocket.send_json(packet.result.model_dump(mode="json"))
            last_frame_id = packet.result.frame_id
        for event in primitive_events:
            await websocket.send_json(event.model_dump(mode="json"))
            primitive_cursor = event.sequence
        for event in composed_events:
            await websocket.send_json(event.model_dump(mode="json"))
        composed_cursor += len(composed_events)
    await websocket.close()


async def wait_for_disconnect(websocket: WebSocket) -> None:
    # Consume disconnect even if the detector stops producing updates.
    while (await websocket.receive())["type"] != "websocket.disconnect":
        pass


@app.websocket("/api/runs/{run_id}/events")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    run = websocket.app.state.service.runs.get(run_id)
    await websocket.accept()
    if run is None:
        await websocket.close(code=1008, reason="Unknown run")
        return
    sender = asyncio.create_task(send_events(websocket, run))
    receiver = asyncio.create_task(wait_for_disconnect(websocket))
    try:
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        receiver.cancel()
        await asyncio.gather(sender, receiver, return_exceptions=True)
