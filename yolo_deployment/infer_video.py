"""Run the four-class SOP detector on Jetson or on ONNX Runtime CPU."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

SIZE = 640
CLASSES = ("Glove", "Seat A", "Seat B", "Suction tube")
COLORS = ((225, 210, 40), (55, 210, 255), (210, 105, 255), (105, 245, 120))


class TensorRT:
    def __init__(self, path):
        import tensorrt as trt

        self.cuda = ct.CDLL("libcudart.so.12")
        self.cuda.cudaMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t]
        self.cuda.cudaMemcpyAsync.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_void_p]
        self.cuda.cudaStreamCreateWithFlags.argtypes = [ct.POINTER(ct.c_void_p), ct.c_uint]
        self.cuda.cudaStreamSynchronize.argtypes = [ct.c_void_p]
        self.cuda.cudaStreamDestroy.argtypes = [ct.c_void_p]
        self.cuda.cudaFree.argtypes = [ct.c_void_p]
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not load the engine")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        input_name, = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output_name, = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        assert tuple(self.engine.get_tensor_shape(input_name)) == (1, 3, SIZE, SIZE)
        assert trt.nptype(self.engine.get_tensor_dtype(input_name)) == np.float32
        self.output = np.empty(
            tuple(self.engine.get_tensor_shape(output_name)),
            dtype=trt.nptype(self.engine.get_tensor_dtype(output_name)),
        )
        self.pointers = []
        self.stream = ct.c_void_p()
        self.check(self.cuda.cudaStreamCreateWithFlags(ct.byref(self.stream), 1))
        try:
            self.input_ptr = self.allocate(input_name, 3 * SIZE * SIZE * 4)
            self.output_ptr = self.allocate(output_name, self.output.nbytes)
        except BaseException:
            self.close()
            raise

    def check(self, code):
        if code != 0:
            raise RuntimeError(f"CUDA runtime error {code}")

    def allocate(self, name, size):
        pointer = ct.c_void_p()
        self.check(self.cuda.cudaMalloc(ct.byref(pointer), size))
        self.pointers.append(pointer)
        if not self.context.set_tensor_address(name, int(pointer.value)):
            raise RuntimeError(f"TensorRT could not bind {name}")
        return pointer

    def __call__(self, data):
        self.check(self.cuda.cudaMemcpyAsync(self.input_ptr, data.ctypes.data, data.nbytes, 1, self.stream))
        if not self.context.execute_async_v3(stream_handle=self.stream.value):
            raise RuntimeError("TensorRT execution failed")
        self.check(self.cuda.cudaMemcpyAsync(self.output.ctypes.data, self.output_ptr, self.output.nbytes, 2, self.stream))
        self.check(self.cuda.cudaStreamSynchronize(self.stream))
        return self.output

    def close(self):
        for pointer in self.pointers:
            self.cuda.cudaFree(pointer)
        self.cuda.cudaStreamDestroy(self.stream)


def prepare(frame):
    height, width = frame.shape[:2]
    gain = min(SIZE / width, SIZE / height)
    new_w, new_h = round(width * gain), round(height * gain)
    left = (SIZE - new_w) // 2
    top = (SIZE - new_h) // 2
    letterbox = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    letterbox[top:top + new_h, left:left + new_w] = cv2.resize(frame, (new_w, new_h))
    tensor = cv2.dnn.blobFromImage(letterbox, scalefactor=1.0 / 255.0, swapRB=True)
    return tensor, gain, left, top


def load_backend(path):
    if path.suffix.lower() == ".engine":
        return TensorRT(path)

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def infer(tensor):
        return session.run(None, {input_name: tensor})[0]

    return infer


def decode(raw, coords, shape, threshold):
    if raw.shape != (1, 8, 8400):
        raise ValueError(f"Unexpected YOLO output shape {raw.shape}")
    rows = raw[0].T
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), classes + 4]
    keep = scores >= threshold
    boxes = rows[keep, :4].copy()
    scores, classes = scores[keep], classes[keep]
    boxes[:, :2] -= boxes[:, 2:] / 2
    gain, left, top = coords
    height, width = shape[:2]
    detections = []
    for category in range(len(CLASSES)):
        indices = np.flatnonzero(classes == category)
        kept = cv2.dnn.NMSBoxes(boxes[indices].tolist(), scores[indices].tolist(), threshold, 0.7)
        for local in np.asarray(kept).reshape(-1):
            idx = indices[int(local)]
            x, y, w, h = boxes[idx]
            bounds = [
                np.clip((x - left) / gain, 0, width),
                np.clip((y - top) / gain, 0, height),
                np.clip((x + w - left) / gain, 0, width),
                np.clip((y + h - top) / gain, 0, height),
            ]
            detections.append((category, float(scores[idx]), [round(float(v)) for v in bounds]))
    return sorted(detections, key=lambda item: item[1], reverse=True)[:300]


def draw_detections(frame, detections):
    for category, score, (x1, y1, x2, y2) in reversed(detections):
        color = COLORS[category]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[category]} {score:.2f}"
        (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)
        x = min(x1, frame.shape[1] - w - 6)
        y = max(y1, 20)
        cv2.rectangle(frame, (x, y - 20), (x + w + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (24, 24, 24), 1, cv2.LINE_AA)


def annotate(frame, detections, index, fps):
    draw_detections(frame, detections)
    tube = next((score for category, score, _ in detections if category == 3), None)
    status = f"{index / fps:06.2f}s  Tube " + (f"{tube:.2f}" if tube is not None else "not detected")
    (w, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    x = frame.shape[1] - w - 16
    cv2.rectangle(frame, (x - 5, 6), (frame.shape[1] - 6, 29), (24, 24, 24), -1)
    cv2.putText(frame, status, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (238, 238, 238), 1, cv2.LINE_AA)


def write_annotated(writer, frame, detections, index, fps):
    annotate(frame, detections, index, fps)
    writer.write(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--sample-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    gpu = args.model.suffix.lower() == ".engine"
    backend = load_backend(args.model)

    capture = cv2.VideoCapture(str(args.video))
    writer = None
    encoder = None
    pending = deque()
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {args.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps > 0:
            raise RuntimeError("Source FPS is unavailable")
        if args.sample_seconds is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(args.sample_seconds * fps))
        if args.output:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not write {args.output}")
            encoder = ThreadPoolExecutor(max_workers=1)

        warmup = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
        backend(warmup)
        times = []
        class_frames = [0] * len(CLASSES)
        started = perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tensor, *coords = prepare(frame)
            before = perf_counter()
            raw = backend(tensor)
            times.append((perf_counter() - before) * 1000)
            detections = decode(raw, coords, frame.shape, args.threshold)
            for category in {item[0] for item in detections}:
                class_frames[category] += 1
            if encoder:
                if len(pending) >= 4:
                    pending.popleft().result()
                pending.append(encoder.submit(write_annotated, writer, frame, detections, len(times) - 1, fps))
            if args.sample_seconds is not None:
                print(json.dumps({"detections": detections, "raw_shape": list(raw.shape)}))
                break
            if len(times) == args.max_frames:
                break
            if len(times) % 600 == 0:
                print(f"Processed {len(times)} frames", flush=True)
        if encoder:
            for item in pending:
                item.result()
            encoder.shutdown()
            encoder = None
            writer.release()
            writer = None
        elapsed = perf_counter() - started
        if not times:
            raise RuntimeError("No frames were decoded")
        print(json.dumps({
            "backend": "TensorRT" if gpu else "ONNX Runtime CPU",
            "frames": len(times), "source_fps": fps,
            "pipeline_sec": round(elapsed, 3), "pipeline_fps": round(len(times) / elapsed, 2),
            "inference_p50_ms": round(float(np.percentile(times, 50)), 2),
            "inference_p95_ms": round(float(np.percentile(times, 95)), 2),
            "class_detected_frames": dict(zip(CLASSES, class_frames)),
            "confidence_threshold": args.threshold,
        }))
    finally:
        capture.release()
        if encoder:
            encoder.shutdown()
        if writer:
            writer.release()
        if gpu:
            backend.close()


if __name__ == "__main__":
    main()
