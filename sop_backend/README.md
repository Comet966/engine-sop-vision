# SOP 实时检测接口

从项目根目录启动。默认使用 `yolo_deployment/best.onnx`，录像目录为 `video/`。本机可以通过 ONNX Runtime 验证接口。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r sop_backend/requirements.txt numpy opencv-python-headless onnxruntime
.venv/bin/python -m uvicorn sop_backend.app:app --host 127.0.0.1 --port 8000
```

启动后在 `http://127.0.0.1:8000/docs` 查看 REST 接口。Jetson 可在已有 OpenCV、NumPy 和 TensorRT 的 Python 环境中安装 API 依赖，设置 `SOP_MODEL_PATH` 指向现有的 `.engine` 文件。可以用 `SOP_VIDEO_DIR` 指定录像目录。服务使用一个 Uvicorn 工作进程并复用加载好的模型。

## 启动检测

```bash
curl -X POST http://127.0.0.1:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"source":{"type":"video","path":"瓦片安装3.mp4"},"confidence":0.25}'
```

本机相机使用 `{"source":{"type":"camera","index":0}}`。录像路径相对于配置的录像目录。请求可带 `sop_version` 作为工序版本标识供前端显示。录像按源时间播放，处理耗时高于帧间隔时，预览速度随实际推理速度下降。

| 接口 | 行为 |
| --- | --- |
| `POST /api/runs` | 返回 `202` 和任务状态。工位已有活动任务时返回 `409` 及该任务 ID。 |
| `GET /api/runs/{id}` | 查看状态、已处理帧数、最后一帧的时间戳、当前工件周期及完成数量。失败时提供原因。 |
| `POST /api/runs/{id}/stop` | 请求停止，先进入 `stopping`，释放视频源后进入 `stopped`。可以重复调用。 |
| `GET /api/runs/{id}/preview` | 返回 MJPEG，可直接用 `<img>` 显示。每部分带 `X-Frame-Id` 和 `X-Timestamp-Ms`。 |
| `WS /api/runs/{id}/events` | 推送任务状态、检测结果、基础元事件、工件周期和并行组合事件，进入终态后关闭。 |

任务状态为 `starting`、`running`、`stopping`、`stopped`、`completed` 或 `failed`。录像正常读完时进入 `completed`。任务和最后一帧保存在当前服务进程中。

## WebSocket 消息

`run.status` 消息的 `run` 字段与查询接口的响应结构一致。首次连接会收到当前状态，随后接收状态变化和最新检测帧。

```json
{
  "type": "detection.frame",
  "run_id": "任务 ID",
  "frame_id": 60,
  "timestamp_ms": 2000,
  "width": 1280,
  "height": 720,
  "detections": [
    {
      "class_id": 3,
      "label": "吸管",
      "confidence": 0.62,
      "bbox_xyxy": [18, 116, 205, 378],
      "track_id": 7
    }
  ],
  "active_close": []
}
```

类别 ID 依次为 `0` 手套、`1` 瓦片安装位置 A、`2` 瓦片安装位置 B、`3` 吸管。框是原图像素坐标。帧号从 0 开始，时间戳为录像媒体时间或相机打开后的相对时间。预览和 WebSocket 检测帧优先发送最新帧，消费者处理慢时帧号可以跳跃。

### 工件周期事件

`sop.workpiece_cycle` 组合基础元事件和逐帧清空观测。A 或 B 的框达到周期置信度时，就视为画面里有工件。

- **开始** · 连续 1.2 秒未检出 A、B。随后从首个工件入镜起，在 3 秒内确认第一次 `In(A)`。A 的有效可见时间满 200 毫秒时确认开始，B 可先于 A 入镜。
- **结束** · 同一条 B 轨迹先发出向左的 `Move`，再在 2.5 秒内到达左侧出口并消失。确认 `Out(B)` 和 A、B 连续未检出 1.2 秒后结束。等待清空期间 B 重新出现会取消当前结束候选。

创建任务时可设置清空时间及关联窗口。以下是 `POST /api/runs` 的请求示例。

```json
{
  "source": {"type": "video", "path": "瓦片安装2.mp4"},
  "cycle_exit_x": 0.16,
  "cycle_rules": {
    "empty_before_ms": 1200,
    "empty_after_ms": 1200,
    "a_wait_ms": 3000,
    "move_to_out_ms": 2500,
    "min_confidence": 0.25
  }
}
```

`cycle_exit_x` 以画面宽度为单位。《瓦片安装2.mp4》默认 `0.16`，左侧软管遮挡处为有效出口。《瓦片安装3.mp4》和其他源默认 `0.03`。周期依赖的三条基础规则沿用同一个跟踪器，默认发出 `cycle.a_in`、`cycle.b_moves_left`、`cycle.b_out_left`。

- `start_ms`、`start_frame_id` 记录触发开头的 A 首次可见的位置，`a_track_id` 是对应轨迹。
- `end_ms`、`end_frame_id` 记录 B 离场后第一次未检出的位置。结束事件同时携带 `b_track_id`、`b_move_confirmed_ms` 和最后一帧 B 的位置。
- `confirmed_ms`、`confirmed_frame_id` 记录当前阶段确认的位置。`empty_ms` 是开始前或结束后的连续空窗。
- 摄像开始时工件已在画面内，后续若确认离场，会发出 `partial: true` 的结束事件，`start_ms` 和 `start_frame_id` 留空。完整周期的开始、结束使用同一 `cycle_id`。停止或录像读完时，已开始但缺少结束证据的周期保持开放。断流或跳帧会停止关联此前的周期证据。
- `GET /api/runs/{id}` 可查 `active_cycle_id` 和完整周期数 `completed_cycles`。重连时可按 `cycle_id` 和 `phase` 去重。

## Move、Close、In、Out

创建任务时可以传入 `primitive_rules`，其结构见 [默认规则](primitive_rules.json)。省略时加载默认的 18 条规则，覆盖四类目标各自的 `Move`、`In`、`Out`，以及六种两两 `Close`。周期所需的三条基础规则由服务生成。传 `null` 可关闭基础元事件的推送，周期仍使用内部证据。规则的目标用别名引用 `targets`，事件 `id` 在配置内唯一。实际检测同时受规则的 `min_confidence` 和任务的 `confidence` 限制。

| 元事件 | 判定和可调参数 |
| --- | --- |
| `Move` | 平滑框中心在 `window_ms` 内达到 `min_displacement_ratio` 后触发；稳定后重新计数。`direction` 可设 `angle_deg` 和 `tolerance_deg`，右、下、左、上分别为 0、90、180、270 度。默认允许任意方向。 |
| `Close` | 两框最短间距进入 `enter_distance_ratio` 后累计 `hold_ms`，超过 `exit_distance_ratio` 复位。两框重叠时间距为 0。每对轨迹独立计算。 |
| `In` | 一条新轨迹累计可见 `confirm_ms` 后触发。视频开头已经可见的目标也可确认 `In`。 |
| `Out` | 最近 `exit_window_ms` 内中心朝边缘移动至少 `min_displacement_ratio`，最后一个框距边缘不超过 `edge_distance_ratio`，随后连续未见满 `missing_ms` 才触发。可用 `edge` 限定方向，`left_edge_x` 设置左侧有效出口。 |

距离与位移的 `*_ratio` 以画面对角线为单位。`max_gap_ms` 允许短暂掉框，`In` 和 `Close` 在缺帧的区间暂停计时。流中断会重置轨迹和计时。画面内遮挡可造成轨迹暂时不可见。长时间遮挡后重新检出时可能分配新 ID，`In` 会按新轨迹确认。工位特定的软管遮挡出口仍由 `sop.workpiece_cycle` 的 `cycle_exit_x` 处理。

### 检测框尺寸

可在目标别名下配置 `bbox_size`，根据框在画面里的投影大小筛选证据。`metric` 支持 `height_ratio`（框高 ÷ 画面高，默认）、`width_ratio`（框宽 ÷ 画面宽）和 `area_ratio`（框面积 ÷ 画面面积）。`Close` 还可配置 `bbox_height_ratio`，计算 `targets[0]` 的框高 ÷ `targets[1]` 的框高。两种设置均可只填 `min` 或只填 `max`，同时填写时边界包含在内。

`Move`、`In` 和 `Close` 只使用尺寸达标的框积累证据。`Close` 的框高比超出范围时，当前帧的关系会从 `active_close` 移除。`Out` 只在配置的离场轨迹窗口内检查是否至少有一帧符合目标尺寸。边缘裁切导致框变小后，符合置信度要求的框仍计作可见，消失计时从第一帧未见开始。

默认的吸管靠近 A、B 使用已复核的尺寸参数。单个 `Close` 可通过 `bbox_sizes` 覆盖对应别名的尺寸约束，公共目标仍可使用同一个别名参与并行组合。下面是只启用 A 位接近规则的 `primitive_rules` 示例。

```json
{
  "schema_version": 1,
  "targets": {
    "tube": {"class_id": 3},
    "seat_a": {"class_id": 1}
  },
  "events": [
    {
      "id": "tube_close_seat_a",
      "type": "Close",
      "targets": ["tube", "seat_a"],
      "enter_distance_ratio": 0.015,
      "exit_distance_ratio": 0.025,
      "hold_ms": 300,
      "bbox_sizes": {"tube": {"metric": "width_ratio", "min": 0.112, "max": 0.22}},
      "bbox_height_ratio": {"min": 1.7, "max": 3.1}
    }
  ]
}
```

启用比例条件时，`Close` 的 `started` 消息在 `evidence.bbox_height_ratio` 给出确认帧的实测值。B 位吸管框高占画面高度 34%–62%，吸管与 B 的框高比为 1.70–3.60。物体朝向、裁切和遮挡也会影响投影尺寸，阈值应根据同机位的实际动作回放确定。

### 事件消息

实时接口在同一个 WebSocket 发送 `type="sop.primitive_event"`。Python 中用 `PrimitiveEventDetector.update(frame)` 消费同一批 YOLO 框，每帧调用一次，包括空检测帧。

- `event` 为 `Move`、`Close`、`In` 或 `Out`，`rule_id` 对应配置的 `id`。`subjects` 从目标别名映射到 `track_id` 和 `class_id`，轨迹 ID 在单次任务内有效。
- `start_ms`、`end_ms` 和对应帧号记录证据区间，`confirmed_ms` 和对应帧号记录确认时刻。`Out` 的 `end_ms` 是第一次未见的时刻，稍后才确认。
- `Move`、`In`、`Out` 的 `phase` 为 `detected`。`Close` 发送 `started` 和 `ended`，共享 `occurrence_id`。`ended` 附带 `reason`，结束时间为最后一帧双方同时可见且满足距离条件的时刻。`unobserved` 表示证据中断。
- 重连时会重发该任务最近 4096 条元事件消息，可用 `run_id` 和 `sequence` 去重。

### 并行组合

在 `primitive_rules.parallel` 加入并行规则。`all` 引用需要同时成立的 `Close` 规则，`same_tracks` 要求分支中相同的别名共享轨迹 ID。`within_cycle` 默认为 `true`，只在已确认开始的周期内执行。下面的两条规则分别表示手套靠近吸管时，同一根吸管同时靠近 A 或 B。

```json
{
  "parallel": [
    {
      "id": "tube_held_near_a",
      "op": "parallel",
      "all": [{"event": "glove_close_tube"}, {"event": "tube_close_seat_a"}],
      "same_tracks": ["tube"],
      "min_overlap_ms": 120
    },
    {
      "id": "tube_held_near_b",
      "op": "parallel",
      "all": [{"event": "glove_close_tube"}, {"event": "tube_close_seat_b"}],
      "same_tracks": ["tube"],
      "min_overlap_ms": 120
    }
  ]
}
```

同一事件要求三个条件时，可把三个分支放进一个 `all`。分支原来使用不同别名时，可在分支中设置 `"bind": {"tube_for_a": "tube"}` 统一引用同一根吸管。组合按每帧的 `active_close` 取连续证据交集。最晚的连续证据起点到当前帧满 `min_overlap_ms` 后，同一组轨迹触发一次。任一目标掉框会中断连续证据，恢复后重新计时。`Close` 表示框间接近，吸附结果仍需工艺证据。

组合消息类型是 `sop.composite_event`。消息包含 `rule_id`、`cycle_id`、`subjects`、重叠起点 `start_ms`、确认位置和各分支的 `evidence_ids`。重连时可按 `occurrence_id` 去重。

```bash
python -m unittest sop_backend.test_primitive_events sop_backend.test_composition
```
