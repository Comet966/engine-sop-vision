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

服务使用三个单线程执行器，让视频采集和预处理、模型推理、预览绘制和 JPEG 编码并行工作。TensorRT 模型始终在同一线程创建、执行和释放；流水线仅预取有限数量的帧，检测包仍按视频帧序逐个交给跟踪与工艺事件判定，停止任务时先等待正在执行的工作结束再释放视频源。

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

周期默认读取 [composite_rules.json](composite_rules.json) 的 `cycle`。以下是文件中的周期写法。

```json
{
  "clear": {"class_ids": [1, 2], "min_confidence": 0.25},
  "start": {"after_clear_ms": 1200, "event": "cycle.a_in", "within_ms": 3000},
  "end": {
    "move": "cycle.b_moves_left",
    "out": "cycle.b_out_left",
    "within_ms": 2500,
    "clear_ms": 1200
  }
}
```

`clear` 定义参与清空判断的类别和置信度。`start.event` 引用 `In`，等待时间从清空后第一次检出工件算起。`end.move` 和 `end.out` 分别引用 `Move`、`Out`，程序要求同一条轨迹，时间从 `Move` 确认算到 `Out` 首次未检出。确认 `Out` 的延迟与清空计时可以重叠。已有请求中的 `cycle_rules` 仍可按字段覆盖周期参数。

`cycle_exit_x` 以画面宽度为单位。《瓦片安装2.mp4》默认 `0.16`，左侧软管遮挡处为有效出口。《瓦片安装3.mp4》和其他源默认 `0.03`。周期依赖的三条基础规则沿用同一个跟踪器，默认发出 `cycle.a_in`、`cycle.b_moves_left`、`cycle.b_out_left`。

- `start_ms`、`start_frame_id` 记录触发开头的 A 首次可见的位置，`a_track_id` 是对应轨迹。
- `end_ms`、`end_frame_id` 记录 B 离场后第一次未检出的位置。结束事件同时携带 `b_track_id`、`b_move_confirmed_ms` 和最后一帧 B 的位置。
- `confirmed_ms`、`confirmed_frame_id` 记录当前阶段确认的位置。`empty_ms` 是开始前或结束后的连续空窗。
- 摄像开始时工件已在画面内，后续若确认离场，会发出 `partial: true` 的结束事件，`start_ms` 和 `start_frame_id` 留空。完整周期的开始、结束使用同一 `cycle_id`。停止或录像读完时，已开始但缺少结束证据的周期保持开放。断流或跳帧会停止关联此前的周期证据。
- `GET /api/runs/{id}` 可查 `active_cycle_id` 和完整周期数 `completed_cycles`。重连时可按 `cycle_id` 和 `phase` 去重。

## Move、Close、In、Out

创建任务时可以传入 `primitive_rules`，其结构见 [默认规则](primitive_rules.json)。默认规则覆盖四类目标各自的 `Move`、`In`、`Out`，以及两两 `Close` 和握持所需的重叠检测。周期所需的三条基础规则由服务生成。传 `null` 可关闭基础元事件的推送，周期仍使用内部证据。规则的目标用别名引用 `targets`，事件 `id` 在配置内唯一。实际检测同时受规则的 `min_confidence` 和任务的 `confidence` 限制。

| 元事件 | 判定和可调参数 |
| --- | --- |
| `Move` | 平滑框中心在 `window_ms` 内达到 `min_displacement_ratio` 后触发；稳定后重新计数。`direction` 可设 `angle_deg` 和 `tolerance_deg`，右、下、左、上分别为 0、90、180、270 度。默认允许任意方向。 |
| `Close` | `distance_metric` 可选 `bbox_gap`（默认，两框最短间距）、`center_distance`（两框中心距离）或兼容旧配置的 `bottom_edge_gap`（第一个框的下边线到第二个框的最短间距）。使用 `bbox_gap` 时，`edge`、`target_edge` 可分别选择第一个和第二个目标框的 `top`、`bottom`、`left` 或 `right` 边线；省略对应字段即使用整个框。可选 `relative_position` 进一步限定相对位置，越出位置范围立即复位。距离进入 `enter_distance_ratio` 后累计 `hold_ms`，超过 `exit_distance_ratio` 复位。 |
| `In` | 一条新轨迹累计可见 `confirm_ms` 后触发。视频开头已经可见的目标也可确认 `In`。 |
| `Out` | 最近 `exit_window_ms` 内中心朝边缘移动至少 `min_displacement_ratio`，最后一个框距边缘不超过 `edge_distance_ratio`，随后连续未见满 `missing_ms` 才触发。可用 `edge` 限定方向，`left_edge_x` 设置左侧有效出口。 |

距离与位移的 `*_ratio` 以画面对角线为单位。`max_gap_ms` 允许短暂掉框，`In` 和 `Close` 在缺帧的区间暂停计时。流中断会重置轨迹和计时。画面内遮挡可造成轨迹暂时不可见。长时间遮挡后重新检出时可能分配新 ID，`In` 会按新轨迹确认。工位特定的软管遮挡出口仍由 `sop.workpiece_cycle` 的 `cycle_exit_x` 处理。

### 检测框尺寸

可在目标别名下配置 `bbox_size`，根据框在画面里的投影大小筛选证据。`metric` 支持 `height_ratio`（框高 ÷ 画面高，默认）、`width_ratio`（框宽 ÷ 画面宽）和 `area_ratio`（框面积 ÷ 画面面积）。`Close` 还可配置 `bbox_height_ratio`，计算 `targets[0]` 的框高 ÷ `targets[1]` 的框高。两种设置均可只填 `min` 或只填 `max`，同时填写时边界包含在内。

`Move`、`In` 和 `Close` 只使用尺寸达标的框积累证据。`Close` 的框高比超出范围时，当前帧的关系会从 `active_close` 移除。`Out` 只在配置的离场轨迹窗口内检查是否至少有一帧符合目标尺寸。边缘裁切导致框变小后，符合置信度要求的框仍计作可见，消失计时从第一帧未见开始。

### 相对位置

`Close.relative_position` 的 `reference` 使用第一个目标别名，或在定义 `union_with` 时使用 `union` 表示联合框。`anchor` 可选 `center`（默认）与 `top`、`bottom`、`left`、`right`（对应边线的中点）。`axis` 为 `x` 或 `y`，将参考点所选坐标减去第二个目标框的对应起始坐标，再除以该框在该轴上的长度。`min`、`max` 至少设置一个，边界包含在内，允许负值和大于 1 的值以覆盖框外位置。位置条件与距离、置信度和尺寸条件同时满足才累计证据。

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

## 组合规则 JSON

省略任务请求中的 `composite_rules` 时，服务读取 [composite_rules.json](composite_rules.json)。文件根字段为 `schema_version`、`cycle` 和 `events`。在其中的 `events` 增加组合规则，或把完整文件内容作为 `POST /api/runs` 的 `composite_rules`。基础元事件 ID 来自 [primitive_rules.json](primitive_rules.json) 及周期内置元事件。服务在创建任务前校验引用、重复 ID 和轨迹绑定。

| 字段 | 写法 |
| --- | --- |
| `id` | 组合事件唯一 ID，供订阅及后续规则引用。 |
| `op: "event"` | 用 `event` 引用单个基础事件或已定义的组合事件，为它定义一个可复用的名称。以 `Close.started` 为接近确认。 |
| `op: "sequence"` | 用 `steps` 写顺序，用 `within_ms` 限制第一步到最后一步的确认时间。相邻步骤须在不同帧按顺序确认。可引用基础事件或数组中已定义的组合事件。 |
| `op: "parallel"` | 用 `all` 写主要的 `Close` 证据，用可选的 `when` 写附加条件。用 `min_overlap_ms` 设全部证据的连续重叠时长。 |
| `when` | 并行规则的条件列表。每项用 `{"event": "基础 Close 规则 ID"}` 引用当帧已确认且有效的 `Close`，支持 `bind`。列表内条件均须满足，轨迹绑定和时间判据与 `all` 一起校验。 |
| `same_tracks` | 填写跨步骤或分支共享的目标别名，要求相同的轨迹 ID。若各事件的别名不同，在对应引用里用 `bind` 映射为同一个别名。 |
| `within_cycle` | 默认为 `true`，仅在同一已开始的周期内匹配。 |

默认配置独立上报三种通用动作和四种位置相关的除尘动作。只传自定义 `primitive_rules` 时，周期仍读取默认配置。需要启用命名动作时，可同时提交引用自定义基础事件的 `composite_rules`。

| 事件 ID | 确认条件 |
| --- | --- |
| `hand_holds_tube` | 手套框与吸管框的交集面积达到较小框面积的 30%，同一对轨迹累计获得 300 毫秒有效证据。用二维重叠推断握持。 |
| `tube_near_a` | 吸管与 A 位满足已调优的框间距和 A 位尺寸规则，累计 300 毫秒有效证据。 |
| `tube_near_b` | 吸管与 B 位满足已调优的框间距和 B 位尺寸规则，累计 300 毫秒有效证据。 |

握持基础规则的 `min_overlap_ratio` 可按机位调整。三个动作均在对应条件开始成立时确认，后续规则可引用各自的 ID。

`held_tube_near_a_left`、`held_tube_near_a_right`、`held_tube_near_b_top` 和 `held_tube_near_b_bottom` 要求握持与对应工件位置接近的证据连续重叠 60 毫秒，且两项证据对应同一对手套和吸管轨迹。这四个规则替代原来的 A、B 整框除尘事件。握持按手套框与吸管框的中心距离判断，进入和退出阈值分别为画面对角线的 12% 和 14%。框交集至少占较小框面积的 25%（A 位组合）或 30%（单独握持和 B 位组合）。

靠近规则使用 `"union_with": "glove"`、`"distance_metric": "bbox_gap"` 和 `"edge": "bottom"`，测量手套与吸管联合框下边线到工件完整框的最短间距。再用 `relative_position` 区分位置：A 左、右分别要求吸管框中心的相对 X ≤ 0.30 和 ≥ 0.40，累计 150 毫秒；B 较高、较低位置分别要求联合框下边线的相对 Y ≤ 0.08 和 ≥ 0.11，累计 60 和 100 毫秒。A 的距离进入/退出阈值为 0.015/0.025、框高比上限 3.25；B 的距离阈值为 0.025/0.035。B 的两个位置为当前机位标定的投影位置，需要结合实际工艺验证其物理含义。不同轨迹的匹配可能分别满足不同位置条件。

靠近独立累计证据。组合规则用 `all` 引用靠近规则，用 `when` 引用握持规则，用 `"same_tracks": ["tube", "glove"]` 确保两者使用同一对轨迹。靠近证据可以在握持确认前开始累计，吸工件事件须等握持确认且满足重叠时长才触发。

下面是并行与顺序的语法示例。三个占位 ID 应替换为实际定义的规则 ID。

```json
{
  "events": [
    {
      "id": "parallel_example",
      "op": "parallel",
      "all": [{"event": "close_first"}, {"event": "close_second"}],
      "same_tracks": ["target"],
      "min_overlap_ms": 120
    },
    {
      "id": "sequence_example",
      "op": "sequence",
      "steps": [{"event": "parallel_example"}, {"event": "next_event"}],
      "same_tracks": ["target"],
      "within_ms": 2000
    }
  ]
}
```

以上各引用假定均含有 `target` 别名。顺序匹配使用基础事件的确认消息，以 `Close.started` 作为接近步骤的确认。并行取每帧 `active_close` 的连续交集，任一分支失去连续证据时重新计时。已有 `primitive_rules.parallel` 配置仍可读取，新规则统一写入 `composite_rules.events`。

组合消息类型是 `sop.composite_event`。消息包含 `op`、`rule_id`、`cycle_id`、`subjects`、`start_ms`、确认位置和各步骤的 `evidence_ids`。`start_ms` 是重叠起点或第一步的确认时间。重连时可按 `occurrence_id` 去重。

```bash
python -m unittest sop_backend.test_primitive_events sop_backend.test_composition
```
