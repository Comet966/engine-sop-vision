from unittest import TestCase

from pydantic import ValidationError

from .event_rules import BoxSize, PrimitiveRules, default_primitive_rules
from .models import Detection, DetectionFrame
from .primitive_events import PrimitiveEventDetector, matches_box_size


def frame(index, *boxes):
    detections = []
    for category, x, y, *size in boxes:
        width, height = size or (36, 36)
        detections.append(
            Detection(class_id=category, label=str(category), confidence=0.9, bbox_xyxy=(x, y, x + width, y + height))
        )
    return DetectionFrame(
        run_id="test", frame_id=index, timestamp_ms=index * 40, width=1000, height=600,
        detections=detections,
    )


class PrimitiveEventTests(TestCase):
    def test_default_rules_cover_four_classes_and_all_six_pairs(self):
        config = default_primitive_rules()
        engine = PrimitiveEventDetector(config)
        events = []
        for index in range(35):
            boxes = []
            if index < 20:
                x = max(7, 124 - max(0, index - 10) * 13)
                sizes = [(36, 150), (36, 90), (36, 90), (130, 216)]
                boxes = [(category, x, (200, 350, 355, 215)[category], *size) for category, size in enumerate(sizes)]
            events.extend(engine.update(frame(index, *boxes)))

        self.assertEqual(
            {e.rule_id for e in events if e.phase != "ended"}, {rule.id for rule in config.events},
        )
        self.assertEqual({e.evidence["edge"] for e in events if e.event == "Out"}, {"left"})
        for event in events:
            for ref in event.subjects.values():
                self.assertEqual(ref.track_id, ref.class_id + 1)

    def test_close_distance_uses_the_selected_geometry(self):
        rules = PrimitiveRules.model_validate({
            "targets": {"tube": {"class_id": 3}, "seat": {"class_id": 2}},
            "events": [
                {"id": metric, "type": "Close", "targets": ["tube", "seat"],
                 "distance_metric": metric, "hold_ms": 0}
                for metric in ("bbox_gap", "center_distance", "bottom_edge_gap")
            ],
        })
        tube = (3, 100, 100, 40, 160)
        fixtures = [
            ((2, 100, 160, 40, 40), {"bbox_gap", "center_distance"}),
            ((2, 133, 245, 40, 40), {"bbox_gap", "bottom_edge_gap"}),
        ]
        for seat, expected in fixtures:
            with self.subTest(seat=seat):
                events = PrimitiveEventDetector(rules).update(frame(0, tube, seat))
                self.assertEqual({event.rule_id for event in events}, expected)

    def test_close_union_uses_visible_boxes_without_a_grip_event(self):
        rules = PrimitiveRules.model_validate({
            "targets": {"tube": {"class_id": 3}, "seat": {"class_id": 2}, "glove": {"class_id": 0}},
            "events": [
                {"id": "suction", "type": "Close", "targets": ["tube", "seat"],
                 "distance_metric": "bottom_edge_gap", "hold_ms": 0,
                 "union_with": "glove"},
            ],
        })
        detector = PrimitiveEventDetector(rules)
        tube, glove = (3, 100, 50, 40, 150), (0, 100, 175, 40, 85)
        seat = (2, 100, 260)
        self.assertEqual(detector.update(frame(0, tube, seat)), [])
        current = frame(1, tube, glove, seat)
        suction, = detector.update(current)
        self.assertEqual(suction.evidence["distance_ratio"], 0)
        self.assertEqual(suction.subjects["glove"].track_id, current.detections[1].track_id)
        self.assertEqual(suction.subjects["tube"].track_id, current.detections[0].track_id)
        current = frame(2, tube, seat)
        detector.update(current)
        self.assertEqual(current.active_close, [])

    def test_direction_wraps_at_zero_degrees(self):
        rules = PrimitiveRules.model_validate({
            "targets": {"tube": {"class_id": 3}},
            "events": [
                {"id": "left", "type": "Move", "target": "tube",
                 "direction": {"angle_deg": 180, "tolerance_deg": 10}},
                {"id": "right", "type": "Move", "target": "tube",
                 "direction": {"angle_deg": 0, "tolerance_deg": 10}},
            ],
        })
        engine = PrimitiveEventDetector(rules)
        events = [event for index in range(10) for event in engine.update(frame(index, (3, 150 + 15 * index, 199 - index)))]
        self.assertEqual([e.rule_id for e in events], ["right"])
        self.assertGreater(events[0].evidence["angle_deg"], 350)

    def test_close_visible_intervals_pause_and_use_same_tube(self):
        config = default_primitive_rules()
        engine = PrimitiveEventDetector(config)
        together = [(0, 200, 200), (3, 225, 200, 130, 216), (1, 255, 200, 36, 90), (3, 800, 200)]
        events = []
        for index in range(9):
            current = frame(index, *together)
            events.extend(engine.update(current))
        pairs = {pair.rule_id: pair for pair in current.active_close}
        first = pairs["glove_close_tube"]
        second = pairs["tube_close_seat_a"]
        self.assertEqual(first.subjects["tube"], second.subjects["tube"])
        self.assertEqual(first.evidence_start_ms, 0)

        missing = frame(9, together[0], together[2], together[3])
        self.assertEqual([e for e in engine.update(missing) if e.event == "Close"], [])
        self.assertFalse(any("tube" in active.subjects for active in missing.active_close))
        back = frame(10, *together)
        self.assertFalse(any(e.event == "Close" and e.phase == "started" for e in engine.update(back)))
        resumed = next(pair for pair in back.active_close if pair.rule_id == first.rule_id)
        self.assertEqual(resumed.occurrence_id, first.occurrence_id)
        self.assertEqual(resumed.evidence_start_ms, 400)
        ended = next(e for e in engine.finish() if e.occurrence_id == first.occurrence_id)
        self.assertEqual((ended.phase, ended.end_ms, ended.reason), ("ended", 400, "run_finished"))

    def test_missing_midframe_and_capture_gap_do_not_emit_out(self):
        for lost_at_edge in (False, True):
            with self.subTest(lost_at_edge=lost_at_edge):
                engine = PrimitiveEventDetector(default_primitive_rules())
                for index in range(9):
                    x = (117 if lost_at_edge else 490) - 13 * index
                    engine.update(frame(index, (2, x, 280)))
                # In the edge case a jump in frame IDs means missing capture data, not observed empty frames.
                first_missing_frame = 25 if lost_at_edge else 9
                for index in range(first_missing_frame, first_missing_frame + 20):
                    self.assertFalse(any(e.event == "Out" for e in engine.update(frame(index))))

    def test_brief_missing_at_edge_keeps_track_and_cancels_out(self):
        engine = PrimitiveEventDetector(default_primitive_rules())
        for index in range(9):
            last = frame(index, (2, 117 - 13 * index, 280))
            engine.update(last)
        track_id = last.detections[0].track_id
        for index in range(9, 18):
            self.assertFalse(any(e.event == "Out" for e in engine.update(frame(index))))
        for index in range(18, 25):
            visible = frame(index, (2, 13, 280))
            self.assertFalse(any(e.event in ("Out", "In") for e in engine.update(visible)))
            self.assertEqual(visible.detections[0].track_id, track_id)

    def test_box_size_is_normalized_and_validated(self):
        for metric, scale in (("width_ratio", 0.08), ("height_ratio", 0.2), ("area_ratio", 0.016)):
            with self.subTest(metric=metric):
                size = BoxSize(metric=metric, min=0.99 * scale, max=1.01 * scale)
                self.assertTrue(matches_box_size(size, (0, 0, 80, 120), 1000, 600))
                self.assertTrue(matches_box_size(size, (0, 0, 160, 240), 2000, 1200))
                self.assertFalse(matches_box_size(size, (0, 0, 0, 0), 1000, 600))
        for bounds in ({}, {"min": 0.2, "max": 0.1}, {"min": 1.1}):
            with self.subTest(bounds=bounds), self.assertRaises(ValidationError):
                BoxSize.model_validate(bounds)

    def test_target_size_filters_appearances_and_motions_without_hiding_boxes(self):
        rules = PrimitiveRules.model_validate({
            "targets": {
                "any_tube": {"class_id": 3},
                "large_tube": {"class_id": 3, "bbox_size": {"min": 0.1}},
            },
            "events": [
                {"id": f"{target}_{kind}", "type": kind, "target": target}
                for target in ("any_tube", "large_tube") for kind in ("Move", "In")
            ],
        })
        engine = PrimitiveEventDetector(rules)
        events = []
        for index in range(16):
            current = frame(index, (3, 100 + 15 * index, 100, 40, 36), (3, 100 + 15 * index, 320, 40, 80))
            events.extend(engine.update(current))
            self.assertEqual(len(current.detections), 2)
        large = current.detections[1].track_id
        self.assertEqual({e.rule_id for e in events}, {rule.id for rule in rules.events})
        self.assertTrue(all(
            e.subjects["large_tube"].track_id == large for e in events if "large_tube" in e.subjects
        ))

    def test_close_filters_projected_size_ratio_and_interrupts_parallel_evidence(self):
        rules = PrimitiveRules.model_validate({
            "targets": {"tube": {"class_id": 3}, "seat_a": {"class_id": 1}},
            "events": [
                {"id": "near", "type": "Close", "targets": ["tube", "seat_a"], "hold_ms": 80},
                {"id": "scaled", "type": "Close", "targets": ["tube", "seat_a"], "hold_ms": 80,
                 "bbox_height_ratio": {"min": 1.5, "max": 2.5}},
            ],
        })
        engine = PrimitiveEventDetector(rules)
        for index in range(3):
            current = frame(index, (3, 100, 100, 36, 40), (1, 125, 100, 36, 40))
            engine.update(current)
        self.assertEqual([pair.rule_id for pair in current.active_close], ["near"])

        for index in range(3, 6):
            current = frame(index, (3, 100, 80, 36, 80), (1, 125, 100, 36, 40))
            events = engine.update(current)
        self.assertEqual([pair.rule_id for pair in current.active_close], ["near", "scaled"])
        hit, = events
        self.assertEqual(hit.evidence["bbox_height_ratio"], 2.0)

        smaller = frame(6, (3, 100, 100, 36, 40), (1, 125, 100, 36, 40))
        engine.update(smaller)
        self.assertEqual([pair.rule_id for pair in smaller.active_close], ["near"])

    def test_out_size_gate_allows_clipping_but_requires_real_disappearance(self):
        rules = PrimitiveRules.model_validate({
            "targets": {"part": {"class_id": 2, "bbox_size": {"metric": "width_ratio", "min": 0.03}}},
            "events": [{"id": "exits", "type": "Out", "target": "part"}],
        })
        for gone_at, should_hit in ((9, True), (24, False)):
            with self.subTest(gone_at=gone_at):
                engine = PrimitiveEventDetector(rules)
                events = []
                for index in range(gone_at + 17):
                    boxes = ()
                    if index < 6:
                        boxes = ((2, 83 - 12 * index, 100, 36, 36),)
                    elif index < gone_at:
                        boxes = ((2, max(0, 11 - 4 * (index - 6)), 100, 8, 36),)
                    events.extend(engine.update(frame(index, *boxes)))
                if should_hit:
                    hit, = events
                    self.assertEqual(hit.end_frame_id, gone_at)
                else:
                    self.assertEqual(events, [])
