from unittest import TestCase

from pydantic import ValidationError

from .composition import ParallelEventDetector, WorkpieceCycleComposer
from .event_rules import ActiveClose, CycleRules, PrimitiveEvent, PrimitiveRules, TrackedTarget
from .primitive_events import PrimitiveEventDetector
from .test_primitive_events import frame


class WorkpieceCycleTests(TestCase):
    def test_clear_then_a_then_left_exit_and_clear(self):
        composer = WorkpieceCycleComposer(CycleRules(empty_before_ms=160, empty_after_ms=800), 0.10, None)
        primitives = PrimitiveEventDetector(composer.primitives)
        events = []
        for index in range(65):
            boxes = []
            if 5 <= index < 29:
                x = max(60, 190 - 13 * max(0, index - 18))
                boxes.append((2, x, 300))
            if 8 <= index < 24:
                boxes.append((1, 200, 200))
            current = frame(index, *boxes)
            event = composer.update(current, primitives.update(current))
            if event:
                events.append(event)

        start, end = events
        self.assertEqual((start.phase, start.start_frame_id, start.empty_ms), ("started", 8, 200))
        self.assertEqual((end.phase, end.end_frame_id, end.confirmed_frame_id), ("ended", 29, 49))
        self.assertEqual((end.cycle_id, end.empty_ms), (start.cycle_id, 800))
        self.assertEqual(composer.completed, 1)

    def test_partial_end_requires_same_b_and_is_canceled_when_b_reappears(self):
        def primitive(composer, name, track, index, first_missing=None):
            return PrimitiveEvent(
                event=name, phase="detected", run_id="test", sequence=index, rule_id=(
                    composer.out_rule if name == "Out" else composer.move_rule
                ),
                occurrence_id=f"event:{index}", subjects={composer.seat_b: TrackedTarget(track_id=track, class_id=2)},
                start_ms=index * 40, start_frame_id=index, confirmed_ms=index * 40, confirmed_frame_id=index,
                end_ms=first_missing * 40 if first_missing is not None else None,
                end_frame_id=first_missing,
                evidence={"last_seen_ms": 80, "last_seen_frame_id": 2},
            )

        for out_track, reappears, expected in ((7, False, 1), (8, False, 0), (7, True, 0)):
            with self.subTest(track=out_track, reappears=reappears):
                composer = WorkpieceCycleComposer(CycleRules(empty_after_ms=800), 0.10, None)
                results = []
                for index in range(30):
                    present = index <= 2 or (reappears and index == 17)
                    current = frame(index, (2, 90, 300)) if present else frame(index)
                    events = []
                    if index == 2:
                        events.append(primitive(composer, "Move", 7, index))
                    if index == 16:
                        events.append(primitive(composer, "Out", out_track, index, first_missing=3))
                    result = composer.update(current, events)
                    if result:
                        results.append(result)
                self.assertEqual(len(results), expected)
                if results:
                    self.assertTrue(results[0].partial)
                    self.assertIsNone(results[0].start_ms)
                    self.assertEqual(results[0].confirmed_frame_id, 23)
                self.assertEqual(composer.completed, 0)

    def test_skipped_frames_do_not_supply_clear_evidence(self):
        composer = WorkpieceCycleComposer(CycleRules(empty_before_ms=160), 0.10, None)
        primitives = PrimitiveEventDetector(composer.primitives)
        for index in (0, 1, *range(15, 30)):
            current = frame(index, (1, 200, 200)) if index >= 15 else frame(index)
            self.assertIsNone(composer.update(current, primitives.update(current)))


class ParallelTests(TestCase):
    def test_parallel_requires_shared_track_and_restarts_after_evidence_gap(self):
        rules = PrimitiveRules.model_validate({
            "targets": {"glove": {"class_id": 0}, "tube": {"class_id": 3},
                        "sized_tube": {"class_id": 3}, "seat_a": {"class_id": 1}},
            "events": [
                {"id": "hold", "type": "Close", "targets": ["glove", "tube"]},
                {"id": "near", "type": "Close", "targets": ["sized_tube", "seat_a"]},
            ],
            "parallel": [{
                "id": "hold_near", "op": "parallel",
                "all": [{"event": "hold"}, {"event": "near", "bind": {"sized_tube": "tube"}}],
                "same_tracks": ["tube"], "min_overlap_ms": 120, "within_cycle": False,
            }],
        })
        detector = ParallelEventDetector(rules.parallel)

        def observed(index, other_tube, start, *, include_near=True):
            current = frame(index)
            current.active_close = [ActiveClose(
                rule_id="hold", occurrence_id="hold:1", evidence_start_ms=0,
                subjects={"tube": TrackedTarget(track_id=1, class_id=3),
                          "glove": TrackedTarget(track_id=10, class_id=0)},
            )]
            if include_near:
                current.active_close.append(ActiveClose(
                    rule_id="near", occurrence_id="near:1", evidence_start_ms=start,
                    subjects={"sized_tube": TrackedTarget(track_id=other_tube, class_id=3),
                              "seat_a": TrackedTarget(track_id=11, class_id=1)},
                ))
            return detector.update(current, None)

        self.assertEqual(observed(10, 2, 0), [])
        self.assertEqual(observed(11, 1, 400), [])
        first, = observed(13, 1, 400)
        self.assertEqual((first.start_ms, first.confirmed_ms, first.subjects["tube"].track_id), (400, 520, 1))
        self.assertEqual(first.evidence_ids, ["hold:1", "near:1"])
        self.assertEqual(observed(14, 1, 400), [])
        self.assertEqual(observed(15, 1, 400, include_near=False), [])
        self.assertEqual(observed(16, 1, 640), [])
        second, = observed(19, 1, 640)
        self.assertNotEqual(first.occurrence_id, second.occurrence_id)

    def test_parallel_rejects_non_close_branch(self):
        with self.assertRaises(ValidationError):
            PrimitiveRules.model_validate({
                "targets": {"tube": {"class_id": 3}, "seat_a": {"class_id": 1}},
                "events": [
                    {"id": "in", "type": "In", "target": "tube"},
                    {"id": "near", "type": "Close", "targets": ["tube", "seat_a"]},
                ],
                "parallel": [{"id": "invalid", "all": [{"event": "in"}, {"event": "near"}]}],
            })
