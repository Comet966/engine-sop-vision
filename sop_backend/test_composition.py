from unittest import TestCase

from pydantic import ValidationError

from .composition import CompositeEventDetector, ParallelEventDetector, WorkpieceCycleComposer
from .composite_rules import SequenceRule, default_composite_rules
from .event_rules import ActiveClose, CycleRules, PrimitiveEvent, PrimitiveRules, TrackedTarget, default_primitive_rules
from .models import RunRequest, WorkpieceCycleEvent
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


class SequenceTests(TestCase):
    def test_order_identity_and_timeout(self):
        rule = SequenceRule.model_validate({
            "id": "ordered", "op": "sequence", "within_cycle": False, "within_ms": 160,
            "steps": [{"event": "first"}, {"event": "last", "bind": {"other": "part"}}],
            "same_tracks": ["part"],
        })
        detector = CompositeEventDetector([rule])

        def event(index, name, track=1):
            alias = "part" if name == "first" else "other"
            return PrimitiveEvent(
                event="Move", phase="detected", run_id="test", sequence=index, rule_id=name,
                occurrence_id=f"{name}:{index}", subjects={alias: TrackedTarget(track_id=track, class_id=2)},
                start_ms=index * 40, start_frame_id=index, confirmed_ms=index * 40, confirmed_frame_id=index,
            )

        self.assertEqual(detector.update(frame(0), None, [event(0, "last")]), [])
        self.assertEqual(detector.update(frame(1), None, [event(1, "first"), event(1, "last")]), [])
        self.assertEqual(detector.update(frame(2), None, [event(2, "last", track=2)]), [])
        hit, = detector.update(frame(3), None, [event(3, "last")])
        self.assertEqual((hit.op, hit.start_ms, hit.confirmed_ms), ("sequence", 40, 120))
        self.assertEqual(hit.evidence_ids, ["first:1", "last:3"])
        self.assertEqual(detector.update(frame(4), None, [event(4, "last")]), [])
        detector.update(frame(5), None, [event(5, "first")])
        for index in range(6, 10):
            detector.update(frame(index), None, [])
        self.assertEqual(detector.update(frame(10), None, [event(10, "last")]), [])

    def test_request_validates_references_and_reads_cycle_parameters(self):
        config = default_composite_rules().model_dump(mode="json")
        config["cycle"]["start"]["after_clear_ms"] = 2400
        config["events"] = [{
            "id": "seat_a_motion", "op": "sequence", "within_ms": 2000,
            "steps": [{"event": "seat_a_in"}, {"event": "seat_a_moves"}],
            "same_tracks": ["seat_a"],
        }]
        source = {"type": "camera", "index": 0}
        request = RunRequest.model_validate({"source": source, "composite_rules": config})
        self.assertEqual(request.composite_rules.cycle.start.after_clear_ms, 2400)
        config["events"][0]["steps"][1]["event"] = "missing_rule"
        with self.assertRaisesRegex(ValidationError, "Unknown event"):
            RunRequest.model_validate({"source": source, "composite_rules": config})


class NamedEventTests(TestCase):
    def test_tube_positions_are_independent_and_grip_requires_overlap(self):
        primitives = PrimitiveEventDetector(default_primitive_rules())
        detector = CompositeEventDetector(default_composite_rules().events)
        cycle = WorkpieceCycleEvent(
            phase="started", run_id="test", cycle_id="test:1", start_ms=0, start_frame_id=0,
            confirmed_ms=0, confirmed_frame_id=0,
        )

        def observe(index, glove_x):
            current = frame(
                index, (3, 200, 200, 130, 216), (1, 250, 220, 36, 90),
                (2, 290, 220, 36, 90), (0, glove_x, 220),
            )
            events = primitives.update(current)
            return detector.update(current, cycle, events)

        approaches = [event for index in range(10) for event in observe(index, 335)]
        self.assertEqual([event.rule_id for event in approaches], ["tube_near_a", "tube_near_b"])
        self.assertTrue(all(event.cycle_id == cycle.cycle_id for event in approaches))
        grip, = [event for index in range(10, 23) for event in observe(index, 310)
                 if event.rule_id == "hand_holds_tube"]
        self.assertEqual((grip.rule_id, grip.confirmed_ms - grip.start_ms), ("hand_holds_tube", 320))
        self.assertEqual(set(grip.subjects), {"glove", "tube"})
        self.assertEqual(observe(23, 335), [])

    def test_custom_primitive_only_request_keeps_working(self):
        source = {"type": "camera", "index": 0}
        request = RunRequest.model_validate({"source": source, "primitive_rules": None})
        self.assertEqual(request.composite_rules.events, [])
