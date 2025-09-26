from __future__ import annotations

import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np

from pi_kiosk import database
from pi_kiosk.detection import ClassifierModel, FaceLocation, FaceMatch
from pi_kiosk.pipeline import AdvertisementPipeline, PipelineConfig, create_pipeline


class SequenceIdentifier:
    """Identifier stub that returns a predefined sequence of detections."""

    def __init__(self, outputs):
        self._outputs = deque(outputs)

    def identify(self, image):  # pragma: no cover - simple helper
        if self._outputs:
            return list(self._outputs.popleft())
        return []


class FaceIdentifierStub:
    """Lightweight stub emulating FaceIdentifier for tests."""

    def __init__(self, outputs):
        self._outputs = deque(outputs)
        self._classifier = None
        self._last_matches: list[FaceMatch] = []

    def identify_faces(self, image):  # pragma: no cover - simple helper
        if self._outputs:
            self._last_matches = list(self._outputs.popleft())
        else:
            self._last_matches = []
        return list(self._last_matches)

    def identify(self, image):  # pragma: no cover - compatibility stub
        return [match.label for match in self._last_matches]

    def set_classifier(self, classifier):  # pragma: no cover - simple helper
        self._classifier = classifier

    def get_classifier(self):  # pragma: no cover - simple helper
        return self._classifier


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.stub_ai = self.StubAI()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    class StubAI:
        def __init__(self) -> None:
            self.outputs = []
            self.pipeline: AdvertisementPipeline | None = None
            self.busy_states: list[bool] = []

        def enqueue(self, message: str) -> None:
            self.outputs.append(message)

        def generate(self, member_id, context):  # pragma: no cover - simple helper
            if self.pipeline is not None:
                self.busy_states.append(self.pipeline.ai_busy())
            if self.outputs:
                return self.outputs.pop(0)
            return None

    def test_simulated_member_updates_latest_message(self) -> None:
        config = PipelineConfig(db_path=self.db_path, simulated_member_ids=("member-001",))
        pipeline = create_pipeline(config, ai_client=self.stub_ai)
        message = pipeline.simulate_member("member-001")
        self.assertIn("會員ID-001", message)
        latest_message, latest_member, timestamp = pipeline.latest_message()
        self.assertEqual(latest_message, message)
        self.assertEqual(latest_member, "member-001")
        self.assertIsNotNone(timestamp)

    def test_process_frame_respects_cooldown(self) -> None:
        config = PipelineConfig(db_path=self.db_path, simulated_member_ids=("member-001",), cooldown_seconds=60)
        pipeline = create_pipeline(config, ai_client=self.stub_ai)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        first = pipeline.process_frame(frame)
        second = pipeline.process_frame(frame)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_ai_output_overrides_default_message(self) -> None:
        config = PipelineConfig(db_path=self.db_path, simulated_member_ids=("member-001",))
        self.stub_ai.enqueue("AI 生成的專屬廣告")
        pipeline = create_pipeline(config, ai_client=self.stub_ai)
        self.stub_ai.pipeline = pipeline
        message = pipeline.simulate_member("member-001")
        self.assertEqual(message, "AI 生成的專屬廣告")
        self.assertFalse(pipeline.ai_busy())

    def test_ai_busy_flag_during_generation(self) -> None:
        config = PipelineConfig(db_path=self.db_path, simulated_member_ids=("member-001",))
        self.stub_ai.enqueue("generated")
        pipeline = create_pipeline(config, ai_client=self.stub_ai)
        self.stub_ai.pipeline = pipeline
        pipeline.simulate_member("member-001")
        self.assertIn(True, self.stub_ai.busy_states)
        self.assertFalse(pipeline.ai_busy())

    def test_idle_reset_restores_waiting_message(self) -> None:
        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=None,
            cooldown_seconds=0,
            idle_reset_seconds=1,
        )
        identifier = SequenceIdentifier([["member-001"], []])
        pipeline = AdvertisementPipeline(config, identifier=identifier, ai_client=self.stub_ai)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with mock.patch("pi_kiosk.pipeline.time.time") as mock_time:
            mock_time.return_value = 0.0
            pipeline.process_frame(frame)
            message, member_id, timestamp = pipeline.latest_message()
            self.assertEqual(member_id, "member-001")
            self.assertIsNotNone(timestamp)

            mock_time.return_value = 2.0
            pipeline.process_frame(frame)
            message, member_id, timestamp = pipeline.latest_message()
            self.assertEqual(message, "等待辨識中...")
            self.assertIsNone(member_id)
            self.assertIsNone(timestamp)

    def test_idle_reset_triggers_across_multiple_cycles(self) -> None:
        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=None,
            cooldown_seconds=0,
            idle_reset_seconds=1,
        )
        identifier = SequenceIdentifier([
            ["member-001"],
            [],
            [],
            ["member-002"],
            [],
            [],
        ])
        pipeline = AdvertisementPipeline(config, identifier=identifier, ai_client=self.stub_ai)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with mock.patch("pi_kiosk.pipeline.time.time") as mock_time:
            mock_time.side_effect = [
                0.0, 0.0,  # first detection (start/end)
                2.0, 2.0,  # first idle reset
                3.0, 3.0,  # idle steady
                4.5, 4.5,  # second detection
                6.0, 6.0,  # second idle reset
                7.5, 7.5,  # idle steady
            ]

            pipeline.process_frame(frame)
            message, member_id, _ = pipeline.latest_message()
            self.assertEqual(member_id, "member-001")

            pipeline.process_frame(frame)
            message, member_id, _ = pipeline.latest_message()
            self.assertEqual(message, "等待辨識中...")
            self.assertIsNone(member_id)

            pipeline.process_frame(frame)  # stays idle
            message, member_id, _ = pipeline.latest_message()
            self.assertEqual(message, "等待辨識中...")

            pipeline.process_frame(frame)
            message, member_id, _ = pipeline.latest_message()
            self.assertEqual(member_id, "member-002")

            pipeline.process_frame(frame)
            message, member_id, _ = pipeline.latest_message()
            self.assertEqual(message, "等待辨識中...")
            self.assertIsNone(member_id)

            pipeline.process_frame(frame)
            message, member_id, _ = pipeline.latest_message()
            self.assertEqual(message, "等待辨識中...")


    def test_auto_enroll_stores_first_face(self) -> None:
        descriptor = np.linspace(0.0, 1.0, 128, dtype=np.float32)
        matches = [
            FaceMatch(
                descriptor=descriptor,
                location=FaceLocation(top=0, right=4, bottom=4, left=0),
                label="member-auto",
                matched=False,
                distance=None,
            )
        ]

        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=None,
            cooldown_seconds=0,
            idle_reset_seconds=None,
            use_trained_classifier=False,
            auto_enroll_first_face=True,
        )

        identifier = FaceIdentifierStub([matches])
        pipeline = AdvertisementPipeline(config, identifier=identifier, ai_client=self.stub_ai)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        pipeline.process_frame(frame)

        row = database.get_face_feature(pipeline.conn, "member-auto")
        self.assertIsNotNone(row)
        self.assertIn("member-auto", pipeline.list_face_feature_ids())
        self.assertIsNotNone(identifier.get_classifier())
        member_row = database.get_member(pipeline.conn, "member-auto")
        self.assertIsNotNone(member_row)
        self.assertEqual(member_row["source"], "auto_enroll")

    def test_manual_face_feature_management(self) -> None:
        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=None,
            cooldown_seconds=0,
            idle_reset_seconds=None,
            use_trained_classifier=False,
        )
        identifier = FaceIdentifierStub([])
        pipeline = AdvertisementPipeline(config, identifier=identifier, ai_client=self.stub_ai)

        descriptor = np.full(128, 0.2, dtype=np.float32)
        pipeline.add_face_feature("member-new", descriptor)

        self.assertIn("member-new", pipeline.list_face_feature_ids())
        self.assertIsNotNone(identifier.get_classifier())
        member_row = database.get_member(pipeline.conn, "member-new")
        self.assertIsNotNone(member_row)
        self.assertEqual(member_row["source"], "api")

        removed = pipeline.remove_face_feature("member-new")
        self.assertTrue(removed)
        self.assertNotIn("member-new", pipeline.list_face_feature_ids())

    def test_only_largest_face_processed(self) -> None:
        large_descriptor = np.zeros(128, dtype=np.float32)
        small_descriptor = np.ones(128, dtype=np.float32)

        large_match = FaceMatch(
            descriptor=large_descriptor,
            location=FaceLocation(top=0, right=100, bottom=120, left=0),
            label="member-large",
            matched=True,
            distance=0.3,
        )
        small_match = FaceMatch(
            descriptor=small_descriptor,
            location=FaceLocation(top=10, right=30, bottom=40, left=5),
            label="member-small",
            matched=True,
            distance=0.3,
        )

        identifier = FaceIdentifierStub([[large_match, small_match]])
        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=None,
            cooldown_seconds=0,
            idle_reset_seconds=None,
            use_trained_classifier=False,
        )
        pipeline = AdvertisementPipeline(config, identifier=identifier, ai_client=self.stub_ai)
        frame = np.zeros((120, 100, 3), dtype=np.uint8)

        pipeline.process_frame(frame)

        _, latest_member, _ = pipeline.latest_message()
        self.assertEqual(latest_member, "member-large")
        self.assertIsNotNone(database.get_member(pipeline.conn, "member-large"))
        self.assertIsNone(database.get_member(pipeline.conn, "member-small"))

        image_bytes, metadata, timestamp = pipeline.debug_snapshot()
        self.assertIsNotNone(image_bytes)
        self.assertGreater(len(image_bytes), 0)
        labels = {entry["label"] for entry in metadata}
        self.assertEqual(labels, {"member-large", "member-small"})
        self.assertIsNotNone(timestamp)

    def test_trained_classifier_does_not_register_member(self) -> None:
        descriptor = np.zeros(128, dtype=np.float32)
        match = FaceMatch(
            descriptor=descriptor,
            location=FaceLocation(top=0, right=4, bottom=4, left=0),
            label="member-trained",
            matched=True,
            distance=0.3,
        )

        base_classifier = ClassifierModel(
            labels=["member-trained"],
            embeddings=np.stack([descriptor]),
            distance_threshold=0.6,
        )

        identifier = FaceIdentifierStub([[match]])
        identifier.set_classifier(base_classifier)
        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=None,
            cooldown_seconds=0,
            idle_reset_seconds=None,
            use_trained_classifier=True,
        )
        pipeline = AdvertisementPipeline(config, identifier=identifier, ai_client=self.stub_ai)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        pipeline.process_frame(frame)

        self.assertIsNone(database.get_member(pipeline.conn, "member-trained"))

    def test_trained_classifier_updates_existing_member(self) -> None:
        descriptor = np.zeros(128, dtype=np.float32)
        match = FaceMatch(
            descriptor=descriptor,
            location=FaceLocation(top=0, right=4, bottom=4, left=0),
            label="member-trained",
            matched=True,
            distance=0.3,
        )

        identifier = FaceIdentifierStub([[match]])
        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=None,
            cooldown_seconds=0,
            idle_reset_seconds=None,
            use_trained_classifier=True,
        )
        pipeline = AdvertisementPipeline(config, identifier=identifier, ai_client=self.stub_ai)

        initial_timestamp = "2024-01-01T00:00:00"
        database.register_member(
            pipeline.conn,
            "member-trained",
            initial_timestamp,
            source="trained",
            updated_at=initial_timestamp,
        )

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        pipeline.process_frame(frame)

        member_row = database.get_member(pipeline.conn, "member-trained")
        self.assertIsNotNone(member_row)
        self.assertEqual(member_row["source"], "trained")
        self.assertNotEqual(member_row["updated_at"], initial_timestamp)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
