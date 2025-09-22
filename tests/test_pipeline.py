import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np

from pi_kiosk.pipeline import AdvertisementPipeline, PipelineConfig, create_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_simulated_member_updates_latest_message(self) -> None:
        config = PipelineConfig(db_path=self.db_path, simulated_member_ids=("member-001",))
        pipeline = create_pipeline(config)
        message = pipeline.simulate_member("member-001")
        self.assertIn("會員ID-001", message)
        latest_message, latest_member, timestamp = pipeline.latest_message()
        self.assertEqual(latest_message, message)
        self.assertEqual(latest_member, "member-001")
        self.assertIsNotNone(timestamp)

    def test_process_frame_respects_cooldown(self) -> None:
        config = PipelineConfig(db_path=self.db_path, simulated_member_ids=("member-001",), cooldown_seconds=60)
        pipeline = create_pipeline(config)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        first = pipeline.process_frame(frame)
        second = pipeline.process_frame(frame)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_idle_reset_restores_waiting_message(self) -> None:
        class SequenceIdentifier:
            def __init__(self, outputs):
                self._outputs = deque(outputs)

            def identify(self, image):
                if self._outputs:
                    return list(self._outputs.popleft())
                return []

        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=None,
            cooldown_seconds=0,
            idle_reset_seconds=1,
        )
        identifier = SequenceIdentifier([["member-001"], []])
        pipeline = AdvertisementPipeline(config, identifier=identifier)
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
