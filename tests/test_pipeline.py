import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi_kiosk.pipeline import PipelineConfig, create_pipeline


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
