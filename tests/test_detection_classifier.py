from __future__ import annotations

import unittest

import numpy as np

from pi_kiosk.detection import ClassifierModel


class ClassifierModelTest(unittest.TestCase):
    def test_classifier_match_thresholds(self) -> None:
        base = np.ones(128, dtype=np.float32)
        model = ClassifierModel(labels=["member-001"], embeddings=np.stack([base]), distance_threshold=0.5)

        close_descriptor = base + 0.01
        far_descriptor = base + 1.0

        self.assertEqual(model.match(close_descriptor), "member-001")
        self.assertIsNone(model.match(far_descriptor))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
