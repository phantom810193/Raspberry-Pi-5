import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi_kiosk import database
from pi_kiosk.detection import FaceLocation, FaceMatch
from pi_kiosk.flask_app import create_app
from pi_kiosk.pipeline import AdvertisementPipeline, PipelineConfig, create_pipeline


class StubAI:
    def __init__(self) -> None:
        self.should_fail = False

    def generate(self, member_id, context):  # pragma: no cover - simple helper
        if self.should_fail:
            raise TimeoutError("simulated timeout")
        return None


class FaceIdentifierStub:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self._classifier = None
        self._last = []

    def identify_faces(self, image):  # pragma: no cover - simple helper
        if self.outputs:
            self._last = list(self.outputs.pop(0))
        else:
            self._last = []
        return list(self._last)

    def identify(self, image):  # pragma: no cover - compatibility
        return [match.label for match in self._last]

    def set_classifier(self, classifier):  # pragma: no cover - simple helper
        self._classifier = classifier

    def get_classifier(self):  # pragma: no cover
        return self._classifier


class TransactionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "api.db"
        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=("member-sim",),
            cooldown_seconds=0,
            idle_reset_seconds=None,
        )
        self.stub_ai = StubAI()
        self.pipeline = create_pipeline(config, ai_client=self.stub_ai)
        self.app = create_app(self.pipeline)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.pipeline.conn.close()
        self.tmp.cleanup()

    def test_insert_transactions_success(self) -> None:
        member_id = "member-abc"
        self.pipeline.simulate_member(member_id)

        payload = {
            "member_id": member_id,
            "transactions": [
                {"item": "咖啡豆", "amount": 150, "timestamp": "2025-01-01T10:00:00"},
                {"item": "牛奶", "amount": 75.5, "timestamp": "2025-01-02T12:30:00"},
            ],
        }
        response = self.client.post("/api/transactions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["inserted"], 2)

        rows = database.get_transactions(self.pipeline.conn, member_id)
        items = {(row["item"], row["timestamp"]) for row in rows}
        self.assertIn(("咖啡豆", "2025-01-01T10:00:00"), items)
        self.assertIn(("牛奶", "2025-01-02T12:30:00"), items)

    def test_insert_transactions_requires_existing_member(self) -> None:
        payload = {
            "member_id": "member-missing",
            "transactions": [{"item": "餅乾", "amount": 50, "timestamp": "2025-01-05T09:00:00"}],
        }
        response = self.client.post("/api/transactions", json=payload)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json().get("error"), "member not found")

    def test_insert_transactions_validates_payload(self) -> None:
        member_id = "member-xyz"
        self.pipeline.simulate_member(member_id)
        payload = {
            "member_id": member_id,
            "transactions": [{"item": "茶葉", "timestamp": "2025-01-05T09:00:00"}],
        }
        response = self.client.post("/api/transactions", json=payload)
        self.assertEqual(response.status_code, 400)

    def test_insert_transactions_handles_ai_timeout(self) -> None:
        member_id = "member-timeout"
        self.pipeline.simulate_member(member_id)
        self.stub_ai.should_fail = True

        payload = {
            "member_id": member_id,
            "transactions": [{"item": "咖啡豆", "amount": 150, "timestamp": "2025-01-01T10:00:00"}],
        }
        response = self.client.post("/api/transactions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["inserted"], 1)


class FaceFeatureApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "face-api.db"
        config = PipelineConfig(
            db_path=self.db_path,
            simulated_member_ids=None,
            cooldown_seconds=0,
            idle_reset_seconds=None,
            use_trained_classifier=False,
        )
        self.stub_ai = StubAI()
        self.identifier = FaceIdentifierStub([])
        self.pipeline = AdvertisementPipeline(config, identifier=self.identifier, ai_client=self.stub_ai)
        self.app = create_app(self.pipeline)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.pipeline.conn.close()
        self.tmp.cleanup()

    def test_create_and_delete_face_feature(self) -> None:
        descriptor = np.linspace(0.0, 1.0, 128, dtype=np.float32).tolist()
        response = self.client.post(
            "/api/face-features",
            json={"member_id": "member-api", "descriptor": descriptor},
        )
        self.assertEqual(response.status_code, 201)
        self.assertIn("member-api", self.pipeline.list_face_feature_ids())

        list_response = self.client.get("/api/face-features")
        self.assertEqual(list_response.status_code, 200)
        self.assertIn("member-api", list_response.get_json().get("members", []))

        member_row = database.get_member(self.pipeline.conn, "member-api")
        self.assertIsNotNone(member_row)
        self.assertEqual(member_row["source"], "api")

        delete_response = self.client.delete("/api/face-features/member-api")
        self.assertEqual(delete_response.status_code, 200)
        self.assertNotIn("member-api", self.pipeline.list_face_feature_ids())
        list_after_delete = self.client.get("/api/face-features")
        self.assertNotIn("member-api", list_after_delete.get_json().get("members", []))

    def test_face_feature_validation_errors(self) -> None:
        response = self.client.post(
            "/api/face-features",
            json={"member_id": "", "descriptor": [0.0] * 128},
        )
        self.assertEqual(response.status_code, 400)

        bad_descriptor = self.client.post(
            "/api/face-features",
            json={"member_id": "member-bad", "descriptor": [1, 2, 3]},
        )
        self.assertEqual(bad_descriptor.status_code, 400)



if __name__ == "__main__":
    unittest.main()
