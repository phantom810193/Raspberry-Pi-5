import tempfile
import unittest
from pathlib import Path

from pi_kiosk import database
from pi_kiosk.flask_app import create_app
from pi_kiosk.pipeline import PipelineConfig, create_pipeline


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
        self.pipeline = create_pipeline(config)
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


if __name__ == "__main__":
    unittest.main()
