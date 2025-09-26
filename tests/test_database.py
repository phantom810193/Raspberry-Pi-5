import tempfile
import unittest
from pathlib import Path

import numpy as np

from pi_kiosk import database


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.conn = database.connect(self.db_path)
        database.initialize_db(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_sample_transactions_inserted(self) -> None:
        database.ensure_sample_transactions(self.conn)
        cursor = self.conn.execute("SELECT COUNT(*) FROM transactions")
        (count,) = cursor.fetchone()
        self.assertGreaterEqual(count, 5)

    def test_register_member_is_idempotent(self) -> None:
        database.register_member(self.conn, "member-abc", "2024-05-20T00:00:00")
        database.register_member(self.conn, "member-abc", "2024-05-21T00:00:00")
        row = database.get_member(self.conn, "member-abc")
        self.assertIsNotNone(row)
        self.assertEqual(row["source"], "unknown")
        self.assertEqual(row["updated_at"], "2024-05-20T00:00:00")

        cursor = self.conn.execute("SELECT COUNT(*) FROM members")
        (count,) = cursor.fetchone()
        self.assertEqual(count, 1)

        database.update_member_metadata(
            self.conn,
            "member-abc",
            source="trained",
            updated_at="2024-06-01T00:00:00",
        )
        updated = database.get_member(self.conn, "member-abc")
        self.assertEqual(updated["source"], "trained")
        self.assertEqual(updated["updated_at"], "2024-06-01T00:00:00")

    def test_face_feature_lifecycle(self) -> None:
        descriptor = np.ones(128, dtype=np.float32).tobytes()
        created_at = "2025-01-01T00:00:00"
        database.register_member(self.conn, "member-xyz", created_at)
        database.store_face_feature(self.conn, "member-xyz", descriptor, created_at)

        row = database.get_face_feature(self.conn, "member-xyz")
        self.assertIsNotNone(row)
        self.assertEqual(row["member_id"], "member-xyz")

        ids = database.list_face_feature_ids(self.conn)
        self.assertIn("member-xyz", ids)

        removed = database.delete_face_feature(self.conn, "member-xyz")
        self.assertTrue(removed)
        self.assertIsNone(database.get_face_feature(self.conn, "member-xyz"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
