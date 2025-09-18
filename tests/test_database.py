import tempfile
import unittest
from pathlib import Path

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
        cursor = self.conn.execute("SELECT COUNT(*) FROM members")
        (count,) = cursor.fetchone()
        self.assertEqual(count, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
