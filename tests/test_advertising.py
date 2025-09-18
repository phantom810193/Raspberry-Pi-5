import unittest

from pi_kiosk import advertising


class AdvertisingTests(unittest.TestCase):
    def test_generate_message_first_time(self) -> None:
        message = advertising.generate_message("member-999", [])
        self.assertIn("會員ID-", message)
        self.assertIn("首次光臨", message)

    def test_generate_message_with_history(self) -> None:
        transactions = [
            advertising.Transaction(item="牛奶", amount=90, timestamp="2024-05-20T09:30:00"),
            advertising.Transaction(item="餅乾", amount=50, timestamp="2024-05-18T10:00:00"),
        ]
        message = advertising.generate_message("member-123", transactions)
        self.assertIn("牛奶", message)
        self.assertIn("9折", message)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
