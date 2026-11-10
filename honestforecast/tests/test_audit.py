import unittest

from honestforecast.audit import AuditError, assert_common_grid, assert_no_future_availability


class AuditTests(unittest.TestCase):
    def test_common_grid(self):
        rows = [{"origin": "2024-01-01", "target_date": "2024-02-01", "horizon": 1}]
        self.assertEqual(len(assert_common_grid(rows, list(rows))), 1)

    def test_future_availability_fails(self):
        with self.assertRaises(AuditError):
            assert_no_future_availability(
                [{"feature_id": "x", "origin": "2024-01-01", "available_at": "2024-01-02"}]
            )


if __name__ == "__main__":
    unittest.main()
