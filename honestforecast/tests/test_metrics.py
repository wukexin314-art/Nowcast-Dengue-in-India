import unittest

from honestforecast.metrics import MetricError, quantile_column, summarize_predictions
from honestforecast.synthetic import clean_predictions


class MetricTests(unittest.TestCase):
    def test_metrics_are_regenerated_from_rows(self):
        metrics = summarize_predictions(clean_predictions())
        self.assertEqual(metrics["n_cases"], 16)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["coverage_95"], 1.0)
        self.assertEqual(metrics["width_95"], 40.0)

    def test_crossing_quantiles_fail(self):
        rows = clean_predictions()
        rows[0][quantile_column(0.025)] = rows[0][quantile_column(0.1)] + 1
        with self.assertRaises(MetricError):
            summarize_predictions(rows)


if __name__ == "__main__":
    unittest.main()
