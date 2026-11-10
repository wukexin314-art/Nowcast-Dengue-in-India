import csv
import json
import tempfile
import unittest
from pathlib import Path

from honestforecast.metrics import summarize_predictions
from honestforecast.synthetic import run


class SyntheticTests(unittest.TestCase):
    def test_end_to_end_outputs_and_metric_recalculation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            receipt = run(output)
            self.assertTrue(all(value in {"passed", "detected"} for value in receipt["scenario_results"].values()))
            with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
                recalculated = summarize_predictions(list(csv.DictReader(handle)))
            saved = json.loads((output / "metrics.json").read_text())
            self.assertEqual(recalculated, saved)
            self.assertTrue((output / "evidence_card.md").is_file())
            self.assertTrue((output / "feature_lineage.json").is_file())
            self.assertLessEqual(
                receipt["null_feature_experiment"]["honest_false_promotion_rate"],
                receipt["null_feature_experiment"]["naive_false_promotion_rate"],
            )
            self.assertEqual(receipt["scenario_results"]["S2_latest_revision_as_old_vintage"], "detected")
            self.assertEqual(receipt["scenario_results"]["S3_post_hoc_null_lags"], "detected")
            self.assertEqual(receipt["scenario_results"]["S8_single_outbreak_gain"], "detected")


if __name__ == "__main__":
    unittest.main()
