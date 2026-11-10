import json
import tempfile
import unittest
from pathlib import Path

from honestforecast.comparison import append_decision, compare_feature_family
from honestforecast.synthetic import _shifted_predictions


class ComparisonTests(unittest.TestCase):
    def test_event_concentrated_gain_is_not_promoted(self):
        decision = compare_feature_family(
            "proxy",
            _shifted_predictions([20.0] + [1.0] * 7),
            _shifted_predictions([0.0] + [2.0] * 7),
            confirmation_frozen=True,
            bootstrap_repetitions=200,
        )
        self.assertLess(decision.mean_wis_difference, 0)
        self.assertIn("event_concentration", decision.warnings)
        self.assertEqual(decision.state, "challenger")

    def test_rejected_decision_is_retained_in_append_only_log(self):
        decision = compare_feature_family(
            "bad_proxy",
            _shifted_predictions([1.0] * 8),
            _shifted_predictions([2.0] * 8),
            bootstrap_repetitions=50,
        )
        self.assertEqual(decision.state, "rejected")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            append_decision(path, decision)
            append_decision(path, decision)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["state"] == "rejected" for row in rows))

    def test_promotion_requires_confirmation_and_all_audit_gates(self):
        baseline = _shifted_predictions([5.0] * 8)
        candidate = _shifted_predictions([1.0] * 8)
        challenger = compare_feature_family(
            "good_proxy", baseline, candidate, bootstrap_repetitions=50
        )
        promoted = compare_feature_family(
            "good_proxy",
            baseline,
            candidate,
            confirmation_frozen=True,
            all_audit_gates_passed=True,
            bootstrap_repetitions=50,
        )
        self.assertEqual(challenger.state, "challenger")
        self.assertEqual(promoted.state, "promoted")


if __name__ == "__main__":
    unittest.main()
