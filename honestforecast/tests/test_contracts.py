import json
import unittest
from pathlib import Path

from honestforecast.contracts import ContractError, ForecastContract, load_feature_manifest


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_example_contract_and_manifest(self):
        data = json.loads((ROOT / "examples/forecast_contract.json").read_text())
        contract = ForecastContract.from_mapping(data)
        self.assertEqual(contract.horizons, (1, 2))
        specs = load_feature_manifest(ROOT / "examples/feature_manifest.csv")
        self.assertEqual(len(specs), 3)

    def test_duplicate_or_invalid_horizons_fail(self):
        with self.assertRaises(ContractError):
            ForecastContract.from_mapping(
                {
                    "study_id": "x",
                    "target_name": "y",
                    "frequency": "monthly",
                    "horizons": [1, 1],
                    "primary_metric": "WIS",
                }
            )


if __name__ == "__main__":
    unittest.main()
