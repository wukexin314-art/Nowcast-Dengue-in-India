import unittest

from honestforecast.features import (
    REGISTERED_TRANSFORMS,
    FeatureError,
    as_of_join,
    fit_fold_transform,
)


class FeatureTests(unittest.TestCase):
    def test_as_of_join_uses_origin_available_revision(self):
        observations = [
            {
                "feature_id": "x",
                "observation_date": "2023-01-01",
                "available_at": "2023-02-01",
                "value": 10,
                "source_version": "early",
            },
            {
                "feature_id": "x",
                "observation_date": "2023-01-01",
                "available_at": "2023-04-01",
                "value": 99,
                "source_version": "revised",
            },
        ]
        joined = as_of_join(
            [
                {
                    "feature_id": "x",
                    "observation_date": "2023-01-01",
                    "origin": "2023-03-01",
                }
            ],
            observations,
        )
        self.assertEqual(joined[0]["value"], 10.0)
        self.assertEqual(joined[0]["source_version"], "early")
        self.assertEqual(joined[0]["vintage_status"], "as_of_value_with_later_revision")

    def test_fold_parameters_ignore_later_values_and_emit_lineage(self):
        base = [
            {"date": f"2023-{month:02d}-01", "value": value}
            for month, value in enumerate([1, 2, None, 4, 1000], start=1)
        ]
        transformed, lineage = fit_fold_transform(
            base,
            train_end="2023-04-01",
            output_feature_id="x_clean",
            source_feature_id="x",
            family="proxy",
            source_version="v1",
            availability_rule="available_at <= origin",
        )
        changed = [dict(row) for row in base]
        changed[-1]["value"] = 1_000_000
        _, changed_lineage = fit_fold_transform(
            changed,
            train_end="2023-04-01",
            output_feature_id="x_clean",
            source_feature_id="x",
            family="proxy",
            source_version="v1",
            availability_rule="available_at <= origin",
        )
        self.assertEqual(lineage.parameters, changed_lineage.parameters)
        self.assertTrue(transformed[2]["was_imputed"])
        self.assertEqual(lineage.fit_end, "2023-04-01")
        self.assertEqual(lineage.source_version, "v1")

    def test_log1p_rejects_negative_values(self):
        with self.assertRaises(FeatureError):
            fit_fold_transform(
                [{"date": "2023-01-01", "value": -1}],
                train_end="2023-01-01",
                output_feature_id="x",
                source_feature_id="x",
                family="x",
                source_version="1",
                availability_rule="known ahead",
                transform="log1p",
            )

    def test_unregistered_transform_is_rejected(self):
        with self.assertRaises(FeatureError):
            fit_fold_transform(
                [{"date": "2023-01-01", "value": 1}],
                train_end="2023-01-01",
                output_feature_id="x",
                source_feature_id="x",
                family="x",
                source_version="1",
                availability_rule="known ahead",
                transform="lag_99",
            )

    def test_all_registered_transforms_execute(self):
        rows = [
            {"date": f"{2022 + (month - 1) // 12}-{(month - 1) % 12 + 1:02d}-01", "value": month}
            for month in range(1, 19)
        ]
        for transform in REGISTERED_TRANSFORMS:
            with self.subTest(transform=transform):
                output, lineage = fit_fold_transform(
                    rows,
                    train_end="2023-06-01",
                    output_feature_id=f"x_{transform}",
                    source_feature_id="x",
                    family="x",
                    source_version="1",
                    availability_rule="known ahead",
                    transform=transform,
                )
                self.assertEqual(len(output), 18)
                self.assertEqual(lineage.transform, transform)


if __name__ == "__main__":
    unittest.main()
