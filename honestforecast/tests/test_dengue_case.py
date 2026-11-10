import unittest

from honestforecast.dengue_case import ridge_predict


class DengueCaseTests(unittest.TestCase):
    def test_ridge_prediction_is_finite_and_residuals_align(self):
        prediction, residuals = ridge_predict(
            [[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0], [4.0], ridge=0.01
        )
        self.assertGreater(prediction, 8.0)
        self.assertLess(prediction, 9.1)
        self.assertEqual(len(residuals), 4)


if __name__ == "__main__":
    unittest.main()
