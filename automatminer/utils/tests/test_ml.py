"""
Tests for machine learning related utils.
"""

import unittest

import pandas as pd

from automatminer.utils.ml import is_greater_better, \
    regression_or_classification


class TestMLTools(unittest.TestCase):

    def test_is_greater_better(self):
        self.assertTrue(is_greater_better('accuracy'))
        self.assertTrue(is_greater_better('r2_score'))
        self.assertTrue(is_greater_better('neg_mean_squared_error'))
        self.assertFalse(is_greater_better('mean_squared_error'))

    def test_regression_or_classification(self):
        s = pd.Series(data=["4", "5", "6"])
        self.assertTrue(regression_or_classification(s) == "regression")

        s = pd.Series(data=[1, 2, 3])
        self.assertTrue(regression_or_classification(s) == "regression")

        s = pd.Series(data=["a", "b", "c"])
        self.assertTrue(regression_or_classification(s) == "classification")

        s = pd.Series(data=["a1", "b", "c"])
        self.assertTrue(regression_or_classification(s) == "classification")


if __name__ == "__main__":
    unittest.main()
