"""
Assorted package utils.
"""

import unittest

import pandas as pd
from sklearn.exceptions import NotFittedError

from automatminer.utils.pkg import compare_columns, check_fitted, \
    set_fitted
from automatminer.base import DFTransformer


class MyTransformer(DFTransformer):
    def __init__(self):
        self.is_fit = False

    @set_fitted
    def fit(self, df, target):
        return df

    @check_fitted
    def transform(self, df, target):
        return df


class TestPackageTools(unittest.TestCase):

    def test_compare_columns(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [2, 3]})
        df2 = pd.DataFrame({"b": [3, 4], "c": [4, 5]})
        comparison = compare_columns(df1, df2)
        self.assertTrue(comparison["mismatch"])
        self.assertListEqual(comparison["df1_not_in_df2"], ["a"])
        self.assertListEqual(comparison["df2_not_in_df1"], ["c"])

        comparison2 = compare_columns(df1, df1)
        self.assertFalse(comparison2["mismatch"])

        comparison3 = compare_columns(df1, df2, ignore=["c"])
        self.assertTrue(comparison3["mismatch"])
        self.assertListEqual(comparison3["df1_not_in_df2"], ["a"])
        self.assertListEqual(comparison3["df2_not_in_df1"], [])

    def test_fitting_decorations(self):
        df = pd.DataFrame({"a": [1, 2], "b": [2, 3]})
        mt = MyTransformer()

        self.assertFalse(mt.is_fit)
        mt.fit(df, "")
        self.assertTrue(mt.is_fit)
        df = mt.transform(df, "")

        mt2 = MyTransformer()
        self.assertRaises(NotFittedError, mt2.transform, [df, ""])


if __name__ == "__main__":
    unittest.main()
