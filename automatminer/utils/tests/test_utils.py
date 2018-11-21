import unittest
import os

import pandas as pd
import logging
from sklearn.exceptions import NotFittedError

from automatminer.utils.package_tools import compare_columns, check_fitted, set_fitted
from automatminer.utils.ml_tools import is_greater_better, regression_or_classification
from automatminer.utils.log_tools import initialize_logger, initialize_null_logger
from automatminer.base import DataframeTransformer, logger_base_name

run_dir = os.getcwd()


class MyTransformer(DataframeTransformer):
    def __init__(self):
        self.is_fit = False

    @set_fitted
    def fit(self, df, target):
        return df

    @check_fitted
    def transform(self, df, target):
        return df


class TestUtils(unittest.TestCase):

    def test_logger_initialization(self):
        log = initialize_logger(logger_base_name, level=logging.DEBUG)
        log.info("Test logging.")
        log.debug("Test debug.")
        log.warning("Test warning.")

        # test the log is written to run dir (e.g. where the script was called
        # from and not the location of this test file
        log_file = os.path.join(run_dir, logger_base_name + ".log")
        self.assertTrue(os.path.isfile(log_file))

        with open(log_file, 'r') as f:
            lines = f.readlines()

        self.assertTrue("logging" in lines[0])
        self.assertTrue("debug" in lines[1])
        self.assertTrue("warning" in lines[2])

        null = initialize_null_logger("matbench_null")
        null.info("Test null log 1.")
        null.debug("Test null log 2.")
        null.warning("Test null log 3.")

        null_log_file = os.path.join(run_dir, logger_base_name + "_null.log")
        self.assertFalse(os.path.isfile(null_log_file))

    def test_is_greater_better(self):
        self.assertTrue(is_greater_better('accuracy'))
        self.assertTrue(is_greater_better('r2_score'))
        self.assertTrue(is_greater_better('neg_mean_squared_error'))
        self.assertFalse(is_greater_better('mean_squared_error'))

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

    def test_regression_or_classification(self):
        s = pd.Series(data=["4", "5", "6"])
        self.assertTrue(regression_or_classification(s) == "regression")

        s = pd.Series(data=[1, 2, 3])
        self.assertTrue(regression_or_classification(s) == "regression")

        s = pd.Series(data=["a", "b", "c"])
        self.assertTrue(regression_or_classification(s) == "classification")

        s = pd.Series(data=["a1", "b", "c"])
        self.assertTrue(regression_or_classification(s) == "classification")

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
