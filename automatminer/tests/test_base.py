"""
Tests for the base classes.
"""
import unittest

import pandas as pd
from sklearn.exceptions import NotFittedError

from automatminer.base import DFTransformer
from automatminer.utils.pkg import check_fitted, set_fitted


class TestTransformerGood(DFTransformer):
    """
    A test transformer and logger.

    Args:
        config_attr: Some attr to be set at initialization
    """

    def __init__(self, config_attr):
        self.config_attr = config_attr
        self.target = None
        super(TestTransformerGood, self).__init__()

    @set_fitted
    def fit(self, df, target):
        """
        Determine the target of the dataframe.

        Args:
            df (pandas.DataFrame): The dataframe to be transformed.
            target (str): The fit target

        Returns:
            TestTransformer
        """
        if target in df.columns:
            self.target = target
        else:
            raise ValueError("Target {} not in dataframe.".format(target))
        return self

    @check_fitted
    def transform(self, df, target):
        """
        Drop the target set during fitting.

        Args:
            df (pandas.DataFrame): The dataframe to be transformed.
            target (str): The transform target (not the same as fit target
                necessarily)

        Returns:
            df (pandas.DataFrame): The transformed dataframe.
        """
        df = df.drop(columns=self.target)
        return df


class TestTransformerBad(DFTransformer):
    """
    A test transformer, implemented incorrectly.
    """

    def __init__(self):
        pass


class TestBaseTransformers(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    def test_DFTransformer(self):
        ttg = TestTransformerGood(5)
        self.assertTrue(hasattr(ttg, "config_attr"))
        self.assertTrue(ttg.config_attr, 5)
        with self.assertRaises(NotFittedError):
            ttg.transform(self.df, "b")

        ttg.fit(self.df, "a")
        self.assertTrue(ttg.config_attr, 5)

        test = ttg.transform(self.df, "b")
        self.assertTrue("b" in test.columns)
        self.assertTrue("c" in test.columns)
        self.assertTrue("a" not in test.columns)

        test = ttg.fit_transform(self.df, "c")
        self.assertTrue("c" not in test.columns)
        self.assertTrue("a" in test.columns)
        self.assertTrue("b" in test.columns)

        with self.assertRaises(TypeError):
            _ = TestTransformerBad()

    def test_DFTransformer_BaseEstimator_behavior(self):
        ttg = TestTransformerGood(5)
        ttg_nested = TestTransformerGood(ttg)

        self.assertEqual(ttg.get_params()["config_attr"], 5)
        self.assertEqual(ttg_nested.get_params()["config_attr__config_attr"], 5)

        ttg.set_params(config_attr=6)
        self.assertEqual(ttg.get_params()["config_attr"], 6)
        self.assertEqual(ttg_nested.get_params()["config_attr__config_attr"], 6)

        ttg_nested.set_params(config_attr__config_attr=7)
        self.assertEqual(ttg.get_params()["config_attr"], 7)
        self.assertEqual(ttg_nested.get_params()["config_attr__config_attr"], 7)


if __name__ == "__main__":
    unittest.main()
