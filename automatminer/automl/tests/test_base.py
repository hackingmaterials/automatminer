"""
Tests for base classes for automl.
"""

import unittest

import pandas as pd
from sklearn.exceptions import NotFittedError

from automatminer.automl.base import DFMLAdaptor
from automatminer.utils.pkg import check_fitted, set_fitted


class TestAdaptorBad(DFMLAdaptor):
    """
    A test adaptor for automl backends, implemented incorrectly.
    """

    def __init__(self):
        pass


class TestAdaptorGood(DFMLAdaptor):
    """
    A test adaptor for automl backends, implemented correctly.
    """

    def __init__(self, config_attr):
        self.config_attr = config_attr
        self.target = None
        self._ml_data = None
        self._best_pipeline = None
        self._backend = None
        self._features = None
        self._fitted_target = None
        super(DFMLAdaptor, self).__init__()

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

        self._fitted_target = target
        self._best_pipeline = "pipeline1"
        self._ml_data = {"y": df[target], "X": df.drop(columns=[target])}
        self._backend = "mybackend"
        self._features = self._ml_data["X"].columns.tolist()
        return self

    @check_fitted
    def predict(self, df, target):
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

    @property
    def backend(self):
        return self._backend

    @property
    def features(self):
        return self._features

    @property
    def best_pipeline(self):
        return self._best_pipeline

    @property
    def fitted_target(self):
        return self._fitted_target


class TestBaseAutoMLTransformers(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    def test_DFMLAdaptor(self):
        tag = TestAdaptorGood(config_attr=5)

        with self.assertRaises(NotFittedError):
            tag.transform(self.df, "a")

        with self.assertRaises(NotFittedError):
            tag.predict(self.df, "a")

        tag.fit(self.df, "a")
        self.assertTrue(hasattr(tag, "features"))
        self.assertTrue(hasattr(tag, "best_pipeline"))
        self.assertTrue(hasattr(tag, "backend"))
        self.assertTrue(hasattr(tag, "fitted_target"))
        self.assertTrue(tag.is_fit)
        self.assertTrue(tag.best_pipeline == "pipeline1")
        self.assertTrue(tag.backend == "mybackend")
        self.assertTrue(tag.features[0] == "b")

        predicted = tag.predict(self.df, "b")
        self.assertTrue("b" in predicted)
        self.assertTrue("c" in predicted)
        self.assertTrue("a" not in predicted)

        predicted2 = tag.fit_transform(self.df, "c")
        self.assertTrue("b" in predicted2)
        self.assertTrue("a" in predicted2)
        self.assertTrue("c" not in predicted2)

        with self.assertRaises(TypeError):
            TestAdaptorBad()


if __name__ == "__main__":
    unittest.main()
