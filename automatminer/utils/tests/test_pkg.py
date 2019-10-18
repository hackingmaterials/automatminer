"""
Assorted package utils.
"""
import os
import unittest

import pandas as pd
from automatminer import __version__
from automatminer.base import DFTransformer
from automatminer.utils.pkg import (
    AMM_SUPPORTED_EXTS,
    check_fitted,
    compare_columns,
    get_version,
    save_dict_to_file,
    set_fitted,
)
from sklearn.exceptions import NotFittedError


class MyTransformer(DFTransformer):
    def __init__(self):
        super(MyTransformer, self).__init__()

    @set_fitted
    def fit(self, df, target):
        return df

    @check_fitted
    def transform(self, df, target):
        return df


class TestPackageTools(unittest.TestCase):
    def setUp(self) -> None:
        self.remant_base_path = os.path.dirname(__file__)
        self.remant_file_prefix = "saved"

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

    def test_save_dict_to_file(self):
        test_dict = {"a": "A", "b": 1, "c": [1, "q"], "d": {"m": [3, 4]}}
        for ext in AMM_SUPPORTED_EXTS:
            filename = self._get_remnant_path(ext)
            save_dict_to_file(test_dict, filename=filename)
            self.assertTrue(os.path.isfile(filename))

    def test_get_version(self):
        v = get_version()
        self.assertEqual(v, __version__)

    def tearDown(self) -> None:
        remnants = [self._get_remnant_path(ext) for ext in AMM_SUPPORTED_EXTS]
        for remnant in remnants:
            if os.path.exists(remnant):
                os.remove(remnant)

    def _get_remnant_path(self, ext):
        relative_fname = self.remant_file_prefix + ext
        filename = os.path.join(self.remant_base_path, relative_fname)
        return filename


if __name__ == "__main__":
    unittest.main()
