import os
import unittest

import numpy as np
import pandas as pd

from mslearn.preprocessing.core import DataCleaner, FeatureReducer
from mslearn.preprocessing.feature_selection import TreeBasedFeatureReduction, \
    rebate
from mslearn.utils.package_tools import compare_columns

test_dir = os.path.dirname(__file__)


class TestPreprocess(unittest.TestCase):
    # todo: need test case for categorical target and categorical features -AD

    def setUp(self):
        df = pd.read_csv(os.path.join(test_dir, 'test_featurized_df.csv'))
        self.test_df = df.drop(columns=["formula"])

    def test_DataCleaner(self):
        """
        A basic test ensuring Preprocess can handle numerical features and
        features/targets  that may be strings but should be numbers.

        Returns: None
        """
        df = self.test_df
        target = 'gap expt'
        dc = DataCleaner()

        # Test the case of numbers as strings
        df[target] = df[target].astype(str)
        df = dc.fit_transform(df, target)
        self.assertAlmostEqual(df[target].iloc[0], 0.35)

        # Test if there is an nan in target
        df[target].iloc[8] = np.nan
        df = dc.fit_transform(df, target)
        self.assertEqual(df.shape[0], self.test_df.shape[0] - 1)

        # Test if there is an nan in feature
        df['HOMO_energy'].iloc[40] = np.nan
        df = dc.fit_transform(df, target)
        self.assertEqual(df.shape[0], self.test_df.shape[0] - 2)

        # Test if nan threshold is exceeded for a feature
        df["LUMO_energy"].iloc[:-2] = [np.nan] * (df.shape[0] - 2)
        df = dc.fit_transform(df, target)
        self.assertEqual(df.shape[1], self.test_df.shape[1] - 1)

        # test transferability
        df2 = self.test_df
        df2 = df2.drop(columns=[target])
        df2 = dc.transform(df2, target)
        self.assertFalse(compare_columns(df, df2, ignore=target)["mismatch"])
        self.assertTrue(target not in df2.columns)

    def test_FeatureReducer(self):
        df = self.test_df
        target = 'gap expt'
        fr = FeatureReducer()

        # ultra-basic case: are we reducing at least 1 feature?
        df = fr.fit_transform(df, target)
        self.assertTrue(df.shape[1] < self.test_df.shape[1])

        # ensure metadata is being written correctly
        self.assertTrue(target not in fr.retained_features)
        self.assertTrue(len(list(fr.removed_features.keys())) == 2)

        # ensure other combinations of feature reducers are working
        fr = FeatureReducer(reducers=('corr', 'rebate'), n_rebate_features=40)
        df = fr.fit_transform(self.test_df, target)
        self.assertEqual(df.shape[1], 41)  # 40 features + target
        self.assertTrue(target in df.columns)

        # test transferability
        df2 = self.test_df
        df2 = fr.transform(df2, target)
        self.assertListEqual(df.columns.tolist(), df2.columns.tolist())


class TestFeatureReduction(unittest.TestCase):

    def setUp(self):
        df = pd.read_csv(os.path.join(test_dir, 'test_featurized_df.csv'))
        self.test_df = df.set_index('formula')
        self.random_state = 12

    def test_TreeBasedFeatureReduction(self):
        X = self.test_df.drop('gap expt', axis=1)
        y = self.test_df['gap expt']
        tbfr = TreeBasedFeatureReduction(mode='regression',
                                         random_state=self.random_state)
        tbfr.fit(X, y, cv=3)
        self.assertEqual(len(tbfr.selected_features), 19)
        X_reduced = tbfr.transform(X)
        self.assertEqual(X_reduced.shape, (len(X), 19))
        self.assertTrue('HOMO_energy' in X_reduced.columns)

    def test_rebate(self):
        df_reduced = rebate(self.test_df, "gap expt", 10)
        self.assertEqual(df_reduced.shape[1], 10)
