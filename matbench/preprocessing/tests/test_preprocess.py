import os
import unittest

import numpy as np
import pandas as pd

from matbench.preprocessing.core import DataCleaner, FeatureReducer
from matbench.preprocessing.feature_selection import TreeBasedFeatureReduction, rebate

test_dir = os.path.dirname(__file__)

class TestPreprocess(unittest.TestCase):
    #todo: add more tests

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
        dc = DataCleaner()

        # Test the case of numbers as strings
        df['gap expt'] = df['gap expt'].astype(str)
        df = dc.fit_transform(df, 'gap expt')
        self.assertAlmostEqual(df['gap expt'].iloc[0], 0.35)

        # Test if there is an nan in target
        df['gap expt'].iloc[8] = np.nan
        df = dc.fit_transform(df, 'gap expt')
        self.assertEqual(df.shape[0], self.test_df.shape[0] - 1)

        # Test if there is an nan in feature
        df['HOMO_energy'].iloc[40] = np.nan
        df = dc.fit_transform(df, 'gap expt')
        self.assertEqual(df.shape[0], self.test_df.shape[0] - 2)

        # Test if nan threshold is exceeded for a feature
        df["LUMO_energy"].iloc[:-2] = [np.nan] * (df.shape[0] - 2)
        df = dc.fit_transform(df, 'gap expt')
        self.assertEqual(df.shape[1], self.test_df.shape[1] - 1)

    def test_FeatureReducer(self):
        df = self.test_df
        fr = FeatureReducer()



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


