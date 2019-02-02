import os
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from automatminer.base import logger_base_name
from automatminer.preprocessing.core import DataCleaner, FeatureReducer
from automatminer.preprocessing.feature_selection import TreeFeatureReducer, \
    rebate, lower_corr_clf
from automatminer.utils.package_tools import compare_columns

test_dir = os.path.dirname(__file__)


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv(os.path.join(test_dir, 'test_featurized_df.csv'))
        self._test_df = df.drop(columns=["formula"])
        self.target = 'gap expt'

    @property
    def test_df(self):
        """
        Prevent any memory problems or accidental overwrites.

        Returns:
            (pd.DataFrame): A pandas deataframe deepcopy of the testing df.
        """
        return deepcopy(self._test_df)

    def test_DataCleaner(self):
        """
        A basic test ensuring Preprocess can handle numerical features and
        features/targets  that may be strings but should be numbers.

        Returns: None
        """
        df = self.test_df
        dc = DataCleaner()

        # Test the case of numbers as strings
        df[self.target] = df[self.target].astype(str)
        df = dc.fit_transform(df, self.target)
        self.assertAlmostEqual(df[self.target].iloc[0], 0.35)

        # Test if there is an nan in target
        df[self.target].iloc[8] = np.nan
        df = dc.fit_transform(df, self.target)
        self.assertEqual(df.shape[0], self.test_df.shape[0] - 1)

        # Test if there is an nan in feature
        df['HOMO_energy'].iloc[40] = np.nan
        df = dc.fit_transform(df, self.target)
        self.assertEqual(df.shape[0], self.test_df.shape[0] - 2)

        # Test if nan threshold is exceeded for a feature
        df["LUMO_energy"].iloc[:-2] = [np.nan] * (df.shape[0] - 2)
        df = dc.fit_transform(df, self.target)
        self.assertEqual(df.shape[1], self.test_df.shape[1] - 1)

        # test transferability
        df2 = self.test_df
        df2 = df2.drop(columns=[self.target])
        df2 = dc.transform(df2, self.target)
        self.assertFalse(compare_columns(df, df2,
                                         ignore=self.target)["mismatch"])
        self.assertTrue(self.target not in df2.columns)

    def test_FeatureReducer_basic(self):
        fr = FeatureReducer()

        # ultra-basic case: are we reducing at least 1 feature?
        df = fr.fit_transform(self.test_df, self.target)
        self.assertTrue(df.shape[1] < self.test_df.shape[1])

        # ensure metadata is being written correctly
        self.assertTrue(self.target not in fr.retained_features)
        self.assertTrue(len(list(fr.removed_features.keys())) == 2)

    def test_FeatureReducer_advanced(self):
        # ensure other combinations of feature reducers are working
        fr = FeatureReducer(reducers=('corr', 'rebate'), n_rebate_features=40)
        df = fr.fit_transform(self.test_df, self.target)
        self.assertEqual(df.shape[1], 41)  # 40 features + self.target
        self.assertTrue(self.target in df.columns)

    def test_FeatureReducer_transferability(self):
        # ensure the same thing works when fraction is used
        fr = FeatureReducer(reducers=('rebate',), n_rebate_features=0.2)
        df = fr.fit_transform(self.test_df, self.target)
        self.assertEqual(df.shape[1], 83 + 1)

        # test transferability
        df2 = deepcopy(self.test_df)
        df2 = fr.transform(df2, self.target)
        self.assertListEqual(df.columns.tolist(), df2.columns.tolist())

    def test_FeatureReducer_classification(self):
        # test classification with corr matrix (no errors)
        fr = FeatureReducer(reducers=('corr',))
        df_class = self.test_df
        df_class[self.target] = ["semiconductor" if 0.0 < g < 3.0 else
                                 "nonmetal" for g in df_class[self.target]]
        df_class = fr.fit_transform(df_class, self.target)
        self.assertEqual(df_class.shape[0], 200)

    def test_FeatureReducer_combinations(self):
        df = self.test_df
        fr = FeatureReducer(reducers=('pca', 'rebate', 'tree'))
        df_reduced = fr.fit_transform(df, self.target)
        self.assertEqual(df_reduced.shape[1], 15)

    def test_manual_feature_reduction(self):
        fr = FeatureReducer(reducers=[], remove_features=['LUMO_element_Th'])

        # ultra-basic case: are we reducing at least 1 feature?
        df = fr.fit_transform(self.test_df, self.target)
        self.assertTrue('LUMO_element_Th' not in df.columns)
        self.assertEqual(fr.removed_features['manual'], ['LUMO_element_Th'])

        # test removing feature that doesn't exist
        fr = FeatureReducer(reducers=[], remove_features=['abcdefg12345!!'])

        with self.assertLogs(logger_base_name, level='INFO') as cm:
            # should give log warning but not throw an error
            fr.fit_transform(self.test_df, self.target)
            self.assertTrue('abcdefg12345!!' in " ".join(cm.output))

    def test_saving_feature_from_removal(self):
        fr = FeatureReducer(keep_features=['maximum X'])

        # ultra-basic case: are we reducing at least 1 feature?
        df = fr.fit_transform(self.test_df, self.target)
        self.assertTrue('maximum X' in df.columns)


class TestFeatureReduction(unittest.TestCase):

    def setUp(self):
        df = pd.read_csv(os.path.join(test_dir, 'test_featurized_df.csv'))
        self.test_df = df.set_index('formula')
        self.random_state = 12

    def test_TreeBasedFeatureReduction(self):
        X = self.test_df.drop('gap expt', axis=1)
        y = self.test_df['gap expt']
        tbfr = TreeFeatureReducer(mode='regression',
                                  random_state=self.random_state)
        tbfr.fit(X, y, cv=3)
        self.assertEqual(len(tbfr.selected_features), 19)
        X_reduced = tbfr.transform(X)
        self.assertEqual(X_reduced.shape, (len(X), 19))
        self.assertTrue('HOMO_energy' in X_reduced.columns)

    def test_rebate(self):
        df_reduced = rebate(self.test_df, "gap expt", 10)
        self.assertEqual(df_reduced.shape[1], 10)

    def test_lower_corr_clf(self):
        df = self.test_df
        targets = []
        for gap in df["gap expt"]:
            if gap == 0:
                targets.append("metal")
            elif gap <= 3.0:
                targets.append("semiconductor")
            else:
                targets.append("insulator")
        df["gap_clf"] = targets
        # worst feature should be worse than a perfect output value...
        worse_feature = lower_corr_clf(df, "gap_clf", "gap expt", "range row")
        self.assertEqual("range row", worse_feature)
