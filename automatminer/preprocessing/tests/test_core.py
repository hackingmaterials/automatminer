import os
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from automatminer.utils.log import AMM_LOGGER_BASENAME
from automatminer.preprocessing.core import DataCleaner, FeatureReducer
from automatminer.preprocessing.feature_selection import TreeFeatureReducer, \
    rebate, lower_corr_clf
from automatminer.utils.pkg import compare_columns

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

    def test_DataCleaner_sample_na_method(self):
        df = self.test_df
        df['HOMO_energy'].loc[40] = np.nan
        df['HOMO_energy'].loc[110] = np.nan

        # Test when transform method is fill
        dc = DataCleaner(max_na_frac=0.9,
                         feature_na_method="drop",
                         na_method_fit="drop",
                         na_method_transform="fill")
        dffit = df.loc[:100]
        fitted = dc.fit_transform(dffit, target=self.target)
        test_shape = tuple(np.subtract(dffit.shape, (1, 0)).tolist())
        self.assertTupleEqual(fitted.shape, test_shape)  # minus one sample

        dftrans = df.iloc[100:]
        tranz = dc.transform(dftrans, target=self.target)
        self.assertTupleEqual(tranz.shape, dftrans.shape)

        # Test when transform method is mean
        dc2 = DataCleaner(max_na_frac=0.9,
                          feature_na_method="drop",
                          na_method_fit="drop",
                          na_method_transform="mean")
        fitted = dc2.fit_transform(dffit, target=self.target)
        test_shape = tuple(np.subtract(dffit.shape, (1, 0)).tolist())
        self.assertTupleEqual(fitted.shape, test_shape)  # minus one sample

        dftrans = df.loc[100:]
        tranz = dc2.transform(dftrans, target=self.target)
        self.assertTupleEqual(tranz.shape, dftrans.shape)
        mean = dftrans.drop(110)["HOMO_energy"].mean()
        self.assertAlmostEqual(tranz["HOMO_energy"].loc[110], mean)

    def test_DataCleaner_feature_na_method(self):
        dc = DataCleaner(max_na_frac=0, feature_na_method="drop")
        df = self.test_df
        df['LUMO_energy'].iloc[40] = np.nan
        df['LUMO_energy'].iloc[110] = np.nan

        # Test normal dropping with transformation
        dffit = df.iloc[:100]
        fitted = dc.fit_transform(dffit, target=self.target)
        self.assertNotIn("LUMO_energy", fitted.columns)
        dftrans = df.iloc[100:]
        tranz = dc.transform(dftrans, target=self.target)
        self.assertNotIn("LUMO_energy", tranz.columns)

        # Test filling
        dc2 = DataCleaner(max_na_frac=0, feature_na_method="fill")
        fitted = dc2.fit_transform(dffit, target=self.target)
        true = fitted["LUMO_energy"].iloc[39]
        filled = fitted["LUMO_energy"].iloc[40]
        self.assertAlmostEqual(true, filled, places=10)
        self.assertTupleEqual((100, 417), fitted.shape)

        # Test mean
        dcmean = DataCleaner(max_na_frac=0, feature_na_method="mean")
        df['minimum X'].iloc[99] = np.nan
        minimum_x = dffit["minimum X"]
        mean_min_x = minimum_x[~minimum_x.isnull()].mean()
        fitted = dcmean.fit_transform(dffit, target=self.target)
        self.assertAlmostEqual(fitted["minimum X"].iloc[99], mean_min_x,
                               places=10)

    def test_DataCleaner_na_method_feature_sample_interaction(self):
        dc = DataCleaner(max_na_frac=0.01, feature_na_method="mean",
                         na_method_transform="fill", na_method_fit="fill")
        df = self.test_df
        # Should be dropped
        df["maximum X"] = [np.nan] * len(df)
        # Should be filled via mean
        df["range X"] = [np.nan] * 100 + df["range X"].iloc[100:].tolist()
        # Should be filled by 39
        df["minimum X"].iloc[40] = np.nan

        mean = df["range X"].loc[~df["range X"].isnull()].mean()
        df = dc.fit_transform(df, self.target)
        self.assertNotIn("maximum X", df.columns)
        self.assertIn("range X", df.columns)

        for r in df["range X"].iloc[:100]:
            self.assertAlmostEqual(r, mean, places=5)

        self.assertIn("minimum X", df.columns)
        self.assertEqual(df["minimum X"].iloc[40], df["minimum X"].iloc[39])

    def test_FeatureReducer_basic(self):
        fr = FeatureReducer(reducers=('corr', 'tree'))

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
        self.assertGreater(df.shape[1], df_reduced.shape[1])

    def test_FeatureReducer_pca(self):
        # Case where n_samples < n_features
        df = self.test_df.iloc[:10]
        fr = FeatureReducer(reducers=("pca",), n_pca_features='auto')
        df_reduced = fr.fit_transform(df, self.target)
        self.assertTupleEqual(df_reduced.shape, (10, 11))

        # Case where n_samples > n_features
        fsubset = ['HOMO_energy', 'LUMO_energy', 'gap_AO', 'minimum X',
                   'maximum X', 'range X', 'mean X', 'std_dev X', 'minimum row',
                   'maximum row', 'range row', 'mean row', 'std_dev row',
                   'minimum group', 'maximum group', 'range group',
                   'mean group', 'std_dev group']
        df = self.test_df[fsubset + [self.target]]
        df_reduced = fr.fit_transform(df, self.target)
        self.assertEqual(df_reduced.shape[0], 200)

        # Manually specified case of n_samples > n_features
        fr = FeatureReducer(reducers=('pca',), n_pca_features=0.5)
        df = self.test_df
        df_reduced = fr.fit_transform(df, self.target)
        self.assertTupleEqual(df_reduced.shape, (200, 201))

    def test_manual_feature_reduction(self):
        fr = FeatureReducer(reducers=[], remove_features=['LUMO_element_Th'])

        # ultra-basic case: are we reducing at least 1 feature?
        df = fr.fit_transform(self.test_df, self.target)
        self.assertTrue('LUMO_element_Th' not in df.columns)
        self.assertEqual(fr.removed_features['manual'], ['LUMO_element_Th'])

        # test removing feature that doesn't exist
        fr = FeatureReducer(reducers=[], remove_features=['abcdefg12345!!'])

        with self.assertLogs(AMM_LOGGER_BASENAME, level='INFO') as cm:
            # should give log warning but not throw an error
            fr.fit_transform(self.test_df, self.target)
            self.assertTrue('abcdefg12345!!' in " ".join(cm.output))

    def test_saving_feature_from_removal(self):
        fr = FeatureReducer(reducers=('corr',), keep_features=['maximum X'])

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
