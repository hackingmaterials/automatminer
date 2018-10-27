import pandas as pd
import os
import unittest

from matminer.datasets import load_dataset
from matminer.featurizers.structure import GlobalSymmetryFeatures
from matbench.featurization.core import Featurization
# from matbench.preprocessing.core import Preprocesser
from matbench.preprocessing.feature_selection import TreeBasedFeatureReduction, rebate

test_dir = os.path.dirname(__file__)

# class TestPreprocess(unittest.TestCase):


    # Todo: Add more tests
    # def test_preprocess_basic(self):
    #     """
    #     A basic test ensuring Preprocess can handle numerical features and
    #     features/targets  that may be strings but should be numbers.
    #
    #     Returns: None
    #     """
    #     df = load_dataset('elastic_tensor_2015')[:15][['K_VRH', 'structure']]
    #     df['K_VRH'] = df['K_VRH'].astype(str)
    #     f = Featurization()
    #     df = f.featurize_structure(df, featurizers=[GlobalSymmetryFeatures()])
    #     p = Preprocesser()
    #     df = p.preprocess(df, 'K_VRH')
    #     self.assertAlmostEqual(df['K_VRH'].iloc[0], 194.26888435900003)
    #     self.assertEqual(df['crystal_system_tetragonal'].iloc[0], 1)


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


