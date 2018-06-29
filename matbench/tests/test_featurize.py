# coding: utf-8

import unittest

from matbench.data.load import load_double_perovskites_gap
from matbench.featurize import Featurize


class TestFeaturize(unittest.TestCase):

    def test_featurize(self, limit=10):
        df_init = load_double_perovskites_gap(return_lumo=False)[:limit]
        ignore_cols = ['a_1', 'a_2', 'b_1', 'b_2']
        featurizer = Featurize(df_init,
                               ignore_cols=ignore_cols,
                               ignore_errors=False)
        df = featurizer.featurize_columns()
        self.assertTrue("composition" in df)
        self.assertTrue(len(df), limit)
        self.assertGreaterEqual(len(df.columns), 90)
        self.assertTrue(featurizer.df.equals(df_init.drop(ignore_cols,axis=1)))

        # making sure featurize_formula works with only composition
        df_init = df_init.drop('formula', axis=1)
        df_init["composition"] = df["composition"]
        df = featurizer.featurize_formula(df_init, featurizers='all')
        self.assertGreaterEqual(len(df.columns), 90)



if __name__ == '__main__':
    unittest.main()