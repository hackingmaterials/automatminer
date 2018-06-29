# coding: utf-8

import unittest

from matbench.data.load import load_double_perovskites_gap
from matbench.featurize import Featurize
from matminer.featurizers.composition import ElementProperty, IonProperty


class TestFeaturize(unittest.TestCase):

    def test_featurize(self, limit=10):
        df_init = load_double_perovskites_gap(return_lumo=False)[:limit]
        ignore_cols = ['a_1', 'a_2', 'b_1', 'b_2']
        featurizer = Featurize(df_init,
                               ignore_cols=ignore_cols,
                               ignore_errors=False)

        df = featurizer.featurize_formula()
        self.assertTrue("composition" in df)
        self.assertTrue(len(df), limit)
        self.assertGreaterEqual(len(df.columns), 90)
        self.assertTrue(featurizer.df.equals(df_init.drop(ignore_cols,axis=1)))
        self.assertAlmostEqual(
            df[df["formula"]=="AgNbSnTiO6"]["band center"].values[0],-2.623, 3)
        self.assertAlmostEqual(
            df[df["formula"]=="AgNbSnTiO6"]["gap gllbsc"].values[0], 2.881, 3)
        self.assertAlmostEqual(
            df[df["formula"]=="AgNbSnTiO6"]["gap_AO"].values[0], 0.129, 3)
        self.assertAlmostEqual(
            df[df["formula"]=="AgNbLaGaO6"]["std_dev EN difference"].values[0], 0.366, 3)

        # making sure:
            # featurize_formula works with only composition and not formula
            # eaturize_formula works with a given list of featurizers
        df_init = df_init.drop('formula', axis=1)
        df_init["composition"] = df["composition"]
        df = featurizer.featurize_formula(df_init, featurizers=[
            ElementProperty.from_preset(preset_name="matminer"),
            IonProperty()
        ])
        self.assertGreaterEqual(len(df.columns), 70)



if __name__ == '__main__':
    unittest.main()