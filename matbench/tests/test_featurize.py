# coding: utf-8

import unittest

from matbench.data.load import load_double_perovskites_gap, \
    load_castelli_perovskites
from matbench.featurize import Featurize
from matminer.featurizers.composition import ElementProperty, IonProperty


class TestFeaturize(unittest.TestCase):

    def test_featurize_formula(self, limit=5):
        df_init = load_double_perovskites_gap(return_lumo=False)[:limit]
        ignore_cols = ['a_1', 'a_2', 'b_1', 'b_2']
        featurizer = Featurize(df_init,
                               ignore_cols=ignore_cols,
                               ignore_errors=False)

        df = featurizer.featurize_formula()

        # sanity checks
        self.assertTrue("composition" in df)
        self.assertTrue(len(df), limit)
        self.assertGreaterEqual(len(df.columns), 70)
        self.assertTrue(featurizer.df.equals(df_init.drop(ignore_cols,axis=1)))
        self.assertAlmostEqual(
            df[df["formula"]=="AgNbSnTiO6"]["gap gllbsc"].values[0], 2.881, 3)

        # BandCenter:
        self.assertAlmostEqual(
            df[df["formula"]=="AgNbSnTiO6"]["band center"].values[0],-2.623, 3)

        # AtomicOrbitals:
        self.assertAlmostEqual(
            df[df["formula"]=="AgNbSnTiO6"]["gap_AO"].values[0], 0.129, 3)

        # Stoichiometry:
        self.assertTrue((df["0-norm"].isin([4, 5])).all()) # all 4- or 5-specie
        self.assertTrue((df["2-norm"] < 1).all())

        # ValenceOrbital:
        self.assertAlmostEqual(
            df[df["formula"]=="AgNbLaAlO6"]["frac p valence electrons"].values[0], 0.431, 3)

        # TMetalFraction:
        self.assertTrue((df["transition metal fraction"] < 0.45).all())

        # YangSolidSolution:
        self.assertAlmostEqual(
            df[df["formula"]=="AgNbSnTiO6"]["Yang delta"].values[0], 0.416, 3)

        # AtomicPackingEfficiency:
        self.assertTrue((df["mean abs simul. packing efficiency"] < 0.1).all())
        self.assertTrue((df["dist from 1 clusters |APE| < 0.010"] < 0.1).all())

        # ElectronegativityDiff:
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


    def test_featurize_structure(self, limit=5):
        df_init = load_castelli_perovskites()[:limit]
        featurizer = Featurize(df_init, ignore_errors=False)
        df = featurizer.featurize_structure(df_init, inplace=False)

        # sanity checks
        self.assertTrue("structure" in df)
        self.assertTrue(len(df), limit)
        self.assertGreater(len(df.columns), len(df_init.columns))
        self.assertTrue(featurizer.df.equals(df_init))

        # DensityFeatures:
        self.assertTrue((df["packing fraction"] < 0.45).all())
        self.assertAlmostEqual(
            df[df["formula"]=="RhTeN3"]["density"].values[0], 7.3176, 4)

        # GlobalSymmetryFeatures:
        self.assertEqual(df[df["formula"] == "HfTeO3"]["spacegroup_num"].values[0], 99)
        self.assertEqual(df[df["formula"] == "HfTeO3"]["crystal_system"].values[0], "tetragonal")
        self.assertTrue(not df[df["formula"] == "HfTeO3"]["is_centrosymmetric"].values[0])

        # Dimensionality
        self.assertEqual(df[df["formula"] == "ReAsO2F"]["dimensionality"].values[0], 2)



if __name__ == '__main__':
    unittest.main()