# coding: utf-8
import inspect
import os
import pandas as pd
import unittest

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.featurizers.base import BaseFeaturizer
import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
import matminer.featurizers.dos as dosf
import matminer.featurizers.bandstructure as bf


from matbench.data.load import load_double_perovskites_gap, \
    load_castelli_perovskites
from matbench.featurization.core import Featurization
from matbench.featurization.sets import AllFeaturizers
from matbench.data.load import load_phonon_dielectric_mp

test_dir = os.path.dirname(__file__)


class TestFeaturize(unittest.TestCase):

    def test_featurize_formula(self, limit=5):
        df_init = load_double_perovskites_gap(return_lumo=False)[:limit]
        ignore_cols = ['a_1', 'a_2', 'b_1', 'b_2']
        featurizer = Featurization(ignore_cols=ignore_cols,
                               ignore_errors=False,
                               exclude=['CohesiveEnergy'],
                               multiindex=False)

        df = featurizer.featurize_formula(df_init,
                                          featurizers="all",
                                          compcol=None,
                                          asindex=False,
                                          guess_oxidstates=True)

        # sanity checks
        self.assertTrue(len(df), limit)
        self.assertGreaterEqual(len(df.columns), 70)
        self.assertAlmostEqual(
            df[df["formula"] == "AgNbSnTiO6"]["gap gllbsc"].values[0], 2.881, 3)

        # BandCenter:
        self.assertAlmostEqual(
            df[df["formula"] == "AgNbSnTiO6"]["band center"].values[0], -2.623,
            3)

        # AtomicOrbitals:
        self.assertAlmostEqual(
            df[df["formula"] == "AgNbSnTiO6"]["gap_AO"].values[0], 0.129, 3)

        # Stoichiometry:
        self.assertTrue((df["0-norm"].isin([4, 5])).all())  # all 4- or 5-specie
        self.assertTrue((df["2-norm"] < 1).all())

        # ValenceOrbital:
        self.assertAlmostEqual(
            df[df["formula"] == "AgNbLaAlO6"][
                "frac p valence electrons"].values[0], 0.431, 3)

        # TMetalFraction:
        self.assertTrue((df["transition metal fraction"] < 0.45).all())

        # YangSolidSolution:
        self.assertAlmostEqual(
            df[df["formula"] == "AgNbSnTiO6"]["Yang delta"].values[0], 0.416, 3)

        # ElectronegativityDiff:
        self.assertAlmostEqual(
            df[df["formula"] == "AgNbLaGaO6"]["std_dev EN difference"].values[
                0], 0.366, 3)

        # making sure:
        # featurize_formula works with only composition and not formula
        # featurize_formula works with a given list of featurizers
        df = featurizer.featurize_formula(df_init, featurizers=[
            cf.ElementProperty.from_preset(preset_name="matminer"),
            cf.IonProperty()
        ])
        self.assertGreaterEqual(len(df.columns), 69)

    def test_featurize_structure(self, limit=5):
        df_init = load_castelli_perovskites()[:limit]
        featurizer = Featurization(ignore_errors=False, multiindex=False)
        df = featurizer.featurize_structure(df_init, inplace=False,
                                            featurizers="all")

        # sanity checks
        self.assertTrue(len(df), limit)
        self.assertGreater(len(df.columns), len(df_init.columns))

        # DensityFeatures:
        self.assertTrue((df["packing fraction"] < 0.45).all())
        self.assertAlmostEqual(
            df[df["formula"] == "RhTeN3"]["density"].values[0], 7.3176, 4)

        # GlobalSymmetryFeatures:
        self.assertEqual(
            df[df["formula"] == "HfTeO3"]["spacegroup_num"].values[0], 99)
        self.assertEqual(
            df[df["formula"] == "HfTeO3"]["crystal_system"].values[0],
            "tetragonal")
        self.assertTrue(
            not df[df["formula"] == "HfTeO3"]["is_centrosymmetric"].values[0])

        # Dimensionality:
        self.assertEqual(
            df[df["formula"] == "ReAsO2F"]["dimensionality"].values[0], 2)

        # RadialDistributionFunction:
        # TODO: add this test after it returns numbers and not dict!

        # TODO: add tests for the following once they return features not matrixes:
        # CoulombMatrix, SineCoulombMatrix, OrbitalFieldMatrix, MinimumRelativeDistances

        # TODO what are the other presets for SiteStatsFingerprint? need implementation and test?
        # SiteStatsFingerprint with preset==CrystalNNFingerprint_ops:
        self.assertEqual(
            df[df["formula"] == "RhTeN3"]["mean wt CN_1"].values[0], 0)
        self.assertAlmostEqual(
            df[df["formula"] == "RhTeN3"]["mean wt CN_2"].values[0], 0.412, 3)

        # EwaldEnergy:
        self.assertAlmostEqual(
            df[df["formula"] == "RhTeN3"]["ewald_energy"].values[0], -405.64, 2)
        self.assertEqual(
            df[df["formula"] == "HfTeO3"]["ewald_energy"].values[0], 0.0)

        # StructuralHeterogeneity:
        self.assertAlmostEqual(
            df[df["formula"] == "RhTeN3"]["min relative bond length"].values[0],
            0.7896, 4)
        self.assertAlmostEqual(
            df[df["formula"] == "RhTeN3"][
                "maximum neighbor distance variation"].values[0],
            0.1224, 4)

        # MaximumPackingEfficiency:
        self.assertAlmostEqual(
            df[df["formula"] == "ReAsO2F"]["max packing efficiency"].values[0],
            0.295, 3)

        # ChemicalOrdering:
        self.assertAlmostEqual(
            df[df["formula"] == "HfTeO3"][
                "mean ordering parameter shell 1"].values[0], 0.599, 3)
        # TODO: umm, make a PR for shorter feature_labels for some structure featurizers?

        # XRDPowderPattern:
        self.assertAlmostEqual(
            df[df["formula"] == "BiHfO2F"]["xrd_127"].values[0], 0.0011, 4)

        # BondFractions:
        self.assertAlmostEqual(
            df[df["formula"] == "WReO2S"]["S2- - W3+ bond frac."].values[0],
            0.1667, 4)

        # BagofBonds:
        self.assertAlmostEqual(
            df[df["formula"] == "HfTeO3"]["O - O bond #1"].values[0], 11.1658,
            4)

    def test_featurize_bsdos(self, refresh_df_init=False, limit=1):
        """
        Tests featurize_dos and featurize_bandstructure.

        Args:
            refresh_df_init (bool): for developers, if the test need to be
                updated set to True. Otherwise set to False to make the final
                test independent of MPRester and faster.
            limit (int): the maximum final number of entries.

        Returns (None):
        """
        df_bsdos_pickled = "mp_data_with_dos_bandstructure.pickle"
        if refresh_df_init:
            mpdr = MPDataRetrieval()
            df_init = mpdr.get_dataframe(criteria={"material_id": "mp-149"},
                                         properties=["pretty_formula",
                                                     "dos",
                                                     "bandstructure",
                                                     "bandstructure_uniform"]
                                         )
            df_init.to_pickle(os.path.join(test_dir, df_bsdos_pickled))
        else:
            df_init = pd.read_pickle(os.path.join(test_dir, df_bsdos_pickled))
        df_init = df_init.dropna(axis=0)
        featurizer = Featurization(ignore_errors=False, multiindex=False)
        df = featurizer.featurize_dos(df_init, inplace=False)

        # sanity checks
        self.assertTrue(len(df), limit)
        self.assertGreater(len(df.columns), len(df_init.columns))

        # DOSFeaturizer:
        self.assertEqual(df["cbm_character_1"][0], "p")

        # DopingFermi:
        self.assertAlmostEqual(df["fermi_c1e+20T300"][0], -0.539, 3)

        # Hybridization:
        self.assertAlmostEqual(df["vbm_sp"][0], 0.181, 3)
        self.assertAlmostEqual(df["cbm_s"][0], 0.4416, 3)
        self.assertAlmostEqual(df["cbm_sp"][0], 0.9864, 3)

        df = featurizer.featurize_bandstructure(df_init,
                                                inplace=False,
                                                col_id="bandstructure_uniform")
        # sanity checks
        self.assertTrue("bandstructure" in df)
        self.assertGreater(len(df.columns), len(df_init.columns))

        # BandFeaturizer:
        self.assertAlmostEqual(df["direct_gap"][0], 2.556, 3)
        self.assertAlmostEqual(df["n_ex1_norm"][0], 0.6285, 4)

        # BranchPointEnergy:
        self.assertAlmostEqual(df["branch_point_energy"][0], 5.7677, 4)

    def test_auto_featurize(self, limit=5):
        df_init = load_phonon_dielectric_mp()[:limit]
        print(df_init.structure)
        featurizer = Featurization(ignore_errors=False, multiindex=True)
        df = featurizer.auto_featurize(df_init,
                                       input_cols=('formula', 'structure'))

        # sanity checks
        self.assertTrue(len(df), limit)
        self.assertGreater(len(df.columns), len(df_init.columns))

        # DensityFeatures:
        self.assertEqual(df[('ElementProperty', 'mode SpaceGroupNumber')][1],
                         194)

        self.assertAlmostEqual(  # making sure structure is also featurized
            df[('SiteStatsFingerprint', 'mean trigonal pyramidal CN_4')][1],
            0.243648, 3)
        self.assertEqual(df.index.name, ('Input Data', 'formula'))


class TestAllFeaturizers(unittest.TestCase):
    """
    Class to ensure the featurizers available in featurizer files in matminer
    match exactly to those defined to AllFeaturizers class. This test is meant
    to catch events when a new featurizer is defined but not listed inside
    AllFeaturizers (e.g. by mistake).
    """

    def setUp(self):
        self.allfs = AllFeaturizers()

    @staticmethod
    def get_featurizers(module, ignore=None):
        """Get a list of featurizers class names defined in a module.

        Args:
            module (module): The python module.
            ignore (`list` of `str`, optional): A list of class names to ignore.

        Returns:
            (`list` of `str`): List of all featurizer class names.
        """
        ignore = ignore if ignore else []

        def is_featurizer(class_object):
            return (issubclass(class_object, BaseFeaturizer)
                    and not class_object == BaseFeaturizer)

        # getmembers returns list of (class_name, class_object)
        classes = [n for n, c in inspect.getmembers(module, inspect.isclass)
                   if is_featurizer(c) and c.__module__ == module.__name__]
        featurizers = [c for c in classes if c not in ignore]
        return featurizers

    def _test_features_implemented(self, test_feats, true_feats):
        """Check two lists of featurizers are the same.

        Note that `test_feats` is a list of objects and `true_feats` is a
        list of class names as strings.
        """
        test_feats = [c.__class__.__name__ for c in test_feats]

        for featurizer_name in true_feats:
            self.assertTrue(featurizer_name in test_feats,
                            ("{} matminer featurizer not in implemented in "
                             "matbench").format(featurizer_name))

        self.assertEqual(len(test_feats), len(true_feats))

    def test_composition_featurizers(self):
        true_feats = TestAllFeaturizers.get_featurizers(cf)
        test_feats = self.allfs.composition
        self._test_features_implemented(test_feats, true_feats)

    def test_structure_featurizers(self):
        true_feats = self.get_featurizers(sf, ['StructureComposition'])
        test_feats = self.allfs.structure
        self._test_features_implemented(test_feats, true_feats)

    def test_dos_featurizers(self):
        true_feats = self.get_featurizers(dosf)
        test_feats = self.allfs.dos
        self._test_features_implemented(test_feats, true_feats)

    def test_bandstructure_featurizers(self):
        true_feats = self.get_featurizers(bf)
        test_feats = self.allfs.bandstructure
        self._test_features_implemented(test_feats, true_feats)


if __name__ == '__main__':
    unittest.main()
