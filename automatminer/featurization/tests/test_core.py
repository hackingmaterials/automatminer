import os
import copy
import unittest

import pandas as pd
from pymatgen import Composition
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.datasets.dataset_retrieval import load_dataset
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import GlobalSymmetryFeatures, \
    DensityFeatures

from automatminer.featurization.core import AutoFeaturizer
from automatminer.featurization.sets import StructureFeaturizers, \
    CompositionFeaturizers

TEST_DIR = os.path.dirname(__file__)
CACHE_FILE = "cache_test.json"
CACHE_PATH = os.path.join(TEST_DIR, CACHE_FILE)

__author__ = ["Alex Dunn <ardunn@lbl.gov>",
              "Alireza Faghaninia <alireza@lbl.gov>",
              "Qi Wang <wqthu11@gmail.com>"]


class TestAutoFeaturizer(unittest.TestCase):
    def setUp(self):
        self.test_df = load_dataset('elastic_tensor_2015').rename(
            columns={"formula": "composition"})
        self.limit = 5

    def test_sanity(self):
        df = copy.copy(self.test_df)
        # sanity checks
        self.assertTrue(df['composition'].iloc[0], "Nb4CoSi")
        self.assertTrue(df["composition"].iloc[1179], "Al2Cu")
        self.assertEqual(df.shape[0], 1181)

    def test_featurize_composition(self):
        """
        Test automatic featurization while only considering formula/composition.
        """
        target = "K_VRH"

        # When compositions are strings
        df = copy.copy(self.test_df[['composition', target]].iloc[:self.limit])
        af = AutoFeaturizer(preset="fast")
        df = af.fit_transform(df, target)
        self.assertAlmostEqual(df["frac f valence electrons"].iloc[2],
                               0.5384615384615384)
        self.assertEqual(df["LUMO_element"].iloc[0], "Nb")
        self.assertTrue("composition" not in df.columns)

        # When compositions are Composition objects
        df = self.test_df[["composition", target]].iloc[:self.limit]
        df["composition"] = [Composition(s) for s in df["composition"]]
        af = AutoFeaturizer(preset="fast")
        df = af.fit_transform(df, target)
        self.assertAlmostEqual(df["frac f valence electrons"].iloc[2],
                               0.5384615384615384)
        self.assertEqual(df["LUMO_element"].iloc[0], "Nb")
        self.assertTrue("composition" not in df.columns)

    def test_featurize_structure(self):
        """
        Test automatic featurization while only considering structure.
        May automatically infer composition features.
        """
        target = "K_VRH"

        # When structures are Structure objects
        df = copy.copy(self.test_df[['structure', target]].iloc[:self.limit])
        af = AutoFeaturizer(preset="fast")
        df = af.fit_transform(df, target)
        # Ensure there are some structure features created
        self.assertTrue("vpa" in df.columns)
        # Ensure that composition features are automatically added without
        # explicit column
        self.assertTrue("HOMO_character" in df.columns)
        self.assertTrue("composition" not in df.columns)
        self.assertTrue("structure" not in df.columns)

        # When structures are dictionaries
        df = copy.copy(self.test_df[['structure', target]].iloc[:self.limit])
        df["structure"] = [s.as_dict() for s in df["structure"]]
        af = AutoFeaturizer(preset="fast")
        df = af.fit_transform(df, target)
        # Ensure there are some structure features created
        self.assertTrue("vpa" in df.columns)
        # Ensure that composition features are automatically added without
        # explicit column
        self.assertTrue("HOMO_character" in df.columns)
        self.assertTrue("composition" not in df.columns)
        self.assertTrue("structure" not in df.columns)

    def test_featurizers_by_users(self):
        df = copy.copy(self.test_df.iloc[:self.limit])
        target = "K_VRH"

        dn = DensityFeatures()
        gsf = GlobalSymmetryFeatures()
        featurizers = {"structure": [dn, gsf]}

        af = AutoFeaturizer(featurizers=featurizers)
        df = af.fit_transform(df, target)

        # Ensure that the featurizers are not set automatically, metaselection
        # is not used, exclude is None and featurizers not passed by the users
        # are not used.
        self.assertFalse(af.auto_featurizer)
        self.assertTrue(af.exclude == [])
        self.assertIn(dn, af.featurizers["structure"])
        self.assertIn(gsf, af.featurizers["structure"])
        ep = ElementProperty.from_preset("matminer")
        ep_feats = ep.feature_labels()
        self.assertFalse(any([f in df.columns for f in ep_feats]))

    def test_exclude_by_users(self):
        """
        Test custom args for featurizers to use.
        """
        df = copy.copy(self.test_df.iloc[:self.limit])
        target = "K_VRH"
        exclude = ["ElementProperty"]

        ep = ElementProperty.from_preset("matminer")
        ep_feats = ep.feature_labels()

        # Test to make sure excluded does not show up
        af = AutoFeaturizer(exclude=exclude, preset="fast")
        af.fit(df, target)
        df = af.fit_transform(df, target)

        self.assertTrue(af.auto_featurizer)
        self.assertIn("ElementProperty", af.exclude)
        self.assertFalse(any([f in df.columns for f in ep_feats]))

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
        target = "color"
        df_bsdos_pickled = "mp_data_with_dos_bandstructure.pickle"
        if refresh_df_init:
            mpdr = MPDataRetrieval()
            df = mpdr.get_dataframe(criteria={"material_id": "mp-149"},
                                    properties=["pretty_formula",
                                                "dos",
                                                "bandstructure",
                                                "bandstructure_uniform"]
                                    )
            df.to_pickle(os.path.join(TEST_DIR, df_bsdos_pickled))
        else:
            df = pd.read_pickle(os.path.join(TEST_DIR, df_bsdos_pickled))
        df = df.dropna(axis=0)
        df = df.rename(columns={"bandstructure_uniform": "bandstructure",
                                "bandstructure": "line bandstructure"})
        df[target] = [["red"]]
        n_cols_init = df.shape[1]

        featurizer = AutoFeaturizer(preset="fast",
                                    ignore_errors=False,
                                    multiindex=False)
        df = featurizer.fit_transform(df, target)

        # sanity checks
        self.assertTrue(len(df), limit)
        self.assertGreater(len(df.columns), n_cols_init)

        # DOSFeaturizer:
        self.assertEqual(df["cbm_character_1"][0], "p")

        # DopingFermi:
        self.assertAlmostEqual(df["fermi_c1e+20T300"][0], -0.539, 3)

        # Hybridization:
        self.assertAlmostEqual(df["vbm_sp"][0], 0.181, 3)
        self.assertAlmostEqual(df["cbm_s"][0], 0.4416, 3)
        self.assertAlmostEqual(df["cbm_sp"][0], 0.9864, 3)

        # BandFeaturizer:
        self.assertAlmostEqual(df["direct_gap"][0], 2.556, 3)
        self.assertAlmostEqual(df["n_ex1_norm"][0], 0.6285, 4)

        # BranchPointEnergy:
        self.assertAlmostEqual(df["branch_point_energy"][0], 5.7677, 4)

    def test_presets(self):
        target = "K_VRH"
        df = copy.copy(self.test_df.iloc[:self.limit])
        af = AutoFeaturizer(preset="fast")
        df = af.fit_transform(df, target)
        known_feats = CompositionFeaturizers().fast + \
                      StructureFeaturizers().fast
        n_structure_featurizers = len(af.featurizers["structure"])
        n_composition_featurizers = len(af.featurizers["composition"])
        n_featurizers = n_structure_featurizers + n_composition_featurizers
        self.assertEqual(n_featurizers, len(known_feats))

    def test_transferability(self):
        """
        Test that an autofeaturizer object is able to be fit on one dataset
        and applied to another.
        """
        target = "K_VRH"
        cols = ["composition", target]
        df1 = self.test_df[cols].iloc[:self.limit]
        df2 = self.test_df[cols].iloc[-1 * self.limit:]

        af = AutoFeaturizer(preset="fast")
        af.fit(df1, target)

        df2 = af.transform(df2, target)
        self.assertAlmostEqual(df2[target].iloc[0], 111.788114, places=5)
        self.assertAlmostEqual(df2["PymatgenData minimum X"].iloc[1], 1.36,
                               places=2)

    def test_column_attr(self):
        """
        Test that the autofeaturizer object correctly takes in composition_col,
        structure_col, bandstruct_col, and dos_col, and checks that
        fit_and_transform()
        works correctly with the attributes.
        """

        # Modification of test_featurize_composition with AutoFeaturizer parameter
        target = "K_VRH"
        df = copy.copy(self.test_df[['composition', target]].iloc[:self.limit])
        af = AutoFeaturizer(composition_col="composition", preset="best", ignore_errors=False)
        df = af.fit_transform(df, target)

        self.assertEqual(df["LUMO_element"].iloc[0], "Nb")
        self.assertTrue("composition" not in df.columns)

        df = self.test_df[["composition", target]].iloc[:self.limit]
        df["composition"] = [Composition(s) for s in df["composition"]]
        af = AutoFeaturizer(composition_col="composition", preset="best")
        df = af.fit_transform(df, target)
        self.assertEqual(df["LUMO_element"].iloc[0], "Nb")
        self.assertTrue("composition" not in df.columns)

        # Modification of test_featurize_structure with AutoFeaturizer parameter
        target = "K_VRH"
        df = copy.copy(self.test_df[['structure', target]].iloc[:self.limit])
        af = AutoFeaturizer(structure_col="structure", preset="fast")
        df = af.fit_transform(df, target)
        self.assertTrue("vpa" in df.columns)
        self.assertTrue("HOMO_character" in df.columns)
        self.assertTrue("composition" not in df.columns)
        self.assertTrue("structure" not in df.columns)

        df = copy.copy(self.test_df[['structure', target]].iloc[:self.limit])
        df["structure"] = [s.as_dict() for s in df["structure"]]
        af = AutoFeaturizer(structure_col="structure", preset="fast")
        df = af.fit_transform(df, target)
        self.assertTrue("vpa" in df.columns)
        self.assertTrue("HOMO_character" in df.columns)
        self.assertTrue("composition" not in df.columns)
        self.assertTrue("structure" not in df.columns)

    def test_functionalization(self):
        target = "K_VRH"
        flimit = 2
        df = self.test_df[['composition', target]].iloc[:flimit]
        af = AutoFeaturizer(functionalize=True, preset="fast")
        df = af.fit_transform(df, target)
        self.assertTupleEqual(df.shape, (flimit, 16848))

    def test_StructureFeaturizers_needs_fitting(self):
        fset_nofit = StructureFeaturizers().best
        fset_needfit = StructureFeaturizers().all
        af_nofit = AutoFeaturizer(featurizers={"structure": fset_nofit})
        af_needfit = AutoFeaturizer(featurizers={"structure": fset_needfit})
        self.assertTrue(af_needfit.needs_fit)
        self.assertFalse(af_nofit.needs_fit)

    def test_caching(self):
        target = "G_VRH"

        self.assertFalse(os.path.exists(CACHE_PATH))
        af = AutoFeaturizer(cache_src=CACHE_PATH, preset="fast")
        df = self.test_df[['composition', target]].iloc[:10]
        df_feats = af.fit_transform(df, target)
        self.assertTrue(os.path.exists(CACHE_PATH))

        df_cache = self.test_df[['composition', target]].iloc[:10]
        df_cache_feats = af.fit_transform(df_cache, target)
        self.assertAlmostEqual(df_feats.iloc[3, 0].tolist(),
                               df_cache_feats.iloc[3, 0].tolist())

    def test_prechecking(self):
        target = "K_VRH"
        af = AutoFeaturizer(preset="best")
        df = self.test_df[['composition', target]]
        af.fit(df, target)
        classes = [f.__class__.__name__ for f in af.featurizers["composition"]]
        self.assertNotIn("YangSolidSolution", classes)
        self.assertNotIn("Miedema", classes)
        self.assertIn("ElementProperty", classes)

    def tearDown(self):
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)


if __name__ == '__main__':
    unittest.main()
