# coding: utf-8
import os
import unittest

import pandas as pd
from pymatgen import Composition
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.datasets.dataset_retrieval import load_dataset
from matminer.featurizers.composition import ElectronAffinity, ElementProperty, \
    AtomicOrbitals
from matminer.featurizers.structure import GlobalSymmetryFeatures, \
    DensityFeatures

from matbench.featurization.core import AutoFeaturizer

test_dir = os.path.dirname(__file__)

__author__ = ["Alex Dunn <ardunn@lbl.gov>",
              "Alireza Faghaninia <alireza@lbl.gov>"]


class TestAutoFeaturizer(unittest.TestCase):

    def setUp(self, limit=5):
        self.test_df = load_dataset('elastic_tensor_2015').rename(
            columns={"formula": "composition"})
        self.limit = limit

    def test_sanity(self):
        df = self.test_df
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
        df = self.test_df[['composition', target]].iloc[:self.limit]
        af = AutoFeaturizer()
        df = af.fit_transform(df, target)
        self.assertAlmostEqual(df["frac f valence electrons"].iloc[2],
                               0.5384615384615384)
        self.assertEqual(df["LUMO_element"].iloc[0], "Nb")
        self.assertTrue("composition" not in df.columns)

        # When compositions are Composition objects
        df = self.test_df[["composition", target]].iloc[:self.limit]
        df["composition"] = [Composition(s) for s in df["composition"]]
        af = AutoFeaturizer()
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
        df = self.test_df[['structure', target]].iloc[:self.limit]
        af = AutoFeaturizer()
        df = af.fit_transform(df, target)
        # Ensure there are some structure features created
        self.assertTrue("dimensionality" in df.columns)
        # Ensure that composition features are automatically added without
        # explicit column
        self.assertTrue("HOMO_character" in df.columns)
        self.assertTrue("composition" not in df.columns)
        self.assertTrue("structure" not in df.columns)

        # When structures are dictionaries
        df = self.test_df[['structure', target]].iloc[:self.limit]
        df["structure"] = [s.as_dict() for s in df["structure"]]
        af = AutoFeaturizer()
        df = af.fit_transform(df, target)
        # Ensure there are some structure features created
        self.assertTrue("dimensionality" in df.columns)
        # Ensure that composition features are automatically added without
        # explicit column
        self.assertTrue("HOMO_character" in df.columns)
        self.assertTrue("composition" not in df.columns)
        self.assertTrue("structure" not in df.columns)

    def test_exclusions(self):
        """
        Test custom args for featurizers to use.
        """
        df = self.test_df[:self.limit]
        target = "K_VRH"
        exclude = ["ElementProperty"]

        dn = DensityFeatures()
        gsf = GlobalSymmetryFeatures()
        ep = ElementProperty.from_preset("matminer")
        ef = ElectronAffinity()
        ao = AtomicOrbitals()
        ep_feats = ep.feature_labels()
        ef_feats = ef.feature_labels()
        ao_feats = ao.feature_labels()

        # Test to make sure excluded does not show up
        af = AutoFeaturizer(exclude=exclude)
        af.fit(df, target)
        print(af.features)
        df = af.fit_transform(df, target)
        self.assertFalse(any([f in df.columns for f in ep_feats]))

        # Test to make sure composition features are not automatically
        # created if explicitly not included in featurizers dict
        # Also make sure no errors when excluding non-present featurizers
        df = self.test_df[:self.limit]
        featurizers = {"structure": [gsf, dn]}
        af = AutoFeaturizer(featurizers=featurizers)
        df = af.fit_transform(df, target)
        for flabels in [ep_feats, ef_feats, ao_feats]:
            self.assertFalse(any([f in df.columns for f in flabels]))

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
            df.to_pickle(os.path.join(test_dir, df_bsdos_pickled))
        else:
            df = pd.read_pickle(os.path.join(test_dir, df_bsdos_pickled))
        df = df.dropna(axis=0)
        df = df.rename(columns={"bandstructure_uniform": "bandstructure",
                                "bandstructure": "line bandstructure"})
        df[target] = [["red"]]
        n_cols_init = df.shape[1]

        featurizer = AutoFeaturizer(ignore_errors=False, multiindex=False)
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

    def test_transferability(self):
        """
        Test that an autofeaturizer object is able to be fit on one dataset
        and applied to another.
        """
        target = "K_VRH"
        cols = ["composition", target]
        df1 = self.test_df[cols].iloc[:self.limit]
        df2 = self.test_df[cols].iloc[-1 * self.limit:]

        af = AutoFeaturizer()
        af.fit(df1, target)

        df2 = af.transform(df2, target)
        print(df2)
        self.assertAlmostEqual(df2[target].iloc[0], 111.788114, places=5)
        self.assertAlmostEqual(df2["minimum X"].iloc[1], 1.36, places=2)


if __name__ == '__main__':
    unittest.main()
