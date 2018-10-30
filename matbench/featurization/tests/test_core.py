# coding: utf-8
import os
import pandas as pd
import unittest

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.datasets.dataset_retrieval import load_dataset

from matbench.featurization.core import AutoFeaturizer
from matbench.data.load import load_phonon_dielectric_mp

test_dir = os.path.dirname(__file__)


class TestAutoFeaturizer(unittest.TestCase):

    def setUp(self):
        self.test_df = load_dataset('elastic_tensor_2015').rename(columns={"formula": "composition"})

    def test_featurize_formula(self, limit=5):
        target = "K_VRH"
        df = self.test_df[['composition', target]]

        # sanity checks
        self.assertTrue(df['composition'].iloc[0], "Nb4CoSi")
        self.assertTrue(df["composition"].iloc[1179], "Al2Cu")
        self.assertEqual(df.shape[0], 1181)
        self.assertEqual(df.shape[1], 2)

        # test automatic featurization abilities with all defaults
        df = df.iloc[:limit]
        af = AutoFeaturizer()
        df = af.fit_transform(df, target)
        self.assertAlmostEqual(df["frac f valence electrons"].iloc[2], 0.5384615384615384)
        self.assertEqual(df["LUMO_element"].iloc[0], "Nb")


    def test_featurize_structure(self, limit=5):
        pass


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
        featurizer = AutoFeaturizer(ignore_errors=False, multiindex=False)
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
        featurizer = AutoFeaturizer(ignore_errors=False, multiindex=True)
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


if __name__ == '__main__':
    unittest.main()
