import unittest
from matbench.core.preprocess import Preprocess

from matminer.datasets.dataframe_loader import load_elastic_tensor
from matminer.featurizers.structure import GlobalSymmetryFeatures
from matbench.core.featurize import Featurize

class TestPreprocess(unittest.TestCase):

    # Todo: Add more tests
    def test_preprocess_basic(self):
        """
        A basic test ensuring Preprocess can handle numerical features and
        features/targets  that may be strings but should be numbers.

        Returns: None
        """
        df = load_elastic_tensor()[:5][['K_VRH', 'structure']]
        df['K_VRH'] = df['K_VRH'].astype(str)
        f = Featurize()
        df = f.featurize_structure(df, featurizers=[GlobalSymmetryFeatures()])

        p = Preprocess()
        df = p.preprocess(df, 'K_VRH')
        self.assertAlmostEqual(df['K_VRH'].iloc[0], 194.26888435900003)
        self.assertEqual(df["crystal_system_tetragonal"].iloc[0], 1)