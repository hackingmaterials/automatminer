import unittest

from matminer.datasets import load_dataset
from matminer.featurizers.structure import GlobalSymmetryFeatures
from matbench.featurization.core import Featurization

from matbench.preprocessing.core import Preprocessing


class TestPreprocess(unittest.TestCase):

    # Todo: Add more tests
    def test_preprocess_basic(self):
        """
        A basic test ensuring Preprocess can handle numerical features and
        features/targets  that may be strings but should be numbers.

        Returns: None
        """
        df = load_dataset("elastic_tensor_2015")[:5][['K_VRH', 'structure']]
        df['K_VRH'] = df['K_VRH'].astype(str)
        f = Featurization()
        df = f.featurize_structure(df, featurizers=[GlobalSymmetryFeatures()])

        p = Preprocessing()
        df = p.preprocess(df, 'K_VRH')
        self.assertAlmostEqual(df['K_VRH'].iloc[0], 194.26888435900003)
        self.assertEqual(df["crystal_system_tetragonal"].iloc[0], 1)