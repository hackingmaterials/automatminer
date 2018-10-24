import unittest
from pymatgen.core import Structure

from matbench.metalearning.core import DatasetMetaFeatures, FeaturizerAutoFilter
from matbench.data.load import load_castelli_perovskites, load_jdft2d, \
    load_glass_binary, load_mp


class TestDatasetMetaFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_glass = load_glass_binary()

    def test_