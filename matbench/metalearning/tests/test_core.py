import unittest
from pymatgen.core.structure import Structure

from matbench.metalearning.core import DatasetMetaFeatures, FeaturizerAutoFilter
from matbench.data.load import load_castelli_perovskites, load_jdft2d, \
    load_glass_binary, load_mp


class TestDatasetMetaFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_glass = load_glass_binary()
        cls.df_castelli = load_castelli_perovskites()
        cls.df_castelli["structure"] = cls.df_castelli["structure"].\
            apply(lambda x: Structure.from_dict(x))

    def test_formula_metafeatures(self):
        mfs = DatasetMetaFeatures(self.df_glass).formula_metafeatures()
        mfs_values = mfs["formula_metafeatures"]
        self.assertEqual(mfs_values["number_of_formulas"], 5959)
        self.assertAlmostEqual(mfs_values["percent_of_all_metal"], 0.6578, 4)
        self.assertAlmostEqual(
            mfs_values["percent_of_metal_nonmetal"], 0.3208, 4)
        self.assertAlmostEqual(mfs_values["percent_of_all_nonmetal"], 0.0214, 4)
        self.assertEqual(mfs_values["number_of_different_elements"], 38)
        self.assertAlmostEqual(mfs_values["avg_number_of_elements"], 1.9802, 4)
        self.assertEqual(mfs_values["max_number_of_elements"], 2)
        self.assertEqual(mfs_values["min_number_of_elements"], 1)

    def test_structure_metafeatures(self):
        mfs = DatasetMetaFeatures(self.df_castelli).structure_metafeatures()
        mfs_values = mfs["structure_metafeatures"]
        self.assertEqual(mfs_values["number_of_structures"], 18928)
        self.assertAlmostEqual(mfs_values["percent_of_ordered_structures"], 1.0)
        self.assertAlmostEqual(mfs_values["avg_number_of_sites"], 5.0)
        self.assertEqual(mfs_values["max_number_of_sites"], 5)
        self.assertEqual(
            mfs_values["number_of_different_elements_in_structures"], 56)

