import unittest

from matbench.data.load import load_castelli_perovskites, load_glass_binary
from matbench.featurization.metaselection.core import _composition_metafeatures, \
    _structure_metafeatures, dataset_metafeatures, FeaturizerAutoFilter
from pymatgen.core.structure import Structure


class TestDatasetMetaFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_glass = load_glass_binary().rename(
            columns={"formula": "composition"})
        cls.df_castelli = load_castelli_perovskites()
        cls.df_castelli["structure"] = cls.df_castelli["structure"].\
            apply(lambda x: Structure.from_dict(x))

    def test_formula_metafeatures(self):
        mfs = _composition_metafeatures(self.df_glass)
        mfs_values = mfs["formula_metafeatures"]
        self.assertEqual(mfs_values["number_of_formulas"], 5959)
        self.assertAlmostEqual(mfs_values["percent_of_all_metal"], 0.6578, 4)
        self.assertAlmostEqual(
            mfs_values["percent_of_metal_nonmetal"], 0.3208, 4)
        self.assertAlmostEqual(mfs_values["percent_of_all_nonmetal"], 0.0214, 4)
        self.assertAlmostEqual(
            mfs_values["percent_of_contain_trans_metal"], 0.6877, 4)
        self.assertEqual(mfs_values["number_of_different_elements"], 38)
        self.assertAlmostEqual(mfs_values["avg_number_of_elements"], 1.9802, 4)
        self.assertEqual(mfs_values["max_number_of_elements"], 2)
        self.assertEqual(mfs_values["min_number_of_elements"], 1)

    def test_structure_metafeatures(self):
        mfs = _structure_metafeatures(self.df_castelli)
        mfs_values = mfs["structure_metafeatures"]
        self.assertEqual(mfs_values["number_of_structures"], 18928)
        self.assertAlmostEqual(mfs_values["percent_of_ordered_structures"], 1.0)
        self.assertAlmostEqual(mfs_values["avg_number_of_sites"], 5.0)
        self.assertEqual(mfs_values["max_number_of_sites"], 5)
        self.assertEqual(
            mfs_values["number_of_different_elements_in_structures"], 56)

    def test_auto_metafeatures(self):
        mfs = dataset_metafeatures(self.df_glass)
        self.assertIn("formula_metafeatures", mfs.keys())
        self.assertIn("structure_metafeatures", mfs.keys())
        self.assertIsNone(mfs["structure_metafeatures"])


class TestFeaturizerAutoFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_glass = load_glass_binary()
        cls.df_castelli = load_castelli_perovskites()
        cls.df_castelli["structure"] = cls.df_castelli["structure"].\
            apply(lambda x: Structure.from_dict(x))

    def test_auto_excludes(self):
        glass_ftz_excludes = \
            FeaturizerAutoFilter(max_na_percent=0.05).\
                auto_excludes(self.df_glass)
        self.assertIn("IonProperty", glass_ftz_excludes)
        self.assertIn("ElectronAffinity", glass_ftz_excludes)
        self.assertIn("ElectronegativityDiff", glass_ftz_excludes)
        self.assertIn("TMetalFraction", glass_ftz_excludes)
        self.assertIn("YangSolidSolution", glass_ftz_excludes)
        self.assertIn("CationProperty", glass_ftz_excludes)
        self.assertIn("Miedema", glass_ftz_excludes)

        glass_ftz_excludes = \
            FeaturizerAutoFilter(max_na_percent=0.40).\
                auto_excludes(self.df_glass)
        self.assertIn("IonProperty", glass_ftz_excludes)
        self.assertIn("ElectronAffinity", glass_ftz_excludes)
        self.assertIn("ElectronegativityDiff", glass_ftz_excludes)
        self.assertIn("OxidationStates", glass_ftz_excludes)
        self.assertIn("CationProperty", glass_ftz_excludes)

        castelli_ftz_excludes = \
            FeaturizerAutoFilter(max_na_percent=0.05).\
                auto_excludes(self.df_castelli)
        self.assertIn("YangSolidSolution", castelli_ftz_excludes)
        self.assertIn("Miedema", castelli_ftz_excludes)
        self.assertIn("TMetalFraction", castelli_ftz_excludes)


if __name__ == "__main__":
    unittest.main()
