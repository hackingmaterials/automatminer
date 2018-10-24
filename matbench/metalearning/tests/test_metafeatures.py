import os
import unittest
from pymatgen.core import Structure

from matbench.metalearning.core import MetaFeatureValue
from matbench.metalearning import metafeatures
from matbench.data.load import load_castelli_perovskites, load_jdft2d, \
    load_glass_binary, load_mp


test_dir = os.path.dirname(__file__)


class TestCompositionMetafeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_glass = load_glass_binary()

        cls.metafeatures_glass = metafeatures.metafeatures
        cls.helpers_glass = metafeatures.helpers

        cls.helpers_glass.set_value(
            "formula_stats", cls.helpers_glass["formula_stats"]
            (cls.df_glass["formula"], cls.df_glass["gfa"]))

    def test_NumberOfFormulas(self):
        nf = self.metafeatures_glass["number_of_formulas"](
             self.df_glass["formula"], self.df_glass["gfa"])
        print(nf)
        self.assertEqual(nf.value, 5959)
        self.assertIsInstance(nf, MetaFeatureValue)

    def test_PercentOfAllMetal(self):
        pm = self.metafeatures_glass["percent_of_all_metal"](
             self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(pm.value, 0.6577857785778578)
        self.assertIsInstance(pm, MetaFeatureValue)

    def test_PercentOfMetalNonmetalCompounds(self):
        pmnc = self.metafeatures_glass["percent_of_metal_nonmetal_compounds"](
               self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(pmnc.value, 0.3207920792079208)
        self.assertIsInstance(pmnc, MetaFeatureValue)

    def test_PercentOfAllNonmetal(self):
        pan = self.metafeatures_glass["percent_of_all_nonmetal"](
              self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(pan.value, 0.021422142214221424)
        self.assertIsInstance(pan, MetaFeatureValue)

    def test_NumberOfDifferentElements(self):
        nde = self.metafeatures_glass["number_of_different_elements"](
              self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(nde.value, 38)
        self.assertIsInstance(nde, MetaFeatureValue)

    def test_AvgNumberOfElements(self):
        ane = self.metafeatures_glass["avg_number_of_elements"](
              self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(ane.value, 1.9801980198019802)
        self.assertIsInstance(ane, MetaFeatureValue)


class TestStructureMetafeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_castelli = load_castelli_perovskites()
        cls.df_castelli["structure"] = cls.df_castelli["structure"].\
            apply(lambda x: Structure.from_dict(x))

        cls.metafeatures_castelli = metafeatures.metafeatures
        cls.helpers_castelli = metafeatures.helpers

        cls.helpers_castelli.set_value(
            "structure_stats", cls.helpers_castelli["structure_stats"]
            (cls.df_castelli["structure"], cls.df_castelli["e_form"]))

    def test_NumberOfStructures(self):
        ns = self.metafeatures_castelli["number_of_structures"](
             self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(ns.value, 18928)
        self.assertIsInstance(ns, MetaFeatureValue)

    def test_PercentOfOrderedStructures(self):
        pos = self.metafeatures_castelli["percent_of_ordered_structures"](
             self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(pos.value, 1.0)
        self.assertIsInstance(pos, MetaFeatureValue)

    def test_AverageNumberOfSites(self):
        ans = self.metafeatures_castelli["avg_number_of_sites"](
             self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(ans.value, 5.0)
        self.assertIsInstance(ans, MetaFeatureValue)

    def test_MaxNumberOfSites(self):
        mns = self.metafeatures_castelli["max_number_of_sites"](
            self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(mns.value, 5.0)
        self.assertIsInstance(mns, MetaFeatureValue)

    def test_NumberOfDifferentElementsInStructure(self):
        mns = self.metafeatures_castelli[
            "number_of_different_elements_in_structure"](
            self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(mns.value, 56)
        self.assertIsInstance(mns, MetaFeatureValue)


if __name__ == "__main__":
    unittest.main()