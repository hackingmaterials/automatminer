import os
import unittest
from pymatgen.core import Structure

from matbench.metalearning.metafeatures import *
from matbench.data.load import load_castelli_perovskites, load_jdft2d, \
    load_glass_binary, load_mp


test_dir = os.path.dirname(__file__)


class TestFormulaMetafeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_glass = load_glass_binary()

    def test_NumberOfFormulas(self):
        nf = NumberOfFormulas().calc(self.df_glass["formula"],
                                     self.df_glass["gfa"])
        self.assertEqual(nf, 5959)

    def test_PercentOfAllMetal(self):
        pm = PercentOfAllMetal().calc(self.df_glass["formula"],
                                      self.df_glass["gfa"])
        self.assertEqual(pm, 0.6577857785778578)

    def test_PercentOfMetalNonmetal(self):
        pmnc = PercentOfMetalNonmetal().calc(self.df_glass["formula"],
                                             self.df_glass["gfa"])
        self.assertEqual(pmnc, 0.3207920792079208)

    def test_PercentOfAllNonmetal(self):
        pan = PercentOfAllNonmetal().calc(self.df_glass["formula"],
                                          self.df_glass["gfa"])
        self.assertEqual(pan, 0.021422142214221424)

    def test_NumberOfDifferentElements(self):
        nde = NumberOfDifferentElements().calc(self.df_glass["formula"],
                                               self.df_glass["gfa"])
        self.assertEqual(nde, 38)

    def test_AvgNumberOfElements(self):
        ane = AvgNumberOfElements().calc(self.df_glass["formula"],
                                         self.df_glass["gfa"])
        self.assertEqual(ane, 1.9801980198019802)


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