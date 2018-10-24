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

    def test_NumberOfStructures(self):
        ns = NumberOfStructures().calc(self.df_castelli["structure"],
                                       self.df_castelli["e_form"])
        self.assertEqual(ns.value, 18928)

    def test_PercentOfOrderedStructures(self):
        pos = PercentOfOrderedStructures().calc(self.df_castelli["structure"],
                                                self.df_castelli["e_form"])
        self.assertEqual(pos.value, 1.0)

    def test_AvgNumberOfSitess(self):
        ans = AvgNumberOfSites().calc(self.df_castelli["structure"],
                                      self.df_castelli["e_form"])
        self.assertEqual(ans.value, 5.0)

    def test_MaxNumberOfSites(self):
        mns = MaxNumberOfSites().calc(self.df_castelli["structure"],
                                      self.df_castelli["e_form"])
        self.assertEqual(mns.value, 5.0)

    def test_NumberOfDifferentElementsInStructure(self):
        mns = NumberOfDifferentElementsInStructure().calc(
            self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(mns.value, 56)


if __name__ == "__main__":
    unittest.main()