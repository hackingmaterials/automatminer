import os
import unittest
from pymatgen.core import Structure

from matbench.metalearning.base import MetaFeatureValue
from matbench.metalearning import metafeatures
from matbench.data.load import load_castelli_perovskites, load_jdft2d, \
    load_glass_formation, load_mp


test_dir = os.path.dirname(__file__)


class TestCompositionMetafeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_glass = load_glass_formation(phase="ternary")

        cls.metafeatures_glass = metafeatures.metafeatures
        cls.helpers_glass = metafeatures.helpers

        cls.helpers_glass.set_value(
            "FormulaStats", cls.helpers_glass["FormulaStats"]
            (cls.df_glass["formula"], cls.df_glass["gfa"]))

    def test_NumberOfFormulas(self):
        nf = self.metafeatures_glass["NumberOfFormulas"](
             self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(nf.value, 7742)
        self.assertIsInstance(nf, MetaFeatureValue)

    def test_PercentOfAllMetal(self):
        pm = self.metafeatures_glass["PercentOfAllMetal"](
             self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(pm.value, 0.3944599195295574)
        self.assertIsInstance(pm, MetaFeatureValue)

    def test_PercentOfMetalNonmetalCompounds(self):
        pmnc = self.metafeatures_glass["PercentOfMetalNonmetalCompounds"](
               self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(pmnc.value, 0.6046115753636645)
        self.assertIsInstance(pmnc, MetaFeatureValue)

    def test_PercentOfAllNonmetal(self):
        pan = self.metafeatures_glass["PercentOfAllNonmetal"](
              self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(pan.value, 0.0007737542556484061)
        self.assertIsInstance(pan, MetaFeatureValue)

    def test_NumberOfDifferentElements(self):
        nde = self.metafeatures_glass["NumberOfDifferentElements"](
              self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(nde.value, 57)
        self.assertIsInstance(nde, MetaFeatureValue)

    def test_AvgNumberOfElements(self):
        ane = self.metafeatures_glass["AvgNumberOfElements"](
              self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(ane.value, 2.8404518724852985)
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
            "StructureStats", cls.helpers_castelli["StructureStats"]
            (cls.df_castelli["structure"], cls.df_castelli["e_form"]))

    def test_NumberOfStructures(self):
        ns = self.metafeatures_castelli["NumberOfFormulas"](
             self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(ns.value, 18928)
        self.assertIsInstance(ns, MetaFeatureValue)

    def test_PercentOfOrderedStructures(self):
        pos = self.metafeatures_castelli["PercentOfOrderedStructures"](
             self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(pos.value, 1.0)
        self.assertIsInstance(pos, MetaFeatureValue)

    def test_AverageNumberOfSites(self):
        ans = self.metafeatures_castelli["AverageNumberOfSites"](
             self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(ans.value, 5.0)
        self.assertIsInstance(ans, MetaFeatureValue)

    def test_MaxNumberOfSites(self):
        mns = self.metafeatures_castelli["MaxNumberOfSites"](
             self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(mns.value, 5.0)
        self.assertIsInstance(mns, MetaFeatureValue)

    def test_NumberOfDifferentElementsInStructure(self):
        mns = self.metafeatures_castelli["NumberOfDifferentElementsInStructure"](
             self.df_castelli["structure"], self.df_castelli["e_form"])
        self.assertEqual(mns.value, 56)
        self.assertIsInstance(mns, MetaFeatureValue)


if __name__ == "__main__":
    unittest.main()