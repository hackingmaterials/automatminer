import inspect
import os
import ast
import pandas as pd
import unittest
from pymatgen.core import Structure

from matbench.metalearning.base import MetaFeatureValue
from matbench.metalearning import metafeatures
from matbench.data.load import load_castelli_perovskites, load_jdft2d, \
    load_glass_formation, load_mp
from matbench.metalearning.metafeatures import FormulaStats, StructureStats
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval


test_dir = os.path.dirname(__file__)


class TestMetafeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_glass = load_glass_formation(phase="ternary")
        cls.df_castelli = load_castelli_perovskites()
        cls.df_castelli["structure"] = cls.df_castelli["structure"].\
            apply(lambda x: Structure.from_dict(x))

        cls.metafeatures_glass = metafeatures.metafeatures
        cls.helpers_glass = metafeatures.helpers
        cls.helpers_castelli = metafeatures.helpers

        cls.helpers_glass.set_value(
            "FormulaStats", cls.helpers_glass["FormulaStats"]
            (cls.df_glass["formula"], cls.df_glass["gfa"]))

        cls.helpers_castelli.set_value(
            "StructureStats", cls.helpers_castelli["StructureStats"]
            (cls.df_castelli["structure"], cls.df_castelli["e_form"]))

    def test_NumberOfFormulas(self):
        nf = self.metafeatures_glass["NumberOfFormulas"]\
            (self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(nf.value, 7742)
        self.assertIsInstance(nf, MetaFeatureValue)

    def test_PercentOfAllMetal(self):
        pm = self.metafeatures_glass["PercentOfAllMetal"]\
            (self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(pm.value, 0.3944599195295574)
        self.assertIsInstance(pm, MetaFeatureValue)

    def test_PercentOfMetalNonmetalCompounds(self):
        pmnc = self.metafeatures_glass["PercentOfMetalNonmetalCompounds"]\
              (self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(pmnc.value, 0.6046115753636645)
        self.assertIsInstance(pmnc, MetaFeatureValue)

    def test_PercentOfAllNonmetal(self):
        pan = self.metafeatures_glass["PercentOfAllNonmetal"]\
              (self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(pan.value, 0.0007737542556484061)
        self.assertIsInstance(pan, MetaFeatureValue)

    def test_NumberOfDifferentElements(self):
        nde = self.metafeatures_glass["NumberOfDifferentElements"]\
              (self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(nde.value, 57)
        self.assertIsInstance(nde, MetaFeatureValue)

    def test_AvgNumberOfElements(self):
        ane = self.metafeatures_glass["AvgNumberOfElements"]\
              (self.df_glass["formula"], self.df_glass["gfa"])
        self.assertEqual(ane.value, 2.8404518724852985)
        self.assertIsInstance(ane, MetaFeatureValue)

if __name__ == "__main__":
    unittest.main()