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
        pass

    def test_PercentOfMetalNonmetalCompounds(self):
        pass

    def test_PercentOfAllNonmetal(self):
        pass

    def test_NumberOfDifferentElements(self):
        pass

    def test_AvgNumberOfElements(self):
        pass

if __name__ == "__main__":
    unittest.main()