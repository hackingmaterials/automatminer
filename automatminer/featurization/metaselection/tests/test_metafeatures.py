import unittest
from matminer.datasets import load_dataset
from automatminer.featurization.metaselection.metafeatures import *

__author__ = ["Qi Wang <wqthu11@gmail.com>"]


class TestFormulaMetafeatures(unittest.TestCase):
    def setUp(self):
        self.test_df = load_dataset('elastic_tensor_2015').rename(
            columns={"formula": "composition"})

    def test_NumberOfCompositions(self):
        nf = NumberOfCompositions().calc(self.test_df["composition"])
        self.assertEqual(nf, 1181)

    def test_PercentOfAllMetal(self):
        pm = PercentOfAllMetal().calc(self.test_df["composition"])
        self.assertAlmostEqual(pm, 0.5919, 4)

    def test_PercentOfMetalNonmetal(self):
        pmnc = PercentOfMetalNonmetal().calc(self.test_df["composition"])
        self.assertAlmostEqual(pmnc, 0.3810, 4)

    def test_PercentOfAllNonmetal(self):
        pan = PercentOfAllNonmetal().calc(self.test_df["composition"])
        self.assertAlmostEqual(pan, 0.0271, 4)

    def test_PercentOfContainTransMetal(self):
        pctm = PercentOfContainTransMetal().calc(self.test_df["composition"])
        self.assertAlmostEqual(pctm, 0.8273, 4)

    def test_NumberOfDifferentElements(self):
        nde = NumberOfDifferentElements().calc(self.test_df["composition"])
        self.assertEqual(nde, 63)

    def test_AvgNumberOfElements(self):
        ane = AvgNumberOfElements().calc(self.test_df["composition"])
        self.assertAlmostEqual(ane, 2.2007, 4)

    def test_MaxNumberOfElements(self):
        mne = MaxNumberOfElements().calc(self.test_df["composition"])
        self.assertEqual(mne, 4)

    def test_MinNumberOfElements(self):
        mne = MinNumberOfElements().calc(self.test_df["composition"])
        self.assertEqual(mne, 1)


class TestStructureMetafeatures(unittest.TestCase):
    def setUp(self):
        self.test_df = load_dataset('elastic_tensor_2015').rename(
            columns={"formula": "composition"})

    def test_NumberOfStructures(self):
        ns = NumberOfStructures().calc(self.test_df["structure"])
        self.assertEqual(ns, 1181)

    def test_PercentOfOrderedStructures(self):
        pos = PercentOfOrderedStructures().calc(self.test_df["structure"])
        self.assertAlmostEqual(pos, 1.0)

    def test_AvgNumberOfSites(self):
        ans = AvgNumberOfSites().calc(self.test_df["structure"])
        self.assertAlmostEqual(ans, 12.4259, 4)

    def test_MaxNumberOfSites(self):
        mns = MaxNumberOfSites().calc(self.test_df["structure"])
        self.assertEqual(mns, 152)

    def test_NumberOfDifferentElementsInStructure(self):
        mns = NumberOfDifferentElementsInStructure().calc(
            self.test_df["structure"])
        self.assertEqual(mns, 63)


if __name__ == "__main__":
    unittest.main()
