"""
Tests for the featurization base classes.
"""

import unittest

import matminer.featurizers.composition as cf

from automatminer.featurization.base import FeaturizerSet


class TestAutoFeaturizer(unittest.TestCase):
    def test_fsets(self):
        """Test the base behavior of ABC FeaturizerSet."""

        class BadFeaturizerSet(FeaturizerSet):
            def __init__(self, exclude=None):
                super(BadFeaturizerSet, self).__init__(exclude=exclude)
                self._fast = [cf.ElementProperty.from_preset("matminer")]

            @property
            def express(self):
                return self._get_featurizers()

        with self.assertRaises(TypeError):
            BadFeaturizerSet()

        class GoodFeaturizerSet(FeaturizerSet):
            def __init__(self, exclude=None):
                super(GoodFeaturizerSet, self).__init__(exclude=exclude)
                self._express = [cf.ElementProperty.from_preset("matminer")]
                self._debug = self.express
                self._all = self.express
                self._heavy = self.express

            @property
            def express(self):
                return self._get_featurizers(self._express)

            @property
            def debug(self):
                return self._get_featurizers(self._debug)

            @property
            def all(self):
                return self._get_featurizers(self._all)

            @property
            def heavy(self):
                return self._get_featurizers(self._heavy)

        GoodFeaturizerSet()
