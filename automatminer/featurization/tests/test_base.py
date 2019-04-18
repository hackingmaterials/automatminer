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
            def fast(self):
                return self._get_featurizers()

        with self.assertRaises(TypeError):
            BadFeaturizerSet()

        class GoodFeaturizerSet(FeaturizerSet):
            def __init__(self, exclude=None):
                super(GoodFeaturizerSet, self).__init__(exclude=exclude)
                self._fast = [cf.ElementProperty.from_preset("matminer")]
                self._debug = self._fast
                self._all = self._fast
                self._best = self._fast

            @property
            def fast(self):
                return self._get_featurizers(self._fast)

            @property
            def debug(self):
                return self._get_featurizers(self._debug)

            @property
            def all(self):
                return self._get_featurizers(self._all)

            @property
            def best(self):
                return self._get_featurizers(self._best)

        GoodFeaturizerSet()
