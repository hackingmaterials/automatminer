import unittest
import inspect

from matminer.featurizers.base import BaseFeaturizer
import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
import matminer.featurizers.dos as dosf
import matminer.featurizers.bandstructure as bf

from automatminer.featurization.sets import AllFeaturizers, StructureFeaturizers

try:
    import dscribe
except ImportError:
    dscribe = None


class TestAllFeaturizers(unittest.TestCase):
    """
    Class to ensure the featurizers available in featurizer files in matminer
    match exactly to those defined to AllFeaturizers class. This test is meant
    to catch events when a new featurizer is defined but not listed inside
    AllFeaturizers (e.g. by mistake).
    """

    def setUp(self):
        self.allfs = AllFeaturizers()

    @staticmethod
    def get_featurizers(module, ignore=None):
        """Get a list of featurizers class names defined in a module.

        Args:
            module (module): A python module.
            ignore (`list` of `str`, optional): Class names to ignore.

        Returns:
            (`list` of `str`): List of all featurizer class names.
        """
        ignore = ignore if ignore else []

        def is_featurizer(class_object):
            return (issubclass(class_object, BaseFeaturizer)
                    and not class_object == BaseFeaturizer)

        # getmembers returns list of (class_name, class_object)
        classes = [n for n, c in inspect.getmembers(module, inspect.isclass)
                   if is_featurizer(c) and c.__module__ == module.__name__]
        featurizers = [c for c in classes if c not in ignore]
        return featurizers

    def _test_features_implemented(self, test_feats, true_feats):
        """Check two lists of featurizers are the same.

        Note that `test_feats` is a list of objects and `true_feats` is a
        list of class names as strings.
        """
        test_feats = [c.__class__.__name__ for c in test_feats]

        for featurizer_name in true_feats:
            self.assertTrue(featurizer_name in test_feats,
                            ("{} matminer featurizer not in implemented in "
                             "automatminer").format(featurizer_name))

    def test_composition_featurizers(self):
        true_feats = TestAllFeaturizers.get_featurizers(cf)
        test_feats = self.allfs.composition
        self._test_features_implemented(test_feats, true_feats)

    def test_structure_featurizers(self):
        ignore = ['StructureComposition', 'CGCNNFeaturizer']
        ignore += [klass.__class__.__name__ for klass in
                   StructureFeaturizers().matrix]
        if not dscribe:
            ignore += ["SOAP"]
        true_feats = self.get_featurizers(sf, ignore)
        test_feats = self.allfs.structure
        self._test_features_implemented(test_feats, true_feats)

    def test_dos_featurizers(self):
        true_feats = self.get_featurizers(dosf)
        test_feats = self.allfs.dos
        self._test_features_implemented(test_feats, true_feats)

    def test_bandstructure_featurizers(self):
        true_feats = self.get_featurizers(bf)
        test_feats = self.allfs.bandstructure
        self._test_features_implemented(test_feats, true_feats)


if __name__ == '__main__':
    unittest.main()
