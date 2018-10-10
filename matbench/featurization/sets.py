"""
Defines sets of featurizers to be used by matbench during featurization.

Featurizer sets are classes with attributes containing lists of featurizers.
For example, the set of all fast structure featurizers could be found with::

    StructureFeaturizers().fast
"""

import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
import matminer.featurizers.dos as dosf
import matminer.featurizers.bandstructure as bf

__authors__ = ["Alex Dunn", "Alex Ganose"]


class FeaturizerSet:
    """Abstract class for defining sets of featurizers.

    All FeaturizerSets should implement at least two sets of featurizers, best
    and all. The set of best featurizers should contain those featurizers
    that balance speed, applicability and usefulness. This should be determined
    by the implementor.

    Each set returned is a list of matminer featurizer objects.

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        self.exclude = exclude if exclude else None

    def __call__(self, *args, **kwargs):
        return self.all

    @property
    def best(self):
        """List of featurizers providing useful features in a reasonable time.

        Featurizers that take a very long time to run, which crash for many
        systems, or which produce a large number of similar features will be
        excluded.
        """
        raise NotImplementedError("This featurizer set must return a set of "
                                  "best featurizers")

    @property
    def all(self):
        """All featurizers available for this featurization type."""
        raise NotImplementedError("This featurizer set must return a set of "
                                  "all featurizers")

    def _get_featurizers(self, featurizers):
        """Utility function for getting featurizers not in the ignore list."""
        return [f for f in featurizers
                if f.__class__.__name__ not in self.exclude]


class AllFeaturizers(FeaturizerSet):
    """Featurizer set containing all available featurizers.

    This class provides subsets for composition, structure, density of states
    and band structure based featurizers. Additional sets containing all
    featurizers and the set of best featurizers are provided.

    Example usage::

        composition_featurizers = AllFeaturizers().composition

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(FeaturizerSet, self).__init__(exclude=exclude)

        self._featurizer_sets = {
            'comp': CompositionFeaturizers(),
            'struct': StructureFeaturizers(),
            'bs': BSFeaturizers(),
            'dos': DOSFeaturizers()
        }

    @property
    def best(self):
        """See base class."""
        featurizers = [f.best for f in self._featurizer_sets.values()]
        return self._get_featurizers(featurizers)

    @property
    def all(self):
        """See base class."""
        featurizers = [f.all for f in self._featurizer_sets.values()]
        return self._get_featurizers(featurizers)

    @property
    def composition(self):
        """List of all composition based featurizers."""
        return self._get_featurizers(self._featurizer_sets['comp'].all)

    @property
    def structure(self):
        """List of all structure based featurizers."""
        return self._get_featurizers(self._featurizer_sets['struct'].all)

    @property
    def bandstructure(self):
        """List of all band structure based featurizers."""
        return self._get_featurizers(self._featurizer_sets['bs'].all)

    @property
    def dos(self):
        """List of all density of states based featurizers."""
        return self._get_featurizers(self._featurizer_sets['dos'].all)


class CompositionFeaturizers(FeaturizerSet):
    """Featurizer set containing composition featurizers.

    This class provides subsets for featurizers that require the composition
    to have oxidation states, as well as fast, and slow featurizers. Additional
    sets containing all featurizers and the set of best featurizers are
    provided.

    Example usage::

        fast_featurizers = CompositionFeaturizers().fast

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(FeaturizerSet, self).__init__(exclude=exclude)

        self._fast_featurizers = [
            cf.AtomicOrbitals(),
            cf.ElementProperty.from_preset("matminer"),
            cf.ElementProperty.from_preset("magpie"),
            cf.ElementFraction(),
            cf.Stoichiometry(),
            cf.TMetalFraction(),
            cf.BandCenter(),
            cf.ValenceOrbital()
        ]

        self._slow_featurizers = [
            cf.Miedema(),
            cf.AtomicPackingEfficiency(),  # slower than the rest
            cf.CohesiveEnergy()  # requires mpid present
        ]

        self._need_oxi_featurizers = [
            cf.CationProperty.from_preset(preset_name='deml'),
            cf.OxidationStates.from_preset(preset_name='deml'),
            cf.ElectronAffinity(),
            cf.ElectronegativityDiff(),
            cf.YangSolidSolution(),
            cf.IonProperty()
        ]

    @property
    def fast(self):
        """List of featurizers that are generally quick to featurize."""
        return self._get_featurizers(self._fast_featurizers)

    @property
    def need_oxi(self):
        """Featurizers that require the composition to have oxidation states.

        If the composition is not decorated with oxidation states the
        oxidation states will be guessed. This can cause a significant increase
        in featurization time.
        """
        return self._get_featurizers(self._need_oxi_featurizers)

    @property
    def slow(self):
        """List of featurizers that are generally slow to featurize."""
        return self._get_featurizers(self._slow_featurizers)

    @property
    def all(self):
        """List of all composition based featurizers."""
        return self.fast + self.need_oxi + self.slow

    @property
    def best(self):
        """See base class."""
        return self.fast


class StructureFeaturizers(FeaturizerSet):
    """
    Lists of structure featurizers, depending on requirements.

    Example usage:
        fast_featurizers = StructureFeaturizers().fast
    """

    @property
    def matrix(self):
        """
        Structure featurizers returning matrices in each column. Not useful
        for vectorized representations of crystal structures.
        """
        featzers = [sf.RadialDistributionFunction(),  # returns dict
                    sf.CoulombMatrix(),  # returns a matrix
                    sf.SineCoulombMatrix(),  # returns a matrix
                    sf.OrbitalFieldMatrix(),  # returns a matrix
                    sf.MinimumRelativeDistances(),  # returns a list
                    sf.ElectronicRadialDistributionFunction()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def fast(self):
        """
        Structure featurizers which are generally fast.
        """
        featzers = [sf.DensityFeatures(),
                    sf.GlobalSymmetryFeatures(),
                    sf.EwaldEnergy()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def many_features(self):
        featzers = [sf.BagofBonds(),
                    sf.PartialRadialDistributionFunction(),
                    sf.BondFractions()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def need_fit(self):
        """
        Structure featurizers which must be .fit before featurizing.
        Alternatively, use .fit_featurize_dataframe.
        """
        featzers = [sf.PartialRadialDistributionFunction(),
                    sf.BondFractions(),
                    sf.BagofBonds()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def slow(self):
        """
        Structure featurizers which are generally slow.
        """
        featzers = [
            sf.SiteStatsFingerprint.from_preset('CrystalNNFingerprint_ops'),
            sf.ChemicalOrdering(),
            sf.StructuralHeterogeneity(),
            sf.MaximumPackingEfficiency(),
            sf.XRDPowderPattern(),
            sf.Dimensionality()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def all(self):
        return self.fast + self.slow + self.need_fit + self.matrix

    @property
    def best(self):
        return self.fast + self.slow


class DOSFeaturizers(FeaturizerSet):
    """
    Lists of DOS featurizers, depending on requirements

    Args:
        exclude ([str]): The class names of the featurizers which should be
            excluded.

    Example usage:
        fast_featurizers = StructureFeaturizers().fast
    """

    @property
    def all(self):
        featzers = [dosf.DOSFeaturizer(),
                    dosf.DopingFermi(),
                    dosf.Hybridization()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def best(self):
        return self.all


class BSFeaturizers(FeaturizerSet):
    """
    Lists of bandstructure featurizers, depending on requirements.

    Args:
        exclude ([str]): The class names of the featurizers which should be
            excluded.

    Example usage:
        fast_featurizers = StructureFeaturizers().fast
    """

    @property
    def all(self):
        featzers = [bf.BandFeaturizer(), bf.BranchPointEnergy()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def best(self):
        return self.all