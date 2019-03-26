"""
Defines sets of featurizers to be used by automatminer during featurization.

Featurizer sets are classes with attributes containing lists of featurizers.
For example, the set of all fast structure featurizers could be found with::

    StructureFeaturizers().fast
"""

import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
import matminer.featurizers.dos as dosf
import matminer.featurizers.bandstructure as bf

try:
    import torch
    import cgcnn
except ImportError:
    torch, cgcnn = None, None

try:
    import dscribe
except ImportError:
    dscribe = None

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
        self.exclude = exclude if exclude else []

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

    @property
    def fast(self):
        """Fast featurizers available for this featurization type."""
        raise NotImplementedError("This featurizer set must return a set of "
                                  "fast featurizers")

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
        super(AllFeaturizers, self).__init__(exclude=exclude)

        self._featurizer_sets = {
            'comp': CompositionFeaturizers(),
            'struct': StructureFeaturizers(),
            'bs': BSFeaturizers(),
            'dos': DOSFeaturizers()
        }

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

    @property
    def all(self):
        featurizers = [f.all for f in self._featurizer_sets.values()]
        return self._get_featurizers(featurizers)

    @property
    def best(self):
        featurizers = [f.best for f in self._featurizer_sets.values()]
        return self._get_featurizers(featurizers)

    @property
    def fast(self):
        featurizers = [f.fast for f in self._featurizer_sets.values()]
        return self._get_featurizers(featurizers)


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
        super(CompositionFeaturizers, self).__init__(exclude=exclude)

        self._fast_featurizers = [
            cf.AtomicOrbitals(),
            cf.ElementProperty.from_preset("matminer"),
            cf.ElementProperty.from_preset("magpie"),
            cf.ElementProperty.from_preset("matscholar_el"),
            cf.ElementProperty.from_preset("deml"),
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
            cf.YangSolidSolution(),
            cf.CationProperty.from_preset(preset_name='deml'),
            cf.OxidationStates.from_preset(preset_name='deml'),
            cf.ElectronAffinity(),
            cf.ElectronegativityDiff(),
            cf.IonProperty()
        ]

        self._intermetallics_only = [
            cf.YangSolidSolution(),
            cf.Miedema(),
        ]

    @property
    def intermetallics_only(self):
        """List of featurizers that applies only to intermetallics.
        Will probably be removed by valid_fraction checking if not actally
        applicable to the dataset.
        """
        return self._get_featurizers(self._intermetallics_only)

    @property
    def fast(self):
        """List of featurizers that are generally quick to featurize."""
        return self._get_featurizers(self._fast_featurizers)

    @property
    def slow(self):
        """List of featurizers that are generally slow to featurize."""
        return self._get_featurizers(self._slow_featurizers)

    @property
    def need_oxi(self):
        """Featurizers that require the composition to have oxidation states.

        If the composition is not decorated with oxidation states the
        oxidation states will be guessed. This can cause a significant increase
        in featurization time.
        """
        return self._get_featurizers(self._need_oxi_featurizers)

    @property
    def all(self):
        """List of all composition based featurizers."""
        return self.fast + self.need_oxi + self.slow

    @property
    def best(self):
        return self.fast + self.intermetallics_only


class StructureFeaturizers(FeaturizerSet):
    """Featurizer set containing structure featurizers.

    This class provides subsets for featurizers that require fitting,
    return matrices rather than vectors, and produce many features, as well as
    fast, and slow featurizers. Additional sets containing all featurizers and
    the set of best featurizers are provided.

    Example usage::

        fast_featurizers = StructureFeaturizers().fast

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(StructureFeaturizers, self).__init__(exclude=exclude)

        self._fast_featurizers = [
            sf.DensityFeatures(),
            sf.GlobalSymmetryFeatures(),
            sf.EwaldEnergy(),
            sf.CoulombMatrix(flatten=True),
            sf.SineCoulombMatrix(flatten=True)
        ]

        ssf = sf.SiteStatsFingerprint
        self._slow_featurizers = [
            ssf.from_preset('CrystalNNFingerprint_ops'),
            ssf.from_preset("BondLength-dejong2016"),
            ssf.from_preset("BondAngle-dejong2016"),
            ssf.from_preset("Composition-dejong2016_SD"),
            ssf.from_preset("Composition-dejong2016_AD"),
            ssf.from_preset("CoordinationNumber_ward-prb-2017"),
            ssf.from_preset("LocalPropertyDifference_ward-prb-2017"),
            sf.ChemicalOrdering(),
            sf.StructuralHeterogeneity(),
            sf.MaximumPackingEfficiency(),
            sf.XRDPowderPattern(),
            sf.Dimensionality(),
            sf.OrbitalFieldMatrix(flatten=True),
            sf.JarvisCFID(),
        ]

        # Prevent import errors
        self._require_external = []
        if torch and cgcnn:
            self._require_external.append(sf.CGCNNFeaturizer())
        if dscribe:
            self._require_external.append(sf.SOAP())

        self._need_fitting_featurizers = [
            sf.PartialRadialDistributionFunction(),
            sf.BondFractions(),
            sf.BagofBonds(coulomb_matrix=sf.CoulombMatrix()),
            sf.BagofBonds(coulomb_matrix=sf.SineCoulombMatrix()),
        ]

        self._matrix_featurizers = [
            sf.RadialDistributionFunction(),  # returns dict
            sf.MinimumRelativeDistances(),  # returns a list
            sf.ElectronicRadialDistributionFunction()
        ]

        # these are the same as _need_fitting_featurizers
        self._many_features_featurizers = [
            sf.PartialRadialDistributionFunction(),
            sf.BondFractions(approx_bonds=False),
            sf.BagofBonds(coulomb_matrix=sf.CoulombMatrix()),
            sf.BagofBonds(coulomb_matrix=sf.SineCoulombMatrix()),
            sf.OrbitalFieldMatrix(flatten=True),
            sf.JarvisCFID()
        ]

    @property
    def fast(self):
        """List of featurizers that are generally fast to featurize."""
        return self._get_featurizers(self._fast_featurizers)

    @property
    def slow(self):
        """List of featurizers that are generally slow to featurize."""
        return self._get_featurizers(self._slow_featurizers)

    @property
    def need_fit(self):
        """List of featurizers which must be fit before featurizing.

        Fitting can be performed using the `Featurizer.fit()` method.
        Alternatively, the `Featurizer.fit_featurize_dataframe()` can be used
        to fit and featurize simultaneously.
        """
        return self._get_featurizers(self._need_fitting_featurizers)

    @property
    def matrix(self):
        """List of featurizers that return matrices as features.

        These featurizers are not useful for vectorized representations of
        crystal structures.
        """
        return self._get_featurizers(self._matrix_featurizers)

    @property
    def many_features(self):
        """List of featurizers that return many features."""
        return self._get_featurizers(self._many_features_featurizers)

    @property
    def require_external(self):
        """Featurizers which require external software not installable via
        Pypi
        """
        return self._get_featurizers(self._require_external)

    @property
    def all_vector(self):
        return self.fast + self.slow + self.need_fit + self.require_external

    @property
    def all(self):
        return self.all_vector

    @property
    def all_including_matrix(self):
        """List of all structure based featurizers."""
        return self.all_vector + self.matrix

    @property
    def best(self):
        return self.fast + self.slow + self.require_external


class DOSFeaturizers(FeaturizerSet):
    """Featurizer set containing density of states featurizers.

    This class provides subsets all featurizers and the set of best featurizers.

    Example usage::

        dos_featurizers = DOSFeaturizers().best

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(DOSFeaturizers, self).__init__(exclude=exclude)

        # Best featurizers work on the entire DOS
        self._best_featurizers = [
            dosf.DOSFeaturizer(),
            dosf.DopingFermi(),
            dosf.Hybridization(),
            dosf.DosAsymmetry()
        ]

        self._site_featurizers = [dosf.SiteDOS()]

    @property
    def all(self):
        """List of all density of states based featurizers."""
        return self.best + self.site

    @property
    def best(self):
        return self._get_featurizers(self._best_featurizers)

    @property
    def fast(self):
        return self._get_featurizers(self._best_featurizers)

    @property
    def site(self):
        return self._get_featurizers(self._site_featurizers)


class BSFeaturizers(FeaturizerSet):
    """Featurizer set containing band structure featurizers.

    This class provides subsets all featurizers and the set of best featurizers.

    Example usage::

        bs_featurizers = BSFeaturizers().best

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(BSFeaturizers, self).__init__(exclude=exclude)

        self._best_featurizers = [
            bf.BandFeaturizer(),
            bf.BranchPointEnergy(),
        ]

    @property
    def all(self):
        """List of all band structure based featurizers."""
        return self.best

    @property
    def best(self):
        return self._get_featurizers(self._best_featurizers)

    @property
    def fast(self):
        return self._get_featurizers(self._best_featurizers)
