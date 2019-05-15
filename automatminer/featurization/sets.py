"""
Defines sets of featurizers to be used by automatminer during featurization.

Featurizer sets are classes with attributes containing lists of featurizers.
For example, the set of all robust structure featurizers could be found with::

    StructureFeaturizers().robust
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

from .base import FeaturizerSet

__authors__ = ["Alex Dunn", "Alex Ganose"]


class CompositionFeaturizers(FeaturizerSet):
    """Featurizer set containing composition featurizers.

    This class provides subsets for featurizers that require the composition
    to have oxidation states, as well as robust, and slow featurizers. Additional
    sets containing all featurizers and the set of best featurizers are
    provided.

    Example usage::

        fast_featurizers = CompositionFeaturizers().robust

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(CompositionFeaturizers, self).__init__(exclude=exclude)

        self.fast_best = [
            cf.ElementProperty.from_preset("magpie"),

        ]

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
    def debug(self):
        return self._get_featurizers([cf.ElementProperty.from_preset("magpie")])

    @property
    def robust(self):
        fs = [
            cf.ElementProperty.from_preset("magpie"),
            cf.OxidationStates.from_preset(preset_name='deml'),
            cf.ElectronAffinity(),
            cf.IonProperty(),
            cf.YangSolidSolution(),
            cf.Miedema(),
            cf.YangSolidSolution()
        ]
        return self._get_featurizers(fs)

    @property
    def best(self):
        fs = [cf.AtomicPackingEfficiency()] + self.robust

    @property
    def all(self):
        fs = [

        ]
    # @property
    # def intermetallics_only(self):
    #     """List of featurizers that applies only to intermetallics.
    #     Will probably be removed by valid_fraction checking if not actally
    #     applicable to the dataset.
    #     """
    #     return self._get_featurizers(self._intermetallics_only)
    #
    # @property
    # def robust(self):
    #     """List of featurizers that are generally quick to featurize."""
    #     return self._get_featurizers(self._fast_featurizers)
    #
    # @property
    # def slow(self):
    #     """List of featurizers that are generally slow to featurize."""
    #     return self._get_featurizers(self._slow_featurizers)
    #
    # @property
    # def need_oxi(self):
    #     """Featurizers that require the composition to have oxidation states.
    #
    #     If the composition is not decorated with oxidation states the
    #     oxidation states will be guessed. This can cause a significant increase
    #     in featurization time.
    #     """
    #     return self._get_featurizers(self._need_oxi_featurizers)
    #
    # @property
    # def all(self):
    #     """List of all composition based featurizers."""
    #     return self.robust + self.need_oxi + self.slow
    #
    # @property
    # def best(self):
    #     return self.robust + self.intermetallics_only
    #
    # @property
    # def debug(self):
    #     return self._get_featurizers([cf.ElementProperty.from_preset("magpie")])


class StructureFeaturizers(FeaturizerSet):
    """Featurizer set containing structure featurizers.

    This class provides subsets for featurizers that require fitting,
    return matrices rather than vectors, and produce many features, as well as
    robust, and slow featurizers. Additional sets containing all featurizers and
    the set of best featurizers are provided.

    Example usage::

        fast_featurizers = StructureFeaturizers().robust

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
            sf.SineCoulombMatrix(flatten=True),
            sf.GlobalInstabilityIndex(),
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
            sf.CoulombMatrix(flatten=False),
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
    def robust(self):
        """List of featurizers that are generally robust to featurize."""
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
        return self.robust + self.slow + self.need_fit + self.require_external

    @property
    def all(self):
        return self.all_vector

    @property
    def all_including_matrix(self):
        """List of all structure based featurizers."""
        return self.all_vector + self.matrix

    @property
    def best(self):
        return self.robust + self.slow + self.require_external

    @property
    def debug(self):
        return self._get_featurizers([sf.SineCoulombMatrix(flatten=True)])


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
    def robust(self):
        return self._get_featurizers(self._best_featurizers)

    @property
    def site(self):
        return self._get_featurizers(self._site_featurizers)

    @property
    def debug(self):
        return self._get_featurizers([dosf.DOSFeaturizer()])


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
    def robust(self):
        return self._get_featurizers(self._best_featurizers)

    @property
    def debug(self):
        return self._get_featurizers([bf.BandFeaturizer()])


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
    def robust(self):
        featurizers = [f.robust for f in self._featurizer_sets.values()]
        return self._get_featurizers(featurizers)

    @property
    def debug(self):
        featurizers = [f.debug for f in self._featurizer_sets.values()]
        return self._get_featurizers(featurizers)
