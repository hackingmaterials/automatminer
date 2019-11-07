"""
Defines sets of featurizers to be used by automatminer during featurization.

Featurizer sets are classes with attributes containing lists of featurizers.
For example, the set of all express structure featurizers could be found with::

    StructureFeaturizers().express
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

    See the FeaturizerSet documentation for inspect of each property (sublist
    of featurizers).

    Example usage::

        best_featurizers = CompositionFeaturizers().express

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(CompositionFeaturizers, self).__init__(exclude=exclude)

    @property
    def debug(self):
        return self._get_featurizers([cf.ElementProperty.from_preset("magpie")])

    @property
    def express(self):
        fs = [
            cf.ElementProperty.from_preset("magpie"),
            cf.OxidationStates.from_preset(preset_name="deml"),
            cf.ElectronAffinity(),
            cf.IonProperty(),
            cf.YangSolidSolution(),
            cf.Miedema(),
        ]
        return self._get_featurizers(fs)

    @property
    def heavy(self):
        fs = [cf.AtomicPackingEfficiency()] + self.express
        return self._get_featurizers(fs)

    @property
    def all(self):
        fs = [
            cf.AtomicOrbitals(),
            cf.ElementProperty.from_preset("matminer"),
            cf.ElementProperty.from_preset("magpie"),
            cf.ElementProperty.from_preset("matscholar_el"),
            cf.ElementProperty.from_preset("deml"),
            cf.Meredig(),
            cf.ElementFraction(),
            cf.Stoichiometry(),
            cf.TMetalFraction(),
            cf.BandCenter(),
            cf.ValenceOrbital(),
            cf.YangSolidSolution(),
            cf.CationProperty.from_preset(preset_name="deml"),
            cf.OxidationStates.from_preset(preset_name="deml"),
            cf.ElectronAffinity(),
            cf.ElectronegativityDiff(),
            cf.IonProperty(fast=True),
            cf.Miedema(),
            cf.AtomicPackingEfficiency(),  # slower than the rest
            cf.CohesiveEnergy(),  # requires mpid present
        ]
        return self._get_featurizers(fs)


class StructureFeaturizers(FeaturizerSet):
    """Featurizer set containing structure featurizers.

    See the FeaturizerSet documentation for inspect of each property (sublist
    of featurizers).

    Example usage::

        best_featurizers = StructureFeaturizers().express

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(StructureFeaturizers, self).__init__(exclude=exclude)
        self.ssf = sf.SiteStatsFingerprint

    def _add_external(self, fset):
        # Prevent import errors
        require_external = []
        if torch and cgcnn:
            require_external.append(sf.CGCNNFeaturizer())
        if dscribe:
            require_external.append(sf.SOAP())
        return fset + require_external

    @property
    def express(self):
        fs = [
            sf.DensityFeatures(),
            sf.GlobalSymmetryFeatures(),
            sf.EwaldEnergy(),
            sf.SineCoulombMatrix(flatten=True),
            sf.GlobalInstabilityIndex(),
            sf.StructuralComplexity(),
        ]
        return self._get_featurizers(fs)

    @property
    def heavy(self):
        fs = [
            self.ssf.from_preset("CrystalNNFingerprint_ops"),
            sf.ChemicalOrdering(),
            sf.StructuralHeterogeneity(),
            sf.MaximumPackingEfficiency(),
            sf.XRDPowderPattern(),
            sf.Dimensionality(),
            sf.OrbitalFieldMatrix(flatten=True),
            sf.JarvisCFID(),
        ]
        fs += self.express
        fs = self._add_external(fs)
        return self._get_featurizers(fs)

    @property
    def all(self):
        fs = [
            # Vector
            self.ssf.from_preset("CrystalNNFingerprint_ops"),
            self.ssf.from_preset("BondLength-dejong2016"),
            self.ssf.from_preset("BondAngle-dejong2016"),
            self.ssf.from_preset("Composition-dejong2016_SD"),
            self.ssf.from_preset("Composition-dejong2016_AD"),
            self.ssf.from_preset("CoordinationNumber_ward-prb-2017"),
            self.ssf.from_preset("LocalPropertyDifference_ward-prb-2017"),
            sf.BondFractions(approx_bonds=False),
            sf.BagofBonds(coulomb_matrix=sf.CoulombMatrix()),
            sf.BagofBonds(coulomb_matrix=sf.SineCoulombMatrix()),
            sf.CoulombMatrix(flatten=True),
            sf.BondFractions(),
            # Non vector
            sf.CoulombMatrix(flatten=False),  # returns matrix
            sf.SineCoulombMatrix(flatten=False),  # returns matrix
            sf.RadialDistributionFunction(),  # returns dict
            sf.MinimumRelativeDistances(),  # returns a list
            sf.ElectronicRadialDistributionFunction(),  # returns ??
            sf.PartialRadialDistributionFunction(),  # returns ??
        ]
        fs += self.heavy
        return self._get_featurizers(fs)

    @property
    def debug(self):
        return self._get_featurizers([sf.SineCoulombMatrix(flatten=True)])

    @property
    def need_fit(self):
        fs = [
            sf.PartialRadialDistributionFunction(),
            sf.BondFractions(),
            sf.BagofBonds(coulomb_matrix=sf.CoulombMatrix()),
            sf.BagofBonds(coulomb_matrix=sf.SineCoulombMatrix()),
        ]
        return self._get_featurizers(fs)


class DOSFeaturizers(FeaturizerSet):
    """Featurizer set containing density of states featurizers.

    See the FeaturizerSet documentation for inspect of each property (sublist
    of featurizers).

    Example usage::

        dos_featurizers = DOSFeaturizers().express

    Density of states featurizers should work on the entire density of states
    if they are in express or heavy. If they are in "all" they may work on
    sites or return matrices.

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(DOSFeaturizers, self).__init__(exclude=exclude)

    @property
    def all(self):
        """List of all density of states based featurizers."""
        return self.heavy + [dosf.SiteDOS()]

    @property
    def express(self):
        fs = [
            dosf.DOSFeaturizer(),
            dosf.DopingFermi(),
            dosf.Hybridization(),
            dosf.DosAsymmetry(),
        ]
        return self._get_featurizers(fs)

    @property
    def heavy(self):
        return self.express

    @property
    def debug(self):
        return self._get_featurizers([dosf.DOSFeaturizer()])


class BSFeaturizers(FeaturizerSet):
    """Featurizer set containing band structure featurizers.

    See the FeaturizerSet documentation for inspect of each property (sublist
    of featurizers).

    Example usage::

        bs_featurizers = BSFeaturizers().express

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(BSFeaturizers, self).__init__(exclude=exclude)

    @property
    def express(self):
        fs = [bf.BandFeaturizer(), bf.BranchPointEnergy()]
        return self._get_featurizers(fs)

    @property
    def heavy(self):
        return self.express

    @property
    def all(self):
        """List of all band structure based featurizers."""
        return self.heavy

    @property
    def debug(self):
        return self._get_featurizers([bf.BandFeaturizer()])


class AllFeaturizers(FeaturizerSet):
    """Featurizer set containing all available featurizers.

    This class provides subsets for composition, structure, density of states
    and band structure based featurizers. Additional sets containing all
    featurizers and the set of express/heavy/etc. featurizers are provided.

    Example usage::

        composition_featurizers = AllFeaturizers().composition

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        super(AllFeaturizers, self).__init__(exclude=exclude)

        self._featurizer_sets = {
            "comp": CompositionFeaturizers(),
            "struct": StructureFeaturizers(),
            "bs": BSFeaturizers(),
            "dos": DOSFeaturizers(),
        }

    @property
    def composition(self):
        """List of all composition based featurizers."""
        return self._get_featurizers(self._featurizer_sets["comp"].all)

    @property
    def structure(self):
        """List of all structure based featurizers."""
        return self._get_featurizers(self._featurizer_sets["struct"].all)

    @property
    def bandstructure(self):
        """List of all band structure based featurizers."""
        return self._get_featurizers(self._featurizer_sets["bs"].all)

    @property
    def dos(self):
        """List of all density of states based featurizers."""
        return self._get_featurizers(self._featurizer_sets["dos"].all)

    @property
    def express(self):
        fs = [f.express for f in self._featurizer_sets.values()]
        return self._get_featurizers(fs)

    @property
    def heavy(self):
        fs = [f.heavy for f in self._featurizer_sets.values()]
        return self._get_featurizers(fs)

    @property
    def all(self):
        fs = [f.all for f in self._featurizer_sets.values()]
        return self._get_featurizers(fs)

    @property
    def debug(self):
        fs = [f.debug for f in self._featurizer_sets.values()]
        return self._get_featurizers(fs)
