"""
Defines sets of featurizers to be used by matbench during featurization.

Classes of featurizer sets should have attributes being lists of featurizers.
For example, the set of all fast structure featurizers could be found with:
    StructureFeaturizers().fast
"""
import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
import matminer.featurizers.dos as dosf
import matminer.featurizers.bandstructure as bf

__authors__ = ["Alex Dunn"]


class FeaturizerSet:
    """
    An abstract class for defining sets of featurizers and the methods they
    must implement.

    Each set returned is a list of matminer featurizer objects.

    Args:
        exclude ([str]): The class names of the featurizers which should be
            excluded.
    """

    def __init__(self, exclude=None):
        self.exclude = [] if exclude is None else exclude

    def __call__(self, *args, **kwargs):
        return self.all

    def best(self):
        """
        A set of featurizers that generally gives informative features without
        excessive featurization time.
        """
        raise NotImplementedError("This featurizer set must return a set of "
                                  "best featurizers")

    def all(self):
        """
        All featurizers available in matminer for this featurization type.
        """
        raise NotImplementedError("This featurizer set must return a set of "
                                  "all featurizers")


class AllFeaturizers(FeaturizerSet):
    """
    Args:
        exclude ([str]): The class names of the featurizers which should be
            excluded.

    Example usage:
        composition_featurizers = AllFeaturizers().composition
    """

    @property
    def best(self):
        featzers = CompositionFeaturizers().best + \
                   StructureFeaturizers().best + \
                   BSFeaturizers().best + \
                   DOSFeaturizers().best

        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def all(self):
        featzers = CompositionFeaturizers().all + \
                   StructureFeaturizers().all + \
                   BSFeaturizers().all + \
                   DOSFeaturizers().all
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def composition(self):
        featzers = CompositionFeaturizers().all
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def structure(self):
        featzers = StructureFeaturizers().all
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def bandstructure(self):
        featzers = BSFeaturizers().all
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def dos(self):
        featzers = DOSFeaturizers().all
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]


class CompositionFeaturizers(FeaturizerSet):
    """
    Lists of composition featurizers, depending on requirements.

    Args:
        exclude ([str]): The class names of the featurizers which should be
            excluded.

    Example usage:
        fast_featurizers = CompositionFeaturizers().fast
    """

    @property
    def fast(self):
        """
        Generally fast featurizers.
        """
        featzers = [cf.AtomicOrbitals(),
                    cf.ElementProperty.from_preset("matminer"),
                    cf.ElementFraction(),
                    cf.Stoichiometry(),
                    cf.TMetalFraction(),
                    cf.BandCenter(),
                    cf.ValenceOrbital()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def need_oxi(self):
        """
        Fast if compositions are already decorated with oxidation states, slow
        otherwise.
        """
        featzers = [cf.CationProperty.from_preset(preset_name='deml'),
                    cf.OxidationStates.from_preset(preset_name='deml'),
                    cf.ElectronAffinity(),
                    cf.ElectronegativityDiff(),
                    cf.YangSolidSolution(),
                    cf.IonProperty()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def slow(self):
        """
        Generally slow featurizers under most conditions.
        """
        featzers = [cf.Miedema(),
                    # much slower than the rest
                    cf.AtomicPackingEfficiency(),
                    # requires mpid present
                    cf.CohesiveEnergy()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def all(self):
        """
        All composition featurizers.
        """
        return self.fast + self.need_oxi + self.slow

    @property
    def best(self):
        return self.fast + [cf.ElementProperty.from_preset("magpie")]


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