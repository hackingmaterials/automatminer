import numpy as np
import pandas as pd
from functools import lru_cache
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure, IStructure
from abc import ABCMeta, abstractmethod
from .utils import FormulaStatistics, StructureStatistics


"""
Derive a list of meta-features of a given dataset to get recommendation of 
featurizers.

The meta-features serve as indicators of the dataset characteristics that may
affect the choice of featurizers.

Based on the meta-features and then benchmarking (i) featurizer availability, 
(ii) featurizer importance and (iii) featurizer computational budget of 
existing featurizers on a variety of datasets, we can get a sense of how these
featurizers perform for datasets with different meta-features, and then make 
some strategies of featurizer selection.

When given a new dataset, we can compute its meta-features, and then get the 
recommended featurizers based on the pre-defined strategies (e.g. one way is 
to get the L1 distances of meta-features of all pre-defined strategies and 
meta-features of the new dataset, then find some "nearest" strategies and make
an estimation of computational budget, and finally taking all these factors 
together to make a final recommendation of featurizers)

Current meta-feat ures to be considered (many can be further added):
(i) formula-related:
    - Number of formulas:
    - Percent of all-metallic alloys:
    - Percent of metallic-nonmetallic compounds:
    - Percent of nonmetallic compounds:
    - Number of elements present in the entire dataset: 
        e.g. can help to decided whether to use ChemicalSRO or Bob featurizers
        that can return O(N^2) features (increase rapidly with the number of 
        elements present)
    - Avg. number of elements in compositions:
    - Max. number of elements in compositions:
    - Min. number of elements in compositions:
    To do:
    - Percent of transitional-metal-containing alloys (dependency: percent of 
        all-metallic alloys): 
        e.g. can be used to determisne whether to use featurizers such as Miedema 
        that is more applicable to transitional alloys.
    - Percent of transitional-nonmetallic compounds (dependency: percent of 
        metallic-nonmetallic compounds): 
    - Prototypes of phases in the dataset:
        e.g. AB; AB2O4; MAX phase; etc maybe useful.
    - Percent of organic/molecules: 
        may need to call other packages e.g. deepchem or just fail this task as 
        we cannot directly support it in matminer.
        
(ii) Structure-related:
    - Percent of  ordered structures:
        e.g. can help to decide whether to use some featurizers that only apply
        to ordered structure such as GlobalSymmetryFeatures
    - Avg. number of atoms in structures:
        e.g. can be important for deciding on some extremely computational 
        expensive featurizers such as Voronoi-based ones or site statistics 
        ones such as SiteStatsFingerprint. They are assumed to be quite slow 
        if there are too many atoms in the structures.
    - Max. number of sites in structures: 
    To do:
    - Percent of 3D structure:
    - Percent of 2D structure:
    - Percent of 1D structure:

(iii) Missing_values-related:
    - Number of instances with missing_values
    - Percent of instances with missing_values
    - Number of missing_values
    - Percent of missing_values
    
To do:
(iv) Task-related:
    - Regression or Classification: 
        maybe some featurizers work better for classification or better for 
        regression

"""


class AbstractMetaFeature(object):
    """

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calc(self, X, y):
        pass


class MetaFeature(AbstractMetaFeature):
    def __init__(self, dependence=None):
        self.dependence = dependence
        super(MetaFeature, self).__init__()


class Helper(AbstractMetaFeature):
    def __init__(self):
        super(Helper, self).__init__()


#######################################################
# formula-related metafeatures
#######################################################


def formula_stats(X):
    if isinstance(X, (pd.Series, pd.DataFrame)):
        return _formula_stats(tuple(X.values))
    if isinstance(X, (list, np.array)):
        return _formula_stats(tuple(X))


@lru_cache()
def _formula_stats(X):
    stats = FormulaStatistics(X).calc()
    return stats


class NumberOfFormulas(MetaFeature):
    def calc(self, X, y=None):
        return len(X)


class PercentOfAllMetal(MetaFeature):
    def calc(self, X, y=None):
        stats = formula_stats(X)
        num = sum([1 if stat["major_formula_category"] == 1 else 0
                  for stat in stats.values()])
        return num / len(stats)


class PercentOfMetalNonmetal(MetaFeature):
    def calc(self, X, y=None):
        stats = formula_stats(X)
        num = sum([1 if stat["major_formula_category"] == 2 else 0
                  for stat in stats.values()])
        return num / len(stats)


class PercentOfAllNonmetal(MetaFeature):
    def calc(self, X, y=None):
        stats = formula_stats(X)
        num = sum([1 if stat["major_formula_category"] == 3 else 0
                  for stat in stats.values()])
        return num / len(stats)


class PercentOfContainTransMetal(MetaFeature):
    def calc(self, X, y=None):
        stats = formula_stats(X)
        num = sum([1 if 1 in stat["el_types_reduced"] else 0
                  for stat in stats.values()])
        return num / len(stats)

class NumberOfDifferentElements(MetaFeature):
    def calc(self, X, y=None):
        stats = formula_stats(X)
        elements = set()
        for stat in stats.values():
            elements = elements.union(set(stat["elements"]))
        return len(elements)


class AvgNumberOfElements(MetaFeature):
    def calc(self, X, y=None):
        stats = formula_stats(X)
        nelements_sum = sum([stat["n_elements"] for stat in stats.values()])
        return nelements_sum / len(stats)


class MaxNumberOfElements(MetaFeature):
    def calc(self, X, y=None):
        stats = formula_stats(X)
        nelements_max = max([stat["n_elements"] for stat in stats.values()])
        return nelements_max


class MinNumberOfElements(MetaFeature):
    def calc(self, X, y=None):
        stats = formula_stats(X)
        nelements_min = min([stat["n_elements"] for stat in stats.values()])
        return nelements_min


#######################################################
# structure-related metafeatures
#######################################################

def structure_stats(X):
    X_struct= list()
    for structure in X:
        if isinstance(structure, Structure):
            X_struct.append(IStructure.from_sites(structure))
        elif isinstance(structure, IStructure):
            X_struct.append(structure)
    return _structure_stats(tuple(X_struct))


@lru_cache()
def _structure_stats(X):
    stats = StructureStatistics(X).calc()
    return stats


class NumberOfStructures(MetaFeature):
    def calc(self, X, y=None):
        return len(X)


class PercentOfOrderedStructures(MetaFeature):
    def calc(self, X, y=None):
        stats = structure_stats(X)
        num = sum([1 if stat["is_ordered"] else 0 for stat in stats.values()])
        return num/len(stats)


class AvgNumberOfSites(MetaFeature):
    def calc(self, X, y=None):
        stats = structure_stats(X)
        nsites_sum = sum([stat["n_sites"] for stat in stats.values()])
        return nsites_sum / len(stats)


class MaxNumberOfSites(MetaFeature):
    def calc(self, X, y=None):
        stats = structure_stats(X)
        nsites_max = max([stat["n_sites"] for stat in stats.values()])
        return nsites_max


class NumberOfDifferentElementsInStructure(MetaFeature):
    def calc(self, X, y=None):
        elements = set()
        for struct in X:
            c = Composition(struct.formula)
            els = [X.symbol for X in c.elements]
            elements = elements.union(set(els))
        return len(elements)


formula_mfs_list = \
    {"number_of_formulas": NumberOfFormulas(),
     "percent_of_all_metal": PercentOfAllMetal(),
     "percent_of_metal_nonmetal": PercentOfMetalNonmetal(),
     "percent_of_all_nonmetal": PercentOfAllNonmetal(),
     "percent_of_contain_trans_metal": PercentOfContainTransMetal(),
     "number_of_different_elements": NumberOfDifferentElements(),
     "avg_number_of_elements": AvgNumberOfElements(),
     "max_number_of_elements": MaxNumberOfElements(),
     "min_number_of_elements": MinNumberOfElements()}

structure_mfs_list = \
    {"number_of_structures": NumberOfStructures(),
     "percent_of_ordered_structures": PercentOfOrderedStructures(),
     "avg_number_of_sites": AvgNumberOfSites(),
     "max_number_of_sites": MaxNumberOfSites(),
     "number_of_different_elements_in_structures":
         NumberOfDifferentElementsInStructure()}