import numpy as np
from pymatgen.core import Composition
from .base import MetafeatureFunctions, MetaFeature
from .base import HelperFunctions, HelperFunction
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

Current meta-features to be considered (many can be further added):
(i) Composition-related:
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
    - Percent of ordered structures:
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

metafeatures = MetafeatureFunctions()
helpers = HelperFunctions()


##################################
# composition-related metafeatures
##################################
@helpers.define("FormulaStats")
class FormulaStats(HelperFunction):
    def _calculate(self, x, y):
        stats = FormulaStatistics(x).calc()
        return stats


@metafeatures.define("NumberOfFormulas")
class NumberOfFormulas(MetaFeature):
    def _calculate(self, x, y):
        return len(x)


@metafeatures.define("PercentOfAllMetal",
                     dependency="FormulaStats")
class PercentOfAllMetal(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("FormulaStats")
        num = sum([1 if stat["major_formula_category"] == 1 else 0
                  for stat in stats.values()])
        return num / len(stats)


@metafeatures.define("PercentOfMetalNonmetalCompounds",
                     dependency="FormulaStats")
class PercentOfMetalNonmetalCompounds(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("FormulaStats")
        num = sum([1 if stat["major_formula_category"] == 2 else 0
                  for stat in stats.values()])
        return num / len(stats)


@metafeatures.define("PercentOfAllNonmetal",
                     dependency="FormulaStats")
class PercentOfAllNonmetal(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("FormulaStats")
        num = sum([1 if stat["major_formula_category"] == 3 else 0
                  for stat in stats.values()])
        return num / len(stats)


@metafeatures.define("NumberOfDifferentElements",
                     dependency="FormulaStats")
class NumberOfDifferentElements(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("FormulaStats")
        elements = set()
        for stat in stats.values():
            elements = elements.union(set(stat["elements"]))
        return len(elements)


@metafeatures.define("AvgNumberOfElements",
                     dependency="FormulaStats")
class AvgNumberOfElements(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("FormulaStats")
        nelements_sum = sum([stat["n_elements"] for stat in stats.values()])
        return nelements_sum / len(stats)


@metafeatures.define("MaxNumberOfElements",
                     dependency="FormulaStats")
class MaxNumberOfElements(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("FormulaStats")
        nelements_max = max([stat["n_elements"] for stat in stats.values()])
        return nelements_max


@metafeatures.define("MinNumberOfElements",
                     dependency="FormulaStats")
class MinNumberOfElements(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("FormulaStats")
        nelements_min = min([stat["n_elements"] for stat in stats.values()])
        return nelements_min


################################
# structure-related metafeatures
################################
@helpers.define("StructureStats")
class StructureStats(HelperFunction):
    def _calculate(self, x, y):
        stats = StructureStatistics(x).calc()
        return stats


@metafeatures.define("NumberOfStructures")
class NumberOfStructures(MetaFeature):
    def _calculate(self, x, y):
        return len(x)


@metafeatures.define("PercentOfOrderedStructures",
                     dependency="StructureStats")
class PercentOfOrderedStructures(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("StructureStats")
        num = sum([1 if stat["is_ordered"] else 0 for stat in stats.values()])
        return num/len(stats)


@metafeatures.define("AverageNumberOfSites",
                     dependency="StructureStats")
class AverageNumberOfSites(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("StructureStats")
        nsites_sum = sum([stat["n_sites"] for stat in stats.values()])
        return nsites_sum / len(stats)


@metafeatures.define("MaxNumberOfSites",
                     dependency="StructureStats")
class MaxNumberOfSites(MetaFeature):
    def _calculate(self, x, y):
        stats = helpers.get_value("StructureStats")
        nsites_max = max([stat["n_sites"] for stat in stats.values()])
        return nsites_max


@metafeatures.define("NumberOfDifferentElementsInStructure")
class NumberOfDifferentElementsInStructure(MetaFeature):
    def _calculate(self, x, y):
        elements = set()
        for struct in x:
            c = Composition(struct.formula)
            els = [x.symbol for x in c.elements]
            elements = elements.union(set(els))
        return len(elements)


####################################
# missing_value related metafeatures
####################################
@helpers.define("MissingValues")
class MissingValues(HelperFunction):
    def _calculate(self, X, y):
        missing = ~np.isfinite(X)
        return missing

    def _calculate_sparse(self, X, y):
        data = [True if not np.isfinite(X) else False for stat in X.data]
        missing = X.__class__((data, X.indices, X.indptr), shape=X.shape,
                              dtype=np.bool)
        return missing


@metafeatures.define("NumberOfInstancesWithMissingValues",
                     dependency="MissingValues")
class NumberOfInstancesWithMissingValues(MetaFeature):
    def _calculate(self, X, y):
        missing = helpers.get_value("MissingValues")
        num_missing = missing.sum(axis=1)
        return sum([1 if num > 0 else 0 for num in num_missing])

    def _calculate_sparse(self, X, y):
        missing = helpers.get_value("MissingValues")
        new_missing = missing.tocsr()
        num_missing = [np.sum(
            new_missing.data[new_missing.indptr[i]:new_missing.indptr[i + 1]])
            for i in range(new_missing.shape[0])]
        return sum([1 if num > 0 else 0 for num in num_missing])


@metafeatures.define("PercentOfInstancesWithMissingValues",
                     dependency="NumberOfInstancesWithMissingValues")
class PercentageOfInstancesWithMissingValues(MetaFeature):
    def _calculate(self, X, y):
        return metafeatures.get_value("NumberOfInstancesWithMissingValues") \
               / metafeatures["NumberOfInstances"](X, y).value


@metafeatures.define("NumberOfMissingValues",
                     dependency="MissingValues")
class NumberOfMissingValues(MetaFeature):
    def _calculate(self, X, y):
        return helpers.get_value("MissingValues").sum()


@metafeatures.define("PercentOfMissingValues",
                     dependency="NumberOfMissingValues")
class PercentageOfMissingValues(MetaFeature):
    def _calculate(self, X, y):
        return metafeatures.get_value("NumberOfMissingValues") \
               / (X.shape[0] * X.shape[1])
