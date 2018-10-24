import warnings
import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
from matbench.metalearning.metafeatures import formula_metafeatures, \
    structure_metafeatures

"""
Get recommendation of featurizers based on meta-features of the new dataset 
and some pre-defined strategies obtained by extensive benchmarking for 
datasets of different meta-features as well as some basic understandings 
of various featurizers. 

"""


class DatasetMetaFeatures:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def formula_metafeatures(self):
        if "formula" in self.X.columns:
            mfs = dict()
            for mf, mf_class in formula_metafeatures.items():
                mfs[mf] = mf_class.calc(self.X, self.y)
            return {"formula_metafeatures": mfs}
        else:
            warnings.warn("Input X does not contain 'formula' column, "
                          "will not return any statistical metafeatures "
                          "for formulas in this dataset!")
            return None

    def structure_metafeatures(self):
        if "structure" in self.X.columns:
            mfs = dict()
            for mf, mf_class in structure_metafeatures.items():
                mfs[mf] = mf_class.calc(self.X, self.y)
            return {"structure_metafeatures": mfs}
        else:
            warnings.warn("Input X does not contain 'structure' column, "
                          "will not return any statistical metafeatures "
                          "for structures in this dataset!")
            return None

    def auto_metafeatures(self, dataset_cols=("formula", "structure")):
        auto_mfs = dict()
        for column in dataset_cols:
            mfs_func = getattr(self, "{}_metafeatures".format(column), None)
            auto_mfs.update(mfs_func() if mfs_func is not None else {})

        return auto_mfs


class FeaturizerAutoFilter:
    """
    Given a dataset as a dataframe, return a featurizer set.
    Currently only support removing definitely useless featurizers.
    Cannot recommend featurizers based on the target.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def formula_excludes(self, mfs):
        excludes = list()
        if mfs:
            if mfs["percent_of_all_nonmetal"] > 0.80:
                excludes.extend(["Miedema",
                                 "YangSolidSolution"])

            if mfs["percent_of_transition_metal_alloy"] < 0.80:
                excludes.extend(["TMetalFraction"])

        # return dict {"formula_excludes": excludes}??
        return excludes

    def structure_excludes(self, mfs):
        excludes = list()
        if mfs:
            if mfs["percent_of_ordered_structures"] < 0.80:
                excludes.extend(["GlobalSymmetryFeatures"])

        return excludes

    def auto_excludes(self, dataset_cols=("formula", "structure")):
        auto_excludes = list()
        auto_mfs = DatasetMetaFeatures(self.X, self.y).\
            auto_metafeatures(dataset_cols)
        for column in dataset_cols:
            mfs = auto_mfs.get("{}_metafeatures".format(column))
            if mfs:
                excludes_func = getattr(self,
                                        "{}_excludes".format(column), None)
                auto_excludes.append(excludes_func(mfs)
                                     if excludes_func is not None else [])

        return auto_excludes

