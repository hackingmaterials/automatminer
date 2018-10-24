import warnings
from matbench.metalearning.metafeatures import formula_metafeatures, \
    structure_metafeatures

"""
Automatically filter some featurizers based on metafeatures calculated for
the dataset. 
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

    @staticmethod
    def formula_featurizer_excludes(mfs, max_na_percent=0.05):
        excludes = list()
        try:
            if mfs["percent_of_all_nonmetal"] > max_na_percent:
                excludes.extend(["Miedema",
                                 "YangSolidSolution"])

            if mfs["percent_of_transition_metal_alloy"] < (1 - max_na_percent):
                excludes.extend(["TMetalFraction",
                                 "Miedema",
                                 "YangSolidSolution"])

            if mfs["PercentOfAllMetal"] > (1 - max_na_percent):
                excludes.extend(["CationProperty",
                                 "OxidationStates",
                                 "ElectronAffinity",
                                 "ElectronegativityDiff",
                                 "IonProperty"])
        except KeyError:
            warnings.warn("The metafeature dict does not contain all the"
                          "metafeatures for filtering featurizers for the "
                          "formula! Please call DatasetMetaFeatures first"
                          "to derive the metafeature dict.")

        # return dict {"formula_excludes": excludes}??
        return list(set(excludes))

    @staticmethod
    def structure_featurizer_excludes(mfs, max_na_percent=0.05):
        excludes = list()
        try:
            if mfs["percent_of_ordered_structures"] < (1 - max_na_percent):
                excludes.extend(["GlobalSymmetryFeatures"])

        except KeyError:
            warnings.warn("The metafeature dict does not contain all the"
                          "metafeatures for filtering featurizers for the "
                          "structure! Please call DatasetMetaFeatures first"
                          "to derive the metafeature dict.")
        return list(set(excludes))

    def auto_excludes(self, dataset_cols=("formula", "structure")):
        auto_excludes = list()
        auto_mfs = DatasetMetaFeatures(self.X, self.y).\
            auto_metafeatures(dataset_cols)
        for column in dataset_cols:
            mfs = auto_mfs.get("{}_metafeatures".format(column))
            excludes_fts = getattr(self,
                                   "{}_featurizer_excludes".format(column),
                                   None)
            auto_excludes.append(excludes_fts(mfs)
                                 if excludes_fts is not None else [])

        return list(set(auto_excludes))
