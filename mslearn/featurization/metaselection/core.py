import warnings

from mslearn.featurization.metaselection.metafeatures import formula_mfs_dict, \
    structure_mfs_dict

"""
Automatically filter some featurizers based on metafeatures calculated for
the dataset. 
"""
_supported_mfs_types = ("formula", "structure")


class DatasetMetaFeatures:
    def __init__(self, df):
        self.df = df

    def formula_metafeatures(self, formula_col="formula"):
        if formula_col in self.df.columns:
            mfs = dict()
            for mf, mf_class in formula_mfs_dict.items():
                mfs[mf] = mf_class.calc(self.df[formula_col])
            return {"formula_metafeatures": mfs}
        else:
            return {"formula_metafeatures": None}

    def structure_metafeatures(self, structure_col="structure"):
        if structure_col in self.df.columns:
            mfs = dict()
            for mf, mf_class in structure_mfs_dict.items():
                mfs[mf] = mf_class.calc(self.df[structure_col])
            return {"structure_metafeatures": mfs}
        else:
            return {"structure_metafeatures": None}

    def auto_metafeatures(self, **kwargs):
        auto_mfs = dict()
        for mfs_type in _supported_mfs_types:
            input_col = kwargs.get("{}_col".format(mfs_type), mfs_type)
            mfs_func = getattr(self, "{}_metafeatures".format(mfs_type), None)
            auto_mfs.update(mfs_func(input_col) if mfs_func is not None else {})

        return auto_mfs


class FeaturizerAutoFilter:
    """
    Given a dataset as a dataframe, return a featurizer set.
    Currently only support removing definitely useless featurizers.
    Cannot recommend featurizers based on the target.
    """
    def __init__(self, df, max_na_percent=0.05):
        self.df = df
        self.max_na_percent = max_na_percent

    @staticmethod
    def formula_featurizer_excludes(mfs, max_na_percent=0.05):
        excludes = list()
        try:
            if mfs["percent_of_all_nonmetal"] > max_na_percent:
                excludes.extend(["Miedema",
                                 "YangSolidSolution"])

            if mfs["percent_of_contain_trans_metal"] < (1 - max_na_percent):
                excludes.extend(["TMetalFraction",
                                 "Miedema",
                                 "YangSolidSolution"])

            if mfs["percent_of_all_metal"] > max_na_percent:
                excludes.extend(["CationProperty",
                                 "OxidationStates",
                                 "ElectronAffinity",
                                 "ElectronegativityDiff",
                                 "IonProperty"])
        except KeyError:
            warnings.warn("The metafeature dict does not contain all the "
                          "metafeatures for filtering featurizers for the "
                          "formula! Please call DatasetMetaFeatures first"
                          "to derive the metafeature dict.")

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

    def auto_excludes(self, **auto_mfs_kwargs):
        auto_excludes = list()
        auto_mfs = DatasetMetaFeatures(self.df).auto_metafeatures(
            **auto_mfs_kwargs)
        for mfs_type in _supported_mfs_types:
            mfs = auto_mfs.get("{}_metafeatures".format(mfs_type))
            if mfs is not None:
                exclude_fts = getattr(self,
                                      "{}_featurizer_excludes".format(mfs_type),
                                      None)
                auto_excludes.extend(exclude_fts(mfs, self.max_na_percent)
                                     if exclude_fts is not None else [])

        return list(set(auto_excludes))
