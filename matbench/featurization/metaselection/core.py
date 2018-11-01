import warnings

from matbench.featurization.metaselection.metafeatures import formula_mfs_dict, \
    structure_mfs_dict

"""
Automatically filter some featurizers based on metafeatures calculated for
the dataset. 
"""
_supported_mfs_types = ("formula", "structure")


class DatasetMetaFeatures:
    """
    Given a dataset as a dataframe, calculate pre-defined metafeatures.
    (see ..metafeatures for more details).
    Calling auto_metafeatures will return metafeatures of the dataset
    organized in a dict:
        {"formula_metafeatures": {"number_of_formulas": 2024,
                                  "percent_of_all_metal": 0.81,
                                  ...},
         "structure_metafeatures": {"number_of_structures": 2024,
                                  "percent_of_ordered_structures": 0.36,
                                  ...}}
    and if there is no corresponding column in the dataset, the value is None.

    These dataset metafeatures will be used in FeaturizerAutoFilter to remove
    some featurizers that definitely do not work for this dataset (returning
    nans more than the allowed max_na_percent).
    """
    def __init__(self, df):
        self.df = df

    def formula_metafeatures(self, formula_col="formula"):
        """
        Calculate formula-based metafeatures of the dataset.
        Args:
            formula_col(str): column name for formula

        Returns:
            (dict): {"formula_metafeatures": mfs/None}
        """
        if formula_col in self.df.columns:
            mfs = dict()
            for mf, mf_class in formula_mfs_dict.items():
                mfs[mf] = mf_class.calc(self.df[formula_col])
            return {"formula_metafeatures": mfs}
        else:
            return {"formula_metafeatures": None}

    def structure_metafeatures(self, structure_col="structure"):
        """
        Calculate structure-based metafeatures of the dataset.
        Args:
            structure_col(str): column name for structures

        Returns:
            (dict): {"structure_metafeatures": mfs/None}
        """
        if structure_col in self.df.columns:
            mfs = dict()
            for mf, mf_class in structure_mfs_dict.items():
                mfs[mf] = mf_class.calc(self.df[structure_col])
            return {"structure_metafeatures": mfs}
        else:
            return {"structure_metafeatures": None}

    def auto_metafeatures(self, **kwargs):
        """
        Automatically calculate metafeatures for all _supported_mfs_types.
        Args:
            kwargs: column names for formula/structures

        Returns:
            (dict): {"formula_metafeatures": formula_mfs/None,
                     "structure_metafeatures": structure_mfs/None}
        """
        auto_mfs = dict()
        for mfs_type in _supported_mfs_types:
            input_col = kwargs.get("{}_col".format(mfs_type), mfs_type)
            mfs_func = getattr(self, "{}_metafeatures".format(mfs_type), None)
            auto_mfs.update(mfs_func(input_col) if mfs_func is not None else {})

        return auto_mfs


class FeaturizerAutoFilter:
    """
    Given a dataset as a dataframe, return a list of featurizer names.
    Currently only support removing definitely useless featurizers.
    Cannot recommend featurizers based on the target.
    """
    def __init__(self, df, max_na_percent=0.05):
        self.df = df
        self.max_na_percent = max_na_percent

    @staticmethod
    def formula_featurizer_excludes(mfs, max_na_percent=0.05):
        """
        Determine the composition featurizers that are definitely do not work
        for this dataset (returning nans more than the allowed max_na_percent).
        Args:
            mfs: formula_metafeatures
            max_na_percent: max percent of nans allowed for the feature columns

        Returns:
            ([str]): list of removable composition featurizers
        """
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
        """
        Determine the structure featurizers that are definitely do not work
        for this dataset (returning nans more than the allowed max_na_percent).
        Args:
            mfs: structure_metafeatures
            max_na_percent: max percent of nans allowed for the feature columns

        Returns:
            ([str]): list of removable structure featurizers
        """
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
        """
        Automatically determine a list of removable featurizers based on
        metafeatures for all _supported_mfs_types.
        Args:
            auto_mfs_kwargs: kwargs for auto_metafeatures in DatasetMetafeatures

        Returns:
            ([str]): list of removable featurizers
        """
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
