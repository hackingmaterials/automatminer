"""
Automatically filter some featurizers based on metafeatures calculated for
the dataset. 
"""
import sys
import warnings
from automatminer.featurization.metaselection.metafeatures import \
    composition_mfs_dict, structure_mfs_dict

__author__ = ["Qi Wang <wqthu11@gmail.com>"]

_supported_mfs_types = ("composition", "structure")


def dataset_metafeatures(df, **mfs_kwargs):
    """
    Given a dataset as a dataframe, calculate pre-defined metafeatures.
    (see ..metafeatures for more details).
    Return metafeatures of the dataset organized in a dict:
        {"composition_metafeatures": {"number_of_compositions": 2024,
                                  "percent_of_all_metal": 0.81,
                                  ...},
         "structure_metafeatures": {"number_of_structures": 2024,
                                  "percent_of_ordered_structures": 0.36,
                                  ...}}
    if there is no corresponding column in the dataset, the value is None.

    These dataset metafeatures will be used in FeaturizerMetaSelector to remove
    some featurizers that definitely do not work for this dataset (returning
    nans more than the allowed max_na_frac).
    Args:
        df: input dataset as pd.DataFrame
        mfs_kwargs: kwargs for _composition/structure_metafeatures

    Returns:
        (dict): {"composition_metafeatures": composition_mfs/None,
                 "structure_metafeatures": structure_mfs/None}
        """
    dataset_mfs = dict()
    for mfs_type in _supported_mfs_types:
        input_col = mfs_kwargs.get("{}_col".format(mfs_type), mfs_type)
        mfs_func = getattr(sys.modules[__name__],
                           "_{}_metafeatures".format(mfs_type), None)
        dataset_mfs.update(mfs_func(df, input_col)
                           if mfs_func is not None else {})

    return dataset_mfs


def _composition_metafeatures(df, composition_col="composition"):
    """
    Calculate composition-based metafeatures of the dataset.
    Args:
        df: input dataset as pd.DataFrame
        composition_col(str): column name for compositions

    Returns:
        (dict): {"composition_metafeatures": mfs/None}
    """
    if composition_col in df.columns:
        mfs = dict()
        for mf, mf_class in composition_mfs_dict.items():
            mfs[mf] = mf_class.calc(df[composition_col])
        return {"composition_metafeatures": mfs}
    else:
        return {"composition_metafeatures": None}


def _structure_metafeatures(df, structure_col="structure"):
    """
    Calculate structure-based metafeatures of the dataset.
    Args:
        df: input dataset as pd.DataFrame
        structure_col(str): column name in the df for structures, as pymatgen
            IStructure or Structure

    Returns:
        (dict): {"structure_metafeatures": mfs/None}
    """
    if structure_col in df.columns:
        mfs = dict()
        for mf, mf_class in structure_mfs_dict.items():
            mfs[mf] = mf_class.calc(df[structure_col])
        return {"structure_metafeatures": mfs}
    else:
        return {"structure_metafeatures": None}


class FeaturizerMetaSelector:
    """
    Given a dataset as a dataframe, heuristically customize the featurizers.
    Currently only support removing definitely useless featurizers.
    Cannot recommend featurizers based on the target now.
    """

    def __init__(self, max_na_frac=0.05):
        self.max_na_frac = max_na_frac
        self.dataset_mfs = None
        self.excludes = None

    @staticmethod
    def composition_featurizer_excludes(mfs, max_na_frac=0.05):
        """
        Determine the composition featurizers that are definitely do not work
        for this dataset (returning nans more than the allowed max_na_frac).
        Args:
            mfs: composition_metafeatures
            max_na_frac: max percent of nans allowed for the feature columns

        Returns:
            ([str]): list of removable composition featurizers
        """
        excludes = list()
        try:
            if mfs["percent_of_all_nonmetal"] > max_na_frac:
                excludes.extend(["Miedema",
                                 "YangSolidSolution"])

            if mfs["percent_of_contain_trans_metal"] < (1 - max_na_frac):
                excludes.extend(["TMetalFraction",
                                 "Miedema",
                                 "YangSolidSolution"])

            if mfs["percent_of_all_metal"] > max_na_frac:
                excludes.extend(["CationProperty",
                                 "OxidationStates",
                                 "ElectronAffinity",
                                 "ElectronegativityDiff",
                                 "IonProperty"])
        except KeyError:
            warnings.warn("The metafeature dict does not contain all the "
                          "metafeatures for filtering featurizers for the "
                          "compositions! Please call DatasetMetaFeatures first"
                          "to derive the metafeature dict.")

        return list(set(excludes))

    @staticmethod
    def structure_featurizer_excludes(mfs, max_na_frac=0.05):
        """
        Determine the structure featurizers that are definitely do not work
        for this dataset (returning nans more than the allowed max_na_frac).
        Args:
            mfs: structure_metafeatures
            max_na_frac: max percent of nans allowed for the feature columns

        Returns:
            ([str]): list of removable structure featurizers
        """
        excludes = list()
        try:
            if mfs["percent_of_ordered_structures"] < (1 - max_na_frac):
                excludes.extend(["GlobalSymmetryFeatures"])

        except KeyError:
            warnings.warn("The metafeature dict does not contain all the"
                          "metafeatures for filtering featurizers for the "
                          "structures! Please call DatasetMetaFeatures first"
                          "to derive the metafeature dict.")
        return list(set(excludes))

    def auto_excludes(self, df, **mfs_kwargs):
        """
        Automatically determine a list of removable featurizers based on
        metafeatures for all _supported_mfs_types.
        Args:
            auto_mfs_kwargs: kwargs for auto_metafeatures in DatasetMetafeatures

        Returns:
            ([str]): list of removable featurizers
        """
        auto_excludes = list()
        self.dataset_mfs = dataset_metafeatures(df, **mfs_kwargs)
        for mfs_type in _supported_mfs_types:
            mfs = self.dataset_mfs.get("{}_metafeatures".format(mfs_type))
            if mfs is not None:
                exclude_fts = getattr(self,
                                      "{}_featurizer_excludes".format(mfs_type),
                                      None)
                auto_excludes.extend(exclude_fts(mfs, self.max_na_frac)
                                     if exclude_fts is not None else [])

        self.excludes = list(set(auto_excludes))
        return self.excludes
