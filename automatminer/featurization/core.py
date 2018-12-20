from pymatgen import Composition
from matminer.featurizers.conversions import StrToComposition, DictToObject, \
    StructureToComposition, StructureToOxidStructure, \
    CompositionToOxidComposition
from matminer.featurizers.conversions import ConversionFeaturizer
from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.function import FunctionFeaturizer

from automatminer.utils.package_tools import check_fitted, set_fitted
from automatminer.base import DataframeTransformer, LoggableMixin
from automatminer.featurization.sets import CompositionFeaturizers, \
    StructureFeaturizers, BSFeaturizers, DOSFeaturizers
from automatminer.featurization.metaselection.core import FeaturizerMetaSelector
from automatminer.utils.package_tools import AutomatminerError

__author__ = ["Alex Dunn <ardunn@lbl.gov>",
              "Alireza Faghaninia <alireza@lbl.gov>",
              "Qi Wang <wqthu11@gmail.com>"]

_supported_featurizer_types = {"composition": CompositionFeaturizers,
                               "structure": StructureFeaturizers,
                               "bandstructure": BSFeaturizers,
                               "dos": DOSFeaturizers}

_composition_aliases = ["comp", "Composition", "composition", "COMPOSITION",
                        "comp.", "formula", "chemical composition", "compositions"]
_structure_aliases = ["structure", "struct", "struc", "struct.", "structures",
                      "STRUCTURES", "Structure", "structures", "structs"]
_bandstructure_aliases = ["bandstructure", "bs", "bsdos", "BS", "BSDOS",
                          "Bandstructure"]
_dos_aliases = ["density of states", "dos", "DOS", "Density of States"]
_aliases = _composition_aliases + _structure_aliases + _bandstructure_aliases + _dos_aliases


class AutoFeaturizer(DataframeTransformer, LoggableMixin):
    """
    Automatically featurize a dataframe.

    Use this object first by calling fit, then by calling transform.

    AutoFeaturizer works with a fixed set of possible column names.
        "composition": To use composition features
        "structure": To use structure features
        "bandstructure": To use bandstructure features
        "dos": To use density of states features

    The featurizers corresponding to each featurizer type cannot be used if
    the correct column name is not present.

    Args:
        preset (str): "best" or "fast" or "all". Determines by preset the
            featurizers that should be applied. See the Featurizer sets for
            specifics of best/fast/all. Default is "best". Incompatible with
            the featurizers arg.
        featurizers (dict): Values are the featurizer types you want applied
            (e.g., "structure", "composition"). The corresponding values
            are lists of featurizer objects you want applied for each featurizer
            type.

            Example:
                 {"composition": [ElementProperty.from_preset("matminer"),
                                  EwaldEnergy()]
                  "structure": [BagofBonds(), GlobalSymmetryFeatures()]}
            Valid keys for each featurizer type are given in the *_aliases
            constants above.
        exclude ([str]): Class names of featurizers to exclude. Only used if
            your own featurizer dict is NOT passed.
        use_metaselector (bool): Whether to use FeaturizerMetaSelector to remove
            featurizers that will return a higher nan fraction than max_na_frac
            (as below) for the dataset. Only used if own featurizer dict is
            NOT passed.
        max_na_frac (float): The maximum fraction (0.0 - 1.0) of samples for a
            given feature allowed. Featurizers that definitely return a higher
            nan fraction are automatically removed by FeaturizerMetaSelector.
            Only used if own featurizer dict is NOT passed and use_metaselector
            is True.
        ignore_cols ([str]): Column names to be ignored/removed from any
            dataframe undergoing fitting or transformation.
        ignore_errors (bool): If True, each featurizer will ignore all errors
            during featurization.
        drop_inputs (bool): Drop the columns containing input objects for
            featurization (e.g., drop composition column following featurization).
        guess_oxistates (bool): If True, try to decorate sites with oxidation
            state.
        multiiindex (bool): If True, returns a multiindexed dataframe.
        n_jobs (int): The number of parallel jobs to use during featurization
            for each featurizer. Default is n_cores
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default automatminer logger will be
            used. If set to False, then no logging will occur.

    Attributes:
        These attributes are set during fitting

        featurizers (dict): Same format as input dictionary in Args. Values
            contain the actual objects being used for featurization.
        features (dict): The features generated from the application of all
            featurizers.
        auto_featurizer (bool): whether the featurizers are set automatically,
            or passed by the users.
        metaselector (object): the FeaturizerMetaSelector class if metaselection
            is used during featurization. The dataset metafeatures and
            auto-excluded featurizers can be accessed by self.metaselector.
            dataset_mfs and self.metaselector.excludes.

    """

    def __init__(self, preset=None, featurizers=None, exclude=None,
                 use_metaselector=False, functionalize=False, max_na_frac=0.05,
                 ignore_cols=None, ignore_errors=True, drop_inputs=True,
                 guess_oxistates=True, multiindex=False, n_jobs=None,
                 logger=True, composition_col="composition", structure_col="structure",
                 bandstructure_col="bandstructure", dos_col="dos"):

        if featurizers and preset:
            raise AutomatminerError("Featurizers and preset were both set. "
                                    "Please either use a preset ('best', 'all',"
                                    " 'fast') or set featurizers manually.")

        self.preset = "best" if preset is None else preset
        self._logger = self.get_logger(logger)
        self.featurizers = featurizers
        self.exclude = exclude if exclude else []
        self.use_metaselector = use_metaselector
        self.functionalize = functionalize
        self.max_na_percent = max_na_frac
        self.ignore_cols = ignore_cols or []
        self.is_fit = False
        self.ignore_errors = ignore_errors
        self.drop_inputs = drop_inputs
        self.multiindex = multiindex
        self.n_jobs = n_jobs
        self.guess_oxistates = guess_oxistates
        self.features = []
        self.auto_featurizer = True if self.featurizers is None else False
        self.metaselector = None
        self.composition_col = composition_col
        self.structure_col = structure_col
        self.bandstruct_col = bandstructure_col
        self.dos_col = dos_col

    @set_fitted
    def fit(self, df, target):
        """
        Fit all featurizers to the df.

        WARNING: for fitting to work, the dataframe must contain the following
        keys and column value types for each kind of featurization:

            Composition features: "composition" - strings or pymatgen
                Composition objects
            Structure features: "structure" - pymatgen Structure objects
            Bandstructure features: "bandstructure" - pymatgen BandStructure objects
            DOS features: "dos" - pymatgen DOS objects

        Args:
            df (pandas.DataFrame): A dataframe containing at least one of the keys
                listed above. The column defined by that key should have the corresponding
                types of objects in it.
            target (str): The ML target key in the dataframe.

        Returns:
            (AutoFeaturizer): self
        """
        df = self._prescreen_df(df, inplace=True)
        df = self._add_composition_from_structure(df)
        self._customize_featurizers(df)

        for featurizer_type, featurizers in self.featurizers.items():
            if not featurizers:
                self.logger.info("No {} featurizers being used."
                                 "".format(featurizer_type))
            if featurizer_type in df.columns:
                df = self._tidy_column(df, featurizer_type)
                for f in featurizers:
                    f.fit(df[featurizer_type].tolist())
                    f.set_n_jobs(self.n_jobs)
                    self.features += f.feature_labels()
                    self.logger.info("Fit {} to {} samples in dataframe."
                                     "".format(f.__class__.__name__, df.shape[0]))
        return self

    @check_fitted
    def transform(self, df, target, tidy_column=True):
        """
        Decorate a dataframe containing composition, structure, bandstructure,
        and/or DOS objects with descriptors.

        Args:
            df (pandas.DataFrame): The dataframe not containing features.
            target (str): The ML-target property contained in the df.

        Returns:
            df (pandas.DataFrame): Transformed dataframe containing features.
        """

        #todo: structure to oxidstructure + comp2oxidcomp can get called twice by _tidy_column, can be fixed with overriding fit_transform
        df = self._prescreen_df(df, inplace=True)
        df = self._add_composition_from_structure(df)

        for featurizer_type, featurizers in self.featurizers.items():
            if featurizer_type in df.columns:
                if tidy_column:
                    df = self._tidy_column(df, featurizer_type)

                for f in featurizers:
                    df = f.featurize_dataframe(df, featurizer_type,
                                               ignore_errors=self.ignore_errors,
                                               multiindex=self.multiindex)
                df = df.drop(columns=[featurizer_type])
            else:
                self.logger.info("Featurizer type {} not in the dataframe. "
                                 "Skipping...".format(featurizer_type))
        if self.functionalize:
            ff = FunctionFeaturizer()
            cols = df.columns.tolist()
            for ft in self.featurizers.keys():
                if ft in cols:
                    cols.pop(ft)
            df = ff.fit_featurize_dataframe(df, cols,
                                            ignore_errors=self.ignore_errors,
                                            multiindex=self.multiindex)

        return df

    @set_fitted
    def fit_transform(self, df, target):
        """
        Fit and transform the dataframe all as one, without reassigning
        oxidation states (if valid).

        Args:
            df (pandas.DataFrame): The dataframe not containing features.
            target (str): The ML-target property contained in the df.

        Returns:
            df (pandas.DataFrame): Transformed dataframe containing features.
        """
        return self.fit(df, target).transform(df, target, tidy_column=False)

    def _prescreen_df(self, df, inplace=True):
        """
        Pre-screen a dataframe.

        Args:
            df (pandas.DataFrame): The dataframe to be screened.
            inplace (bool): Manipulate the df in-place in memory

        Returns:
            df (pandas.DataFrame) The screened dataframe.

        """
        if not inplace:
            df = df.copy(deep=True)
        if self.ignore_errors is not None:
            for col in self.ignore_cols:
                if col in df:
                    df = df.drop([col], axis=1)
        return df

    def _customize_featurizers(self, df):
        """
        Customize the featurizers that will be used in featurization, stored
        in self.featurizers.
        If users have passed the featurizers, just use them and normalize
        the names from the aliases; If users have not passed the featurizers,
        will auto-set the featurizers, and if use_metaselection is True, will
        use FeaturizerMetaSelector to remove the featurizers that return a
        higher nan fraction than self.max_na_frac for the dataset.
        Args:
            df (pandas.DataFrame)

        """
        # auto-set featurizers
        if not self.featurizers:
            self.auto_featurizer = True
            self.featurizers = dict()
            # use FeaturizerMetaSelector to get removable featurizers
            if self.use_metaselector:
                self.logger.info("Running metaselector.")
                self.metaselector = FeaturizerMetaSelector(self.max_na_percent)
                auto_exclude = self.metaselector.auto_excludes(df)
                if auto_exclude:
                    self.logger.info("Based on metafeatures of the dataset, "
                                     "these featurizers are excluded for "
                                     "returning nans more than the "
                                     "max_na_percent of {}: {}".
                                     format(self.max_na_percent, auto_exclude))
                    self.exclude.extend(auto_exclude)

            for featurizer_type in _supported_featurizer_types.keys():
                if featurizer_type in df.columns:
                    featurizer_set = _supported_featurizer_types[featurizer_type]
                    self.featurizers[featurizer_type] = \
                        getattr(featurizer_set(exclude=self.exclude),
                                self.preset)
                else:
                    self.logger.info("Featurizer type {} not in the dataframe "
                                     "to be fitted. Skipping...".
                                     format(featurizer_type))
        # user-set featurizers
        else:
            if not isinstance(self.featurizers, dict):
                raise TypeError("Featurizers must be a dictionary with keys"
                                "of 'composition', 'structure', 'bandstructure', "
                                "and 'dos' and values of corresponding lists of "
                                "featurizers.")
            else:
                for ftype in self.featurizers:
                    # Normalize the names from the aliases
                    if ftype in _composition_aliases:
                        self.featurizers[self.composition_col] = self.featurizers.pop(ftype)
                    elif ftype in _structure_aliases:
                        self.featurizers[self.structure_col] = self.featurizers.pop(ftype)
                    elif ftype in _bandstructure_aliases:
                        self.featurizers[self.bandstruct_col] = self.featurizers.pop(ftype)
                    elif ftype in _dos_aliases:
                        self.featurizers[self.dos_col] = self.featurizers.pop(ftype)
                    else:
                        raise ValueError(
                            "The featurizers dict key {} is not a valid "
                            "featurizer type. Please choose from {}".format(
                                ftype, _aliases))

    def _tidy_column(self, df, featurizer_type):
        """
        Various conversions to homogenize columns for featurization input.
        For example, take a column of compositions and ensure they are decorated
        with oxidation states, are not strings, etc.

        Args:
            df (pandas.DataFrame)
            featurizer_type: The key defining the featurizer input. For example,
                composition featurizers should have featurizer_type of
                "composition".

        Returns:
            df (pandas.DataFrame): DataFrame with featurizer_type column
                ready for featurization.
        """
        # todo: Make the following conversions more robust (no lazy [0] type checking)
        if featurizer_type == self.composition_col:
            # Convert formulas to composition objects
            if isinstance(df[featurizer_type].iloc[0], str):
                self.logger.info("Compositions detected as strings. Attempting "
                                 "conversion to Composition objects...")
                stc = StrToComposition(overwrite_data=True,
                                       target_col_id=featurizer_type)
                df = stc.featurize_dataframe(df, featurizer_type,
                                             multiindex=self.multiindex,
                                             ignore_errors=True)

            elif isinstance(df[featurizer_type].iloc[0], dict):
                self.logger.info("Compositions detected as dicts. Attempting "
                                 "conversion to Composition objects...")
                df[featurizer_type] = [Composition.from_dict(d) for d in df[featurizer_type]]

            # Convert non-oxidstate containing comps to oxidstate comps
            if self.guess_oxistates:
                self.logger.info("Guessing oxidation states of compositions, as"
                                 " they were not present in input.")
                cto = CompositionToOxidComposition(
                    target_col_id=featurizer_type, overwrite_data=True,
                    return_original_on_error=True, max_sites=-50)
                try:
                    df = cto.featurize_dataframe(df, featurizer_type,
                                                 multiindex=self.multiindex)
                except Exception as e:
                    self.logger.info("Could not decorate oxidation states due "
                                     "to {}. Excluding featurizers based on "
                                     "composition oxistates".format(e))
                    classes_require_oxi = [c.__class__.__name__ for c in
                                           CompositionFeaturizers().need_oxi]
                    self.exclude.extend(classes_require_oxi)

        else:
            # Convert structure/bs/dos dicts to objects (robust already)
            self.logger.info("{} detected as strings. Attempting "
                             "conversion to {} objects..."
                             "".format(featurizer_type, featurizer_type))
            dto = DictToObject(overwrite_data=True, target_col_id=featurizer_type)
            df = dto.featurize_dataframe(df, featurizer_type)

            # Decorate with oxidstates
            if featurizer_type == self.structure_col and self.guess_oxistates:
                self.logger.info("Guessing oxidation states of structures, as "
                                 "they were not present in input.")
                sto = StructureToOxidStructure(
                    target_col_id=featurizer_type, overwrite_data=True,
                    return_original_on_error=True, max_sites=-50)
                try:
                    df = sto.featurize_dataframe(df, featurizer_type,
                                             multiindex=self.multiindex)
                except Exception as e:
                    self.logger.info("Could not decorate oxidation states on"
                                        " structures due to {}.".format(e))
        return df

    def _add_composition_from_structure(self, df, overwrite=True):
        """
        Automatically deduce compositions from structures if:
            1. structures are available
            2. composition features are actually desired. (deduced from whether
                composition featurizers are present in self.featurizers).
        Args:
            df (pandas.DataFrame): May or may not contain composition column.
            overwrite (bool): Whether to overwrite the composition column if it
                already exists.

        Returns:
            df (pandas.DataFrame): Contains composition column if desired
        """
        if (self.structure_col in df.columns and
                (self.auto_featurizer or (set(_composition_aliases) and
                                          set(self.featurizers.keys())))
                and (self.composition_col not in df.columns or overwrite)):

            if "composition" in df.columns:
                self.logger.info("composition column already exists, "
                                 "overwriting with composition from structure.")
            else:
                self.logger.debug("Adding compositions from structures.")

            df = self._tidy_column(df, self.structure_col)

            # above tidy column will add oxidation states, these oxidation
            # states will then be transferred to composition.
            struct2comp = StructureToComposition(
                target_col_id=self.composition_col, overwrite_data=overwrite)
            df = struct2comp.featurize_dataframe(df, self.structure_col)
        return df


if __name__ == "__main__":
    from matminer.datasets.dataset_retrieval import load_dataset, get_available_datasets

    print(get_available_datasets())
    df = load_dataset("elastic_tensor_2015").rename(columns={"formula": "composition"}).iloc[:20]
    af = AutoFeaturizer(functionalize=True)
    print(df)
    df = af.fit_transform(df, "K_VRH")
    print(df.describe())
