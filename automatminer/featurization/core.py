from pymatgen import Composition
from matminer.featurizers.conversions import StrToComposition, DictToObject, \
    StructureToComposition, StructureToOxidStructure, \
    CompositionToOxidComposition
from matminer.featurizers.function import FunctionFeaturizer

from automatminer.utils.package_tools import check_fitted, set_fitted
from automatminer.base import DataframeTransformer, LoggableMixin
from automatminer.featurization.sets import CompositionFeaturizers, \
    StructureFeaturizers, BSFeaturizers, DOSFeaturizers
from automatminer.utils.package_tools import AutomatminerError

__author__ = ["Alex Dunn <ardunn@lbl.gov>",
              "Alireza Faghaninia <alireza@lbl.gov>",
              "Qi Wang <wqthu11@gmail.com>"]

_COMMON_COL_ERR_STR = "composition_col/structure_col/bandstructure_col/dos_col"


class AutoFeaturizer(DataframeTransformer, LoggableMixin):
    """
    Automatically featurize a dataframe.

    Use this object first by calling fit, then by calling transform.

    AutoFeaturizer requires you to specify the column names for each type of
        featurization, or just use the defaults:

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
        featurizers (dict): Use this option if you want to manually specify
            the featurizers to use. Keys are the featurizer types you want
            applied (e.g., "structure", "composition"). The corresponding values
            are lists of featurizer objects you want for each featurizer type.

            Example:
                 {"composition": [ElementProperty.from_preset("matminer"),
                                  EwaldEnergy()]
                  "structure": [BagofBonds(), GlobalSymmetryFeatures()]}

        exclude ([str]): Class names of featurizers to exclude. Only used if
            you use a preset.
        ignore_cols ([str]): Column names to be ignored/removed from any
            dataframe undergoing fitting or transformation. If columns are
            not ignored, they may be used later on for learning.
        ignore_errors (bool): If True, each featurizer will ignore all errors
            during featurization.
        drop_inputs (bool): Drop the columns containing input objects for
            featurization after they are featurized.
        guess_oxistates (bool): If True, try to decorate sites with oxidation
            state.
        multiiindex (bool): If True, returns a multiindexed dataframe. Not
            recommended for use in MatPipe.
        n_jobs (int): The number of parallel jobs to use during featurization
            for each featurizer. Default is n_cores
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default automatminer logger will
            be used. If set to False, then no logging will occur.

    Attributes:
        These attributes are set during fitting

        featurizers (dict): Same format as input dictionary in Args. Values
            contain the actual objects being used for featurization.
        features (dict): The features generated from the application of all
            featurizers.
        auto_featurizer (bool): whether the featurizers are set automatically,
            or passed by the users.
    """

    def __init__(self, preset=None, featurizers=None, exclude=None,
                 functionalize=False, max_na_frac=0.05, ignore_cols=None,
                 ignore_errors=True, drop_inputs=True,
                 guess_oxistates=True, multiindex=False, n_jobs=None,
                 logger=True, composition_col="composition",
                 structure_col="structure",
                 bandstructure_col="bandstructure", dos_col="dos"):

        if featurizers and preset:
            raise AutomatminerError("Featurizers and preset were both set. "
                                    "Please either use a preset ('best', 'all',"
                                    " 'fast') or set featurizers manually.")
        if not featurizers and not preset:
            raise AutomatminerError("Please specify set(s) of featurizers to "
                                    "use either through the featurizers"
                                    "argument or through the preset argument.")

        self.preset = "best" if preset is None else preset
        self._logger = self.get_logger(logger)
        self.featurizers = featurizers
        self.exclude = exclude if exclude else []
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
        self.composition_col = composition_col
        self.structure_col = structure_col
        self.bandstruct_col = bandstructure_col
        self.dos_col = dos_col

        _supported_featurizers = {composition_col: CompositionFeaturizers,
                                  structure_col: StructureFeaturizers,
                                  bandstructure_col: BSFeaturizers,
                                  dos_col: DOSFeaturizers}

        # user-set featurizers
        if self.featurizers:
            if not isinstance(self.featurizers, dict):
                raise TypeError("Featurizers must be a dictionary with keys"
                                "matching your {}".format(_COMMON_COL_ERR_STR))

            invalid_ftypes = [f for f in self.featurizers.keys() if
                              f not in _supported_featurizers.keys()]
            if invalid_ftypes:
                raise KeyError("The following keys were specified as featurizer"
                               " types but were not set in {}"
                               "".format(_COMMON_COL_ERR_STR))

            for ftype, fset in self.featurizers.items():
                _allowed = [f.__class__.__name__ for f in
                            _supported_featurizers[ftype]().all]
                for f in fset:
                    if f.__class__.__name__ not in _allowed:
                        raise ValueError(
                            "The {} featurizer {} is not supported by "
                            "AutoFeaturizer. Try updating your version of "
                            "automatminer and matminer.".format(ftype, f))

        # auto-set featurizers
        else:
            featurizers = dict()
            for featurizer_type in _supported_featurizers.keys():
                featurizer_set = _supported_featurizers[featurizer_type]
                featurizers[featurizer_type] = getattr(
                    featurizer_set(exclude=self.exclude), self.preset)
            self.featurizers = featurizers

        # Check if any featurizers need fitting (useful for MatPipe)
        needs_fit = False
        fittable_fs = StructureFeaturizers().need_fit
        fittable_fcls = set([f.__class__.__name__ for f in fittable_fs])

        # Currently structure featurizers are the only featurizer types which
        # can be fittable
        for f in self.featurizers[self.structure_col]:
            if f.__class__.__name__ in fittable_fcls:
                needs_fit = True
                break
        self.needs_fit = needs_fit

    @set_fitted
    def fit(self, df, target):
        """
        Fit all featurizers to the df.

        WARNING: for fitting to work, the dataframe must contain the
        corresponding *col keys set in __init__. So for composition
        featurization to work, you must have the composition_col in the
        dataframe.

            Composition features: composition_col - pymatgen Composition objs
            Structure features: structure_col - pymatgen Structure objs
            Bandstructure features: bandstructure_col - pymatgen Bandstructure
                objs
            DOS features: dos_col - pymatgen DOS objs

        Args:
            df (pandas.DataFrame): A dataframe containing at least one of the
                keys listed above. The column defined by that key should have
                the corresponding types of objects in it.
            target (str): The ML target key in the dataframe.

        Returns:
            (AutoFeaturizer): self
        """
        df = self._prescreen_df(df, inplace=True)
        df = self._add_composition_from_structure(df)

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
                                     "".format(f.__class__.__name__,
                                               df.shape[0]))
            else:
                self.logger.info("Featurizer type {} not in the dataframe to be"
                                 " fitted. Skipping...".format(featurizer_type))

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

        # todo: structure to oxidstructure + comp2oxidcomp can get called twice
        # todo: by _tidy_column, can be fixed with overriding fit_transform
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
        # todo: Make the following conversions more robust (no [0] type checking)
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
                df[featurizer_type] = [Composition.from_dict(d) for d in
                                       df[featurizer_type]]

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
            dto = DictToObject(overwrite_data=True,
                               target_col_id=featurizer_type)
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
            2. compositions are not available (unless overwrite)
            3. composition features are actually desired. (deduced from whether
                composition featurizers are present in self.featurizers).

        Args:
            df (pandas.DataFrame): May or may not contain composition column.
            overwrite (bool): Whether to overwrite the composition column if it
                already exists.

        Returns:
            df (pandas.DataFrame): Contains composition column if desired
        """
        if self.structure_col in df.columns \
                and (self.composition_col not in df.columns or overwrite) \
                and self.composition_col in self.featurizers:

            if self.composition_col in df.columns:
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
    from matminer.datasets.dataset_retrieval import load_dataset, \
        get_available_datasets

    print(get_available_datasets())
    df = load_dataset("elastic_tensor_2015").rename(
        columns={"formula": "composition"}).iloc[:20]
    af = AutoFeaturizer(functionalize=True, preset="best")
    print(df)
    df = af.fit_transform(df, "K_VRH")
    print(df.describe())
