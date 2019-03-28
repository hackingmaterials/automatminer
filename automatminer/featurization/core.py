"""
Classes for automatic featurization and core featurizer functionality.
"""

import os
import math

import pandas as pd
from pymatgen import Composition
from matminer.featurizers.conversions import StrToComposition, DictToObject, \
    StructureToComposition, StructureToOxidStructure, \
    CompositionToOxidComposition
from matminer.featurizers.function import FunctionFeaturizer

from automatminer.utils.log import log_progress, AMM_LOG_FIT_STR, \
    AMM_LOG_TRANSFORM_STR
from automatminer.utils.pkg import check_fitted, set_fitted
from automatminer.base import DFTransformer, LoggableMixin
from automatminer.featurization.sets import CompositionFeaturizers, \
    StructureFeaturizers, BSFeaturizers, DOSFeaturizers
from automatminer.utils.pkg import AutomatminerError
from automatminer.utils.ml import regression_or_classification

__author__ = ["Alex Dunn <ardunn@lbl.gov>",
              "Alireza Faghaninia <alireza@lbl.gov>",
              "Qi Wang <wqthu11@gmail.com>"]

_COMMON_COL_ERR_STR = "composition_col/structure_col/bandstructure_col/dos_col"


class AutoFeaturizer(DFTransformer, LoggableMixin):
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
        cache_src (str): An absolute path to a json file holding feature
            information. If file exists, will read features (loc indexwise)
            from this file instead of featurizing. If this file does not exist,
            AutoFeaturizer will featurize normally, then save the features to a
            new file. Only features (not featurizer input objects) will be saved
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
            contain the actual objects being used for featurization. Featurizers
            can be removed if check_validity=True and the featurizer is not
            valid for more than self.min_precheck_frac fraction of the fitting
            dataset.
        features (dict): The features generated from the application of all
            featurizers.
        auto_featurizer (bool): whether the featurizers are set automatically,
            or passed by the users.
        fitted_input_df (pd.DataFrame): The dataframe which was fitted on
        converted_input_df (pd.DataFrame): The converted dataframe which
            was fitted on (i.e., strings converted to compositions).

        Attributes not set during fitting and not specified by arguments:

        min_precheck_frac (float): The minimum fraction of a featuriser's input
            that can be valid (via featurizer.precheck(data).
    """

    def __init__(self, cache_src=None, preset=None, featurizers=None,
                 exclude=None, functionalize=False, ignore_cols=None,
                 ignore_errors=True, drop_inputs=True, guess_oxistates=True,
                 multiindex=False, do_precheck=True, n_jobs=None,
                 logger=True, composition_col="composition",
                 structure_col="structure", bandstructure_col="bandstructure",
                 dos_col="dos"):

        if featurizers and preset:
            raise AutomatminerError("Featurizers and preset were both set. "
                                    "Please either use a preset ('best', 'all',"
                                    " 'fast') or set featurizers manually.")
        if not featurizers and not preset:
            raise AutomatminerError("Please specify set(s) of featurizers to "
                                    "use either through the featurizers"
                                    "argument or through the preset argument.")

        self.cache_src = cache_src
        self.preset = "best" if preset is None else preset
        self._logger = self.get_logger(logger)
        self.featurizers = featurizers
        self.exclude = exclude if exclude else []
        self.functionalize = functionalize
        self.ignore_cols = ignore_cols or []
        self.is_fit = False
        self.fitted_input_df = None
        self.converted_input_df = None
        self.ignore_errors = ignore_errors
        self.drop_inputs = drop_inputs
        self.multiindex = multiindex
        self.do_precheck = do_precheck
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
        self.fittable_fcls = set([f.__class__.__name__ for f in fittable_fs])

        # Currently structure featurizers are the only featurizer types which
        # can be fittable
        for f in self.featurizers[self.structure_col]:
            if f.__class__.__name__ in self.fittable_fcls:
                needs_fit = True
                break
        self.needs_fit = needs_fit

        if self.needs_fit and self.cache_src:
            self.logger.warn(self._log_prefix +
                             "Using cached features on fittable featurizers! "
                             "Please make sure you are not benchmarking with "
                             "these options enabled; it is likely you will be"
                             "leaking data (i.e., features) from the testing"
                             "sets into the training.")

        if self.cache_src and "json" not in self.cache_src.lower():
            raise ValueError("The cache_src filename does not contain json."
                             "JSON is the required file type for featurizer"
                             "caching.")

        self.min_precheck_frac = 0.9

    @log_progress(AMM_LOG_FIT_STR)
    @set_fitted
    def fit(self, df, target):
        """
        Fit all featurizers to the df and remove featurizers which are out of
        scope or otherwise cannot be robustly applied to the dataset.

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
        if self.cache_src and os.path.exists(self.cache_src):
            self.logger.info(self._log_prefix +
                             "Cache {} found. Fit aborted."
                             "".format(self.cache_src))
            return self

        self.fitted_input_df = df
        df = self._prescreen_df(df, inplace=True)
        df = self._add_composition_from_structure(df)

        for featurizer_type, featurizers in self.featurizers.items():
            if not featurizers:
                self.logger.info(
                    self._log_prefix +
                    "No {} featurizers being used.".format(featurizer_type))
                continue

            if featurizer_type in df.columns:
                df = self._tidy_column(df, featurizer_type)

                # Remove invalid featurizers by looking at valid_fraction
                if self.do_precheck:
                    self.logger.debug(self._log_prefix +
                                      "Prechecking featurizers.")
                    invalid_featurizers = []
                    for f in featurizers:
                        try:
                            frac = f.precheck_dataframe(df, featurizer_type,
                                                        return_frac=True)
                            if frac < self.min_precheck_frac:
                                invalid_featurizers.append(f)
                                msg = "Will remove {} because it's fraction " \
                                      "passing the precheck for this " \
                                      "dataset ({}) was less than the minimum" \
                                      " ({})" \
                                      "".format(f.__class__.__name__, frac,
                                                self.min_precheck_frac)
                                self.logger.info(self._log_prefix + msg)
                        except (AttributeError, ValueError, KeyError) as E:
                            self.logger.warning(
                                self._log_prefix +
                                "{} precheck failed with {}. Ignoring...."
                                "".format(f, E))
                    self.featurizers[featurizer_type] = [f for f in featurizers
                                                         if f not in
                                                         invalid_featurizers]
                    featurizers = self.featurizers[featurizer_type]

                # Fit the featurizers
                for f in featurizers:
                    log_fit = False
                    if featurizer_type == self.structure_col:
                        if f.__class__.__name__ in self.fittable_fcls:
                            log_fit = True
                    if log_fit:
                        self.logger.info(
                            self._log_prefix +
                            "Fitting {}.".format(f.__class__.__name__))

                    f.fit(df[featurizer_type].tolist())
                    f.set_n_jobs(self.n_jobs)
                    self.features += f.feature_labels()

                    if log_fit:
                        self.logger.info(
                            self._log_prefix +
                            "Fit {} to {} samples in dataframe."
                            "".format(f.__class__.__name__, df.shape[0]))
            else:
                self.logger.info(self._log_prefix +
                                 "Featurizer type {} not in the dataframe to be"
                                 " fitted. Skipping...".format(featurizer_type))
        self.converted_input_df = df
        return self

    @log_progress(AMM_LOG_TRANSFORM_STR)
    @check_fitted
    def transform(self, df, target):
        """
        Decorate a dataframe containing composition, structure, bandstructure,
        and/or DOS objects with descriptors.

        Args:
            df (pandas.DataFrame): The dataframe not containing features.
            target (str): The ML-target property contained in the df.

        Returns:
            df (pandas.DataFrame): Transformed dataframe containing features.
        """
        if self.cache_src and os.path.exists(self.cache_src):
            self.logger.debug(self._log_prefix +
                              "Reading cache_src {}".format(self.cache_src))
            cached_df = pd.read_json(self.cache_src)
            if not all([loc in cached_df.index for loc in df.index]):
                raise AutomatminerError("Feature cache does not contain all "
                                        "entries (by DataFrame index) needed "
                                        "to transform the input df.")
            else:
                cached_subdf = cached_df.loc[df.index]
                if target in cached_subdf.columns:
                    if target not in df.columns:
                        self.logger.warn(
                            self._log_prefix +
                            "Target not present in both cached df and input df."
                            " Cannot perform comparison to ensure index match.")
                    else:
                        cached_targets = cached_subdf[target]
                        input_targets = df[target]
                        cached_type = regression_or_classification(
                            cached_targets)
                        input_type = regression_or_classification(input_targets)
                        if cached_type != input_type:
                            raise AutomatminerError(
                                "Cached targets appear to be '{}' type, while "
                                "input targets appear to be '{}'."
                                "".format(cached_type, input_type))

                        problems = {}
                        for ix in input_targets.index:
                            iv = input_targets[ix]
                            cv = cached_targets[ix]
                            if iv != cv:
                                try:
                                    if not math.isclose(iv, cv):
                                        problems[ix] = [iv, cv]
                                except TypeError:
                                    pass
                        if problems:
                            self.logger.warning(
                                self._log_prefix +
                                "Mismatch between cached targets and input "
                                "targets: \n{}".format(problems))

                self.logger.info(self._log_prefix +
                                 "Restored {} features on {} samples from "
                                 "cache {}".format(len(cached_subdf.columns),
                                                   len(df.index),
                                                   self.cache_src))
                return cached_subdf
        else:
            transforming_on_fitted = df is self.fitted_input_df
            df = self._prescreen_df(df, inplace=True)

            if transforming_on_fitted:
                df = self.converted_input_df
            else:
                df = self._add_composition_from_structure(df)

            for featurizer_type, featurizers in self.featurizers.items():
                if featurizer_type in df.columns:
                    if not transforming_on_fitted:
                        df = self._tidy_column(df, featurizer_type)

                    for f in featurizers:
                        self.logger.info(self._log_prefix +
                                         "Featurizing with {}."
                                         "".format(f.__class__.__name__))
                        df = f.featurize_dataframe(
                            df,
                            featurizer_type,
                            ignore_errors=self.ignore_errors,
                            multiindex=self.multiindex,
                            inplace=False)
                    if self.drop_inputs:
                        df = df.drop(columns=[featurizer_type])
                else:
                    self.logger.info(self._log_prefix +
                                     "Featurizer type {} not in the dataframe. "
                                     "Skipping...".format(featurizer_type))
            if self.functionalize:
                ff = FunctionFeaturizer()
                cols = df.columns.tolist()
                for ft in self.featurizers.keys():
                    if ft in cols:
                        cols.pop(ft)
                df = ff.fit_featurize_dataframe(df, cols,
                                                ignore_errors=self.ignore_errors,
                                                multiindex=self.multiindex,
                                                inplace=False)
            if self.cache_src and not os.path.exists(self.cache_src):
                df.to_json(self.cache_src)
            return df

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
        type_tester = df[featurizer_type].iloc[0]

        if featurizer_type == self.composition_col:
            # Convert formulas to composition objects
            if isinstance(type_tester, str):
                self.logger.info(self._log_prefix +
                                 "Compositions detected as strings. Attempting "
                                 "conversion to Composition objects...")
                stc = StrToComposition(overwrite_data=True,
                                       target_col_id=featurizer_type)
                df = stc.featurize_dataframe(df, featurizer_type,
                                             multiindex=self.multiindex,
                                             ignore_errors=True,
                                             inplace=False)

            elif isinstance(type_tester, dict):
                self.logger.info(self._log_prefix +
                                 "Compositions detected as dicts. Attempting "
                                 "conversion to Composition objects...")
                df[featurizer_type] = [Composition.from_dict(d) for d in
                                       df[featurizer_type]]

            # Convert non-oxidstate containing comps to oxidstate comps
            if self.guess_oxistates:
                self.logger.info(self._log_prefix +
                                 "Guessing oxidation states of compositions, as"
                                 " they were not present in input.")
                cto = CompositionToOxidComposition(
                    target_col_id=featurizer_type, overwrite_data=True,
                    return_original_on_error=True, max_sites=-50)
                try:
                    df = cto.featurize_dataframe(df, featurizer_type,
                                                 multiindex=self.multiindex,
                                                 inplace=False)
                except Exception as e:
                    self.logger.info(self._log_prefix +
                                     "Could not decorate oxidation states due "
                                     "to {}. Excluding featurizers based on "
                                     "composition oxistates".format(e))
                    classes_require_oxi = [c.__class__.__name__ for c in
                                           CompositionFeaturizers().need_oxi]
                    self.exclude.extend(classes_require_oxi)

        else:
            # Convert structure/bs/dos dicts to objects (robust already)
            if isinstance(type_tester, (dict, str)):
                self.logger.info(self._log_prefix.capitalize() +
                                 "{} detected as string or dict. Attempting "
                                 "conversion to {} objects..."
                                 "".format(featurizer_type, featurizer_type))
                if isinstance(type_tester, str):
                    raise ValueError(
                        "{} column is type {}. Cannot convert."
                        "".format(featurizer_type, type(type_tester)))
                dto = DictToObject(overwrite_data=True,
                                   target_col_id=featurizer_type)
                df = dto.featurize_dataframe(df, featurizer_type, inplace=False)

                # Decorate with oxidstates
                if featurizer_type == self.structure_col and \
                        self.guess_oxistates:
                    self.logger.info(
                        self._log_prefix +
                        "Guessing oxidation states of structures if they were "
                        "not present in input.")
                    sto = StructureToOxidStructure(
                        target_col_id=featurizer_type, overwrite_data=True,
                        return_original_on_error=True, max_sites=-50)
                    try:
                        df = sto.featurize_dataframe(df, featurizer_type,
                                                     multiindex=self.multiindex,
                                                     inplace=False)
                    except Exception as e:
                        self.logger.info(
                            self._log_prefix +
                            "Could not decorate oxidation states on structures "
                            "due to {}.".format(e))
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
                self.logger.info(self._log_prefix +
                                 "composition column already exists, "
                                 "overwriting with composition from structure.")
            else:
                self.logger.info(self._log_prefix +
                                 "Adding compositions from structures.")

            df = self._tidy_column(df, self.structure_col)

            # above tidy column will add oxidation states, these oxidation
            # states will then be transferred to composition.
            struct2comp = StructureToComposition(
                target_col_id=self.composition_col, overwrite_data=overwrite)
            df = struct2comp.featurize_dataframe(df, self.structure_col)
        return df
