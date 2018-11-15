from sklearn.exceptions import NotFittedError
from pymatgen import Composition
from matminer.featurizers.conversions import StructureToOxidStructure, \
    StrToComposition, DictToObject, StructureToComposition

from mslearn.utils.utils import MatbenchError, check_fitted, set_fitted
from mslearn.base import DataframeTransformer, LoggableMixin
from mslearn.featurization.sets import CompositionFeaturizers, \
    StructureFeaturizers, BSFeaturizers, DOSFeaturizers
from mslearn.featurization.metaselection.core import FeaturizerMetaSelector

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


# todo: use the matminer version once its fixed and pushed
from matminer.featurizers.conversions import ConversionFeaturizer
class CompositionToOxidComposition(ConversionFeaturizer):
    """Utility featurizer to add oxidation states to a pymatgen Composition.

    Oxidation states are determined using pymatgen's guessing routines.
    The expected input is a `pymatgen.core.composition.Composition` object.

    Note that this Featurizer does not produce machine learning-ready features
    but instead can be applied to pre-process data or as part of a Pipeline.

    Args:
        **kwargs: Parameters to control the settings for
            `pymatgen.io.structure.Structure.add_oxidation_state_by_guess()`.
        target_col_id (str or None): The column in which the converted data will
            be written. If the column already exists then an error will be
            thrown unless `overwrite_data` is set to `True`. If `target_col_id`
            begins with an underscore the data will be written to the column:
            `"{}_{}".format(col_id, target_col_id[1:])`, where `col_id` is the
            column being featurized. If `target_col_id` is set to None then
            the data will be written "in place" to the `col_id` column (this
            will only work if `overwrite_data=True`).
        overwrite_data (bool): Overwrite any data in `target_column` if it
            exists.
        coerce_mixed (bool): If a composition has both species containing
            oxid states and not containing oxid states, strips all of the
            oxid states and guesses the entire composition's oxid states.

    """

    def __init__(self, target_col_id='composition_oxid', overwrite_data=False,
                 coerce_mixed=True, **kwargs):
        super().__init__(target_col_id, overwrite_data)
        self.oxi_guess_params = kwargs
        self.coerce_mixed = coerce_mixed

    def featurize(self, comp):
        """Add oxidation states to a Structure using pymatgen's guessing routines.

        Args:
            comp (`pymatgen.core.composition.Composition`): A composition.

        Returns:
            (`pymatgen.core.composition.Composition`): A Composition object
                decorated with oxidation states.
        """
        els_have_oxi_states = [hasattr(s, "oxi_state") for s in comp.elements]
        if all(els_have_oxi_states):
            return [comp]
        elif any(els_have_oxi_states):
            if self.coerce_mixed:
                comp = comp.element_composition
            else:
                raise ValueError("Composition {} has a mix of species with "
                                 "and without oxidation states. Please enable "
                                 "coercion to all oxidation states with "
                                 "coerce_mixed.".format(comp))
        return [comp.add_charges_from_oxi_state_guesses(
            **self.oxi_guess_params)]

    def citations(self):
        return [(
            "@article{ward_agrawal_choudary_wolverton_2016, title={A "
            "general-purpose machine learning framework for predicting "
            "properties of inorganic materials}, volume={2}, "
            "DOI={10.1038/npjcompumats.2017.28}, number={1}, journal={npj "
            "Computational Materials}, author={Ward, Logan and Agrawal, Ankit "
            "and Choudhary, Alok and Wolverton, Christopher}, year={2016}}")]

    def implementors(self):
        return ["Anubhav Jain", "Alex Ganose", "Alex Dunn"]


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
            for each featurizer. -1 sets n_jobs = n_cores
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default mslearn logger will be
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

    def __init__(self, featurizers=None, exclude=None, use_metaselector=True,
                 max_na_frac=0.05, ignore_cols=None, ignore_errors=True,
                 drop_inputs=True, guess_oxistates=True, multiindex=False,
                 n_jobs=None, logger=True):

        self._logger = self.get_logger(logger)
        self.featurizers = featurizers
        self.exclude = exclude
        self.use_metaselector = use_metaselector
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
        df = self._prescreen_df(df, inplace=True)
        df = self._add_composition_from_structure(df)

        for featurizer_type, featurizers in self.featurizers.items():
            if featurizer_type in df.columns:
                df = self._tidy_column(df, featurizer_type)

                for f in featurizers:
                    df = f.featurize_dataframe(df, featurizer_type,
                                               ignore_errors=self.ignore_errors,
                                               multiindex=self.multiindex)
                df = df.drop(columns=[featurizer_type])
            else:
                self.logger.info("Featurizer type {} not in the dataframe. "
                                 "Skipping...".format(featurizer_type))
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
                self.metaselector = FeaturizerMetaSelector(self.max_na_percent)
                auto_exclude = self.metaselector.auto_excludes(df)
                if auto_exclude:
                    self.logger.info("Based on metafeatures of the dataset, "
                                     "these featurizers are excluded for "
                                     "returning nans more than the "
                                     "max_na_percent of {}: {}".
                                     format(self.max_na_percent, auto_exclude))
                    if self.exclude:
                        auto_exclude.extend(self.exclude)
                    self.exclude = auto_exclude

            for featurizer_type in _supported_featurizer_types.keys():
                if featurizer_type in df.columns:
                    featurizer_set = _supported_featurizer_types[featurizer_type]
                    self.featurizers[featurizer_type] = \
                        featurizer_set(exclude=self.exclude).best
                else:
                    self.logger.info("Featurizer type {} not in the dataframe"
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
                        self.featurizers["composition"] = self.featurizers.pop(ftype)
                    elif ftype in _structure_aliases:
                        self.featurizers["structure"] = self.featurizers.pop(ftype)
                    elif ftype in _bandstructure_aliases:
                        self.featurizers["bandstructure"] = self.featurizers.pop(ftype)
                    elif ftype in _dos_aliases:
                        self.featurizers["dos"] = self.featurizers.pop(ftype)
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
        if featurizer_type == "composition":
            # Convert formulas to composition objects
            if isinstance(df[featurizer_type].iloc[0], str):
                self.logger.info("Compositions detected as strings. Attempting "
                                 "conversion to Composition objects...")
                stc = StrToComposition(overwrite_data=True,
                                       target_col_id=featurizer_type)
                df = stc.featurize_dataframe(df, featurizer_type,
                                             multiindex=self.multiindex)

            elif isinstance(df[featurizer_type].iloc[0], dict):
                self.logger.info("Compositions detected as dicts. Attempting "
                                 "conversion to Composition objects...")
                df[featurizer_type] = [Composition.from_dict(d) for d in df[featurizer_type]]

            # Convert non-oxidstate containing comps to oxidstate comps
            if self.guess_oxistates:
                self.logger.info("Guessing oxidation states of compositions, as"
                                 " they were not present in input.")
                cto = CompositionToOxidComposition(
                    target_col_id=featurizer_type,
                    overwrite_data=True)
                df = cto.featurize_dataframe(df, featurizer_type,
                                             multiindex=self.multiindex)

        else:
            # Convert structure/bs/dos dicts to objects (robust already)
            self.logger.info("{} detected as strings. Attempting "
                             "conversion to Composition objects..."
                             "".format(featurizer_type))
            dto = DictToObject(overwrite_data=True, target_col_id=featurizer_type)
            df = dto.featurize_dataframe(df, featurizer_type)

            # Decorate with oxidstates
            if featurizer_type == "structure" and self.guess_oxistates:
                self.logger.info("Guessing oxidation states of structures, as "
                                 "they were not present in input.")
                sto = StructureToOxidStructure(target_col_id=featurizer_type,
                                               overwrite_data=True)
                df = sto.featurize_dataframe(df, featurizer_type,
                                             multiindex=self.multiindex)
        return df

    def _add_composition_from_structure(self, df):
        """
        Automatically deduce compositions from structures if:
            1. structures are available
            2. composition features are actually desired. (deduced from whether
                composition featurizers are present in self.featurizers).
        Args:
            df (pandas.DataFrame): May or may not contain composition column.

        Returns:
            df (pandas.DataFrame): Contains composition column if desired
        """
        if "structure" in df.columns and "composition" not in df.columns:
            if self.auto_featurizer or (set(_composition_aliases)
                                        & set(self.featurizers.keys())):
                df = self._tidy_column(df, "structure")
                struct2comp = StructureToComposition(
                    target_col_id="composition", overwrite_data=False)
                df = struct2comp.featurize_dataframe(df, "structure")
                self.logger.debug("Adding compositions from structures.")
        return df


if __name__ == "__main__":
    from matminer.datasets.dataset_retrieval import load_dataset
    df = load_dataset("flla")
    df = df.iloc[:100]
    df = df[["structure",  "e_above_hull"]]
    print(df)
    af = AutoFeaturizer()
    af.fit(df, "e_above_hull")
    df = af.transform(df, "e_above_hull")
    print(df.columns.tolist())