from sklearn.exceptions import NotFittedError
from pymatgen import Composition
from matminer.featurizers.conversions import CompositionToOxidComposition, StructureToOxidStructure, StrToComposition, DictToObject, StructureToComposition

from matbench.utils.utils import MatbenchError, setup_custom_logger
from matbench.base import DataframeTransformer, LoggableMixin
from matbench.featurization.sets import CompositionFeaturizers, \
    StructureFeaturizers, BSFeaturizers, DOSFeaturizers


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
        ignore_cols ([str]): Column names to be ignored/removed from any
            dataframe undergoing fitting or transformation.
        ignore_errors (bool): If True, each featurizer will ignore all errors
            during featurization.
        drop_inputs (bool): Drop the columns containing input objects for
            featurization (e.g., drop composition column folllowing featurization).
        exclude ([str]): Class names of featurizers to exclude.
        guess_oxistates (bool): If True, try to decorate sites with oxidation state.
        multiiindex (bool): If True, returns a multiindexed dataframe.
        n_jobs (int): The number of parallel jobs to use during featurization
            for each featurizer. -1 sets n_jobs = n_cores
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default matbench logger will be
            used. If set to False, then no logging will occur.

    Attributes:
        featurizers (dict): Same format as input dictionary in Args. Values
            contain the actual objects being used for featurization.

    """

    def __init__(self, featurizers=None, ignore_cols=None, ignore_errors=True,
                 drop_inputs=True, exclude=None, guess_oxistates=True,
                 multiindex=False, n_jobs=None, logger=True):

        self._logger = self.get_logger(logger)
        self.ignore_cols = ignore_cols or []
        self.is_fit = False
        self.ignore_errors = ignore_errors
        self.drop_inputs = drop_inputs
        self.multiindex = multiindex
        self.n_jobs = n_jobs
        self.guess_oxistates = guess_oxistates
        self.features = []

        # Set featurizers
        if not featurizers:
            cfset = CompositionFeaturizers(exclude=exclude).best
            sfset = StructureFeaturizers(exclude=exclude).best
            bsfset = BSFeaturizers(exclude=exclude).best
            dosfset = DOSFeaturizers(exclude=exclude).best
            self.featurizers = {"composition": cfset,
                                "structure": sfset,
                                "bandstructure": bsfset,
                                "dos": dosfset}
        else:
            if not isinstance(featurizers, dict):
                raise TypeError("Featurizers must be a dictionary with keys"
                                "of 'composition', 'structure', 'bandstructure', "
                                "and 'dos' and values of corresponding lists of "
                                "featurizers.")
            else:
                for ftype in featurizers:
                    # Normalize the names from the aliases
                    if ftype in _composition_aliases:
                        featurizers["composition"] = featurizers.pop(ftype)
                    elif ftype in _structure_aliases:
                        featurizers["structure"] = featurizers.pop(ftype)
                    elif ftype in _bandstructure_aliases:
                        featurizers["bandstructure"] = featurizers.pop(ftype)
                    elif ftype in _dos_aliases:
                        featurizers["dos"] = featurizers.pop(ftype)
                    else:
                        raise ValueError(
                            "The featurizers dict key {} is not a valid "
                            "featurizer type. Please choose from {}".format(
                                ftype, _aliases))
                # Assign empty featurizer list to featurizers not specified by type
                for ftype in ["composition", "structure", "bandstructure", "dos"]:
                    if ftype not in featurizers:
                        featurizers[ftype] = []
                self.featurizers = featurizers

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
        df = self._prescreen_df(df, inplace=True, col_id=target)
        df = self._add_composition_from_structure(df)
        for featurizer_type, featurizers in self.featurizers.items():
            if not featurizers:
                self.logger.info("No {} featurizers being used.".format(featurizer_type))
            if featurizer_type in df.columns:
                df = self._tidy_column(df, featurizer_type)
                for f in featurizers:
                    f.fit(df[featurizer_type].tolist())
                    f.set_n_jobs(self.n_jobs)
                    self.features += f.feature_labels()
                    self.logger.info("Fit {} to {} samples in dataframe.".format(f.__class__.__name__, df.shape[0]))
        self.is_fit = True
        return self

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
        if not self.is_fit:
            # Featurization requires featurizers already be fit...
            raise NotFittedError("AutoFeaturizer has not been fit!")
        df = self._prescreen_df(df, inplace=True, col_id=target)
        df = self._add_composition_from_structure(df)

        for featurizer_type, featurizers in self.featurizers.items():
            if featurizer_type in df.columns:
                df = self._tidy_column(df, featurizer_type)

                for f in featurizers:
                    df = f.featurize_dataframe(df, featurizer_type, ignore_errors=self.ignore_errors, multiindex=self.multiindex)
                df = df.drop(columns=[featurizer_type])
            else:
                self.logger.info("Featurizer type {} not in the dataframe. Skiping...".format(featurizer_type))
        return df

    def _prescreen_df(self, df, inplace=True, col_id=None):
        """
        Pre-screen a dataframe.

        Args:
            df (pandas.DataFrame): The dataframe to be screened.
            inplace (bool): Manipulate the df in-place in memory
            col_id: The column id to ensure is present.

        Returns:
            df (pandas.DataFrame) The screened dataframe.

        """
        if not inplace:
            df = df.copy(deep=True)
        if col_id and col_id not in df:
            raise MatbenchError("'{}' column must be in data!".format(col_id))
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
        # todo: Make the following conversions more robust (no lazy [0] type checking)
        # todo: needs MAJOR logging!!!!!!!!!!!!!!!!!
        if featurizer_type == "composition":
            # Convert formulas to composition objects
            if isinstance(df[featurizer_type][0], str):
                stc = StrToComposition(overwrite_data=True,
                                       target_col_id=featurizer_type)
                df = stc.featurize_dataframe(df, featurizer_type,
                                             multiindex=self.multiindex)

            elif isinstance(df[featurizer_type][0], dict):
                df[featurizer_type] = [Composition.from_dict(d) for d in df[featurizer_type]]

            # Convert non-oxidstate containing comps to oxidstate comps
            if self.guess_oxistates:
                cto = CompositionToOxidComposition(
                    target_col_id=featurizer_type,
                    overwrite_data=True)
                df = cto.featurize_dataframe(df, featurizer_type,
                                             multiindex=self.multiindex)

        else:
            # Convert structure/bs/dos dicts to objects (robust already)
            dto = DictToObject(overwrite_data=True, target_col_id=featurizer_type)
            df = dto.featurize_dataframe(df, featurizer_type)

            # Decorate with oxidstates
            if featurizer_type == "structure" and self.guess_oxistates:
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
        if self.featurizers["composition"]:
            if "structure" in df.columns and "composition" not in df.columns:
                struct2comp = StructureToComposition(
                    target_col_id="composition", overwrite_data=False)
                df = struct2comp.featurize_dataframe(df, "structure")
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