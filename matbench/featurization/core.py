import logging

from sklearn.exceptions import NotFittedError
from pymatgen import Composition, Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.dos import CompleteDos
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
    """

    def __init__(self, featurizers=None, ignore_cols=None, ignore_errors=True,
                 drop_inputs=True, exclude=None, multiindex=False,
                 n_jobs=None, logger=setup_custom_logger()):

        self.logger = logger
        self.ignore_cols = ignore_cols or []

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
                            "The featurizers dict key {} is not a valid featurizer type. Please choose from {}".format(
                                ftype, _aliases))
                # Assign empty featurizer list to featurizers not specified by type
                for ftype in ["composition", "structure", "bandstructure", "dos"]:
                    if ftype not in featurizers:
                        featurizers[ftype] = []

                self.featurizers = featurizers

        self.is_fit = False
        self.ignore_errors = ignore_errors
        self.drop_inputs = drop_inputs
        self.multiindex = multiindex
        self.n_jobs = n_jobs

    def _prescreen_df(self, df, inplace=True, col_id=None):
        if not inplace:
            df = df.copy(deep=True)
        if col_id and col_id not in df:
            raise MatbenchError("'{}' column must be in data!".format(col_id))
        if self.ignore_errors is not None:
            for col in self.ignore_cols:
                if col in df:
                    df = df.drop([col], axis=1)
        return df

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
        for featurizer_type, featurizers in self.featurizers.items():
            if not featurizers:
                self.logger._log("info", "No {} featurizers being used.".format(featurizer_type))
            for f in featurizers:
                f.fit(df[featurizer_type].tolist())
                f.set_n_jobs(self.n_jobs)
                self.logger._log("info", "Fit {} to {} samples in dataframe.".format(f.__class__.__name__, df.shape[0]))
        self.is_fit = True
        return self

    def transform(self, df, target, guess_oxidstates=True):
        """
        Decorate a dataframe containing composition, structure, bandstructure,
        and/or DOS objects with descriptors.

        Args:
            df (pandas.DataFrame): The dataframe not containing features.
            target (str): The ML-target property contained in the df.
            guess_oxidstates (bool): Whether to guess oxidation states for
                Composition and Structure objects if not already present.

        Returns:
            df (pandas.DataFrame): Transformed dataframe containing features.
        """
        if not self.is_fit:
            # Featurization requires featurizers already be fit...
            raise NotFittedError("AutoFeaturizer has not been fit!")
        df = self._prescreen_df(df, inplace=True, col_id=target)

        # Add compositions from structures if not already present and composition features are desired
        if "composition" in self.featurizers.keys():
            if "structure" in df.columns and "composition" not in df.columns:
                struct2comp = StructureToComposition(target_col_id="composition", overwrite_data=False)
                df = struct2comp.featurize_dataframe(df, "structure")

        for featurizer_type, featurizers in self.featurizers.items():
            if featurizer_type in df.columns:
                #todo: Make the following conversions more robust (no lazy [0] type checking)
                #todo: needs logging
                if featurizer_type == "composition":
                    # Convert formulas to composition objects
                    if isinstance(df[featurizer_type][0], str):
                        stc = StrToComposition(overwrite_data=True,
                                               target_col_id=featurizer_type)
                        df = stc.featurize_dataframe(df, featurizer_type,
                                                     multiindex=self.multiindex)

                    # Convert non-oxidstate containing comps to oxidstate comps
                    if guess_oxidstates:
                        cto = CompositionToOxidComposition(target_col_id=featurizer_type,
                                                           overwrite_data=True)
                        df = cto.featurize_dataframe(df, featurizer_type,
                                                     multiindex=self.multiindex)

                else:
                    # Convert structure/bs/dos dicts to objects (robust already)
                    dto = DictToObject(overwrite_data=True)
                    df = dto.featurize_dataframe(df, featurizer_type)

                    # Decorate with oxidstates
                    if featurizer_type == "structure" and guess_oxidstates:
                        sto = StructureToOxidStructure(target_col_id=featurizer_type,
                                                       overwrite_data=True)
                        df = sto.featurize_dataframe(df, featurizer_type,
                                                     multiindex=self.multiindex)

                for f in featurizers:
                    df = f.featurize_dataframe(df, featurizer_type, ignore_errors=self.ignore_errors, multiindex=self.multiindex)
                df = df.drop(labels=[featurizer_type])
            else:
                self._log("info", "Featurizer type {} not in the dataframe. Skiping...".format(featurizer_type))
        return df

