import logging

from pymatgen import Composition, Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.dos import CompleteDos
from matminer.featurizers.conversions import (CompositionToOxidComposition,
                                              StructureToOxidStructure)

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

    def __init__(self, ignore_cols=None, ignore_errors=True,
                 drop_inputs=True, exclude=None, multiindex=False,
                 n_jobs=None, featurizers=None, logger=setup_custom_logger()):

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
                self.featurizers = featurizers

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

    def _pre_screen_col(self, col_id, prefix='Input Data', multiindex=None):
        multiindex = multiindex or self.multiindex
        if multiindex and isinstance(col_id, str):
            return (prefix, col_id)
        else:
            return col_id

    # todo: Remove once MultipleFeaturizer is fixed
    def _featurize_sequentially(self, df, fset, col_id, **kwargs):
        for idx, f in enumerate(fset):
            if idx > 0:
                col_id = self._pre_screen_col(col_id)
            f.set_n_jobs(n_jobs=self.n_jobs)
            df = f.fit_featurize_dataframe(df, col_id, **kwargs)
        return df

    def fit(self, df, target):



    def auto_featurize(self, df, input_cols=("formula", "structure"),
                       **kwargs):
        """
        Featurizes the dataframe based on input_columns.

        **Note: use only if you want call featurize_* methods with the default
        arguments or when **kwargs are shared between these methods.

        Args:
            df (pandas.DataFrame):
            input_cols ([str]): columns used for featurization (e.g. "structure"),
                set to None to try all preset columns.
            kwargs: other keywords arguments related to featurize_* methods

        Returns (pandas.DataFrame):
            self.df w/ new features added via featurizering input_cols
        """
        df_cols_init = df.columns.values
        df = self._prescreen_df(df)
        for idx, column in enumerate(input_cols):
            featurizer = getattr(self, "featurize_{}".format(column), None)
            if column in df_cols_init:
                if featurizer is not None:
                    if idx > 0:
                        col_id = self._pre_screen_col(column)
                    else:
                        col_id = column
                    df = featurizer(df, col_id=col_id, **kwargs)
                else:
                    self.logger.warning(
                        'No method available to featurize "{}"'.format(column))
            else:
                self.logger.warning(
                    "{} not found in the dataframe! Skipping...".format(column))
        return df

    def featurize_formula(self, df, featurizers="best", col_id="formula",
                          compcol="composition", guess_oxidstates=False,
                          inplace=True, asindex=True):
        """
        Featurizes based on formula or composition (pymatgen Composition).

        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.Featurizer] or str): Either a list of
                featurizers or a set from CompositionFeaturizers. For example,
                "best" will use the CompositionFeaturizers.best featurizers.
                Default is None, which uses only the best featurizers.
            col_id (str): actual column name to be used as composition
            compcol (str): default or final column name for composition
            guess_oxidstates (bool): whether to guess elements oxidation states
                which is required for some featurizers such as CationProperty,
                OxidationStates, ElectronAffinity and ElectronegativityDiff.
                Set to False if oxidation states already available in composition.
            asindex (bool): whether to set formula col_id as df index
            kwargs: keyword args that are accepted by AllFeaturizers.composition
                may be accepted by other featurize_* methods

        Returns (pandas.DataFrame):
            Dataframe with compositional features added.
        """
        df = self._prescreen_df(df=df, inplace=inplace)
        if compcol not in df:
            df[compcol] = df[col_id].apply(Composition)
        if guess_oxidstates:
            cto = CompositionToOxidComposition(target_col_id=compcol,
                                               overwrite_data=True)
            df = cto.featurize_dataframe(df, compcol,
                                         multiindex=self.multiindex)

        if isinstance(featurizers, str):
            featurizers = getattr(self.cfset, featurizers)

        # Multiple featurizer has issues, just use this bc we get pbar!
        df = self._featurize_sequentially(df, featurizers, compcol,
                                          ignore_errors=self.ignore_errors,
                                          multiindex=self.multiindex)
        if asindex:
            df = df.set_index(self._pre_screen_col(col_id))
        if self.drop_inputs:
            return df.drop([self._pre_screen_col(compcol)], axis=1)
        else:
            return df

    def featurize_structure(self, df, featurizers="best",
                            col_id="structure",
                            inplace=True, guess_oxidstates=True):
        """
        Featurizes based on crystal structure (pymatgen Structure object)

        Args:
            df (pandas.DataFrame):
            col_id (str): column name containing pymatgen Structure
        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.Featurizer] or str): Either a list of
                featurizers or a set from StructureFeaturizers. For example,
                "best" will use the StructureFeaturizers.best featurizers.
                Default is None, which uses only the best featurizers.
            col_id (str): actual column name to be used as structure
            inplace (bool): whether to modify the input df
            guess_oxidstates (bool): whether to guess elements oxidation states
                in the structure which is required for some featurizers such as
                EwaldEnergy, ElectronicRadialDistributionFunction. Set to
                False if oxidation states already available in the structure.
            kwargs: keyword args that are accepted by AllFeaturizers.structure
                may be accepted by other featurize_* methods

        Returns (pandas.DataFrame):
            Dataframe with structure features added.
        """
        df = self._prescreen_df(df=df, inplace=inplace, col_id=col_id)
        if isinstance(df[col_id][0], dict):
            df[col_id] = df[col_id].apply(Structure.from_dict)
        if guess_oxidstates:
            sto = StructureToOxidStructure(target_col_id=col_id,
                                           overwrite_data=True)
            df = sto.featurize_dataframe(df, col_id,
                                         multiindex=self.multiindex)
        if isinstance(featurizers, str):
            featurizers = getattr(self.sfset, featurizers)

        # Multiple featurizer has issues, just use this bc we get pbar!
        df = self._featurize_sequentially(df, featurizers, col_id,
                                          ignore_errors=self.ignore_errors,
                                          multiindex=self.multiindex)

        if self.drop_inputs:
            return df.drop([self._pre_screen_col(col_id)], axis=1)
        else:
            return df

    def featurize_dos(self, df, featurizers="best", col_id="dos",
                      inplace=True):
        """
        Featurizes based on density of state (pymatgen CompleteDos object)

        Args:
            df (pandas.DataFrame):
            col_id (str): column name containing pymatgen Dos (or CompleteDos)
        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.Featurizer] or str): Either a list of
                featurizers or a set from DOSFeaturizers. For example,
                "best" will use the DOSFeaturizers.best featurizers.
                Default is None, which uses only the best featurizers.
            col_id (str): actual column name to be used as dos
            inplace (bool): whether to modify the input df
            kwargs: keyword arguments that may be accepted by other featurize_*
                methods passed through auto_featurize

        Returns (pandas.DataFrame):
            Dataframe with dos features added.
        """
        df = self._prescreen_df(df=df, inplace=inplace, col_id=col_id)
        if isinstance(df[col_id][0], dict):
            df[col_id] = df[col_id].apply(CompleteDos.from_dict)
        if isinstance(featurizers, str):
            featurizers = getattr(self.dosfset, featurizers)

        # Multiple featurizer has issues, just use this bc we get pbar!
        df = self._featurize_sequentially(df, featurizers, col_id,
                                          ignore_errors=self.ignore_errors,
                                          multiindex=self.multiindex)
        if self.drop_inputs:
            return df.drop([self._pre_screen_col(col_id)], axis=1)
        else:
            return df

    def featurize_bandstructure(self, df, featurizers="all",
                                col_id="bandstructure", inplace=True):
        """
        Featurizes based on density of state (pymatgen BandStructure object)

        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.Featurizer] or str): Either a list of
                featurizers or a set from BSFeaturizers. For example,
                "best" will use the BSFeaturizers.best featurizers.
                Default is None, which uses only the best featurizers.
            col_id (str): actual column name containing the bandstructure data
            inplace (bool): whether to modify the input df
            kwargs: keyword arguments that may be accepted by other featurize_*
                methods passed through auto_featurize

        Returns (pandas.DataFrame):
            Dataframe with bandstructure features added.
        """
        df = self._prescreen_df(df=df, inplace=inplace, col_id=col_id)
        if isinstance(df[col_id][0], dict):
            df[col_id] = df[col_id].apply(BandStructure.from_dict)
        if isinstance(featurizers, str):
            featurizers = getattr(self.bsfset, featurizers)
        # Multiple featurizer has issues, just use this bc we get pbar!
        df = self._featurize_sequentially(df, featurizers, col_id,
                                          ignore_errors=self.ignore_errors,
                                          multiindex=self.multiindex)
        if self.drop_inputs:
            return df.drop([self._pre_screen_col(col_id)], axis=1)
        else:
            return df
