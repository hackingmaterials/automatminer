import logging

from matminer.featurizers.conversions import (CompositionToOxidComposition,
                                              StructureToOxidStructure)
from pymatgen import Composition, Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.dos import CompleteDos

from matbench.utils.utils import MatbenchError, setup_custom_logger
from matbench.featurization.sets import CompositionFeaturizers, \
    StructureFeaturizers, BSFeaturizers, DOSFeaturizers


class Featurization(object):
    """
    Takes in a dataframe and generate features from preset columns such as
    "formula", "structure", "bandstructure", "dos", etc. One may use
    the auto_featurize method to featurize via all available featurizers
    with default setting or selectively call featurizer methods.
    Usage examples:
        featurizer = Featurize()
            df = featurizer.auto_featurize(df) # all features of all types
        or:
            df = featurizer.featurize_formula(df) # all formula-related feature
        or:
            df = featurizer.featurize_dos(df, featurizers=[Hybridization()])

    Args:
        ignore_cols ([str]): if set, these columns are excluded
        preset_name (str): some featurizers (w/ from_preset) take in this arg
        ignore_errors (bool): whether to ignore exceptions raised when
            featurize_dataframe is called
        drop_featurized_col (bool): whether to drop the featurized column after
            the corresponding featurize_* method is called
        exclude ([str]): list of the featurizer names to be excluded. Note
            that these names are str (e.g. "ElementProperty") and not the class
        multiindex (bool): a matminer featurizer argument that transforms
            feature label to tuples that makes it easier to track them back to
            which featurizer they come from.
            * Note: if you set to True and your target is "gap", you need to
            pass target = ("Input Data", "gap") in classes such as PreProcess.
        n_jobs (int): number of CPUs/workers used in featurization. Default
            behavior is matminer's default behavior.
    """

    def __init__(self, ignore_cols=None, ignore_errors=True,
                 drop_featurized_col=True, exclude=None, multiindex=False,
                 n_jobs=None, logger=None):

        if logger is None:
            # Log to the current directory
            self.logger = setup_custom_logger(filepath='.', level=logging.INFO)
        else:
            # Use the passed logger
            self.logger = logger

        self.ignore_cols = ignore_cols or []
        self.cfset = CompositionFeaturizers(exclude=exclude)
        self.sfset = StructureFeaturizers(exclude=exclude)
        self.bsfset = BSFeaturizers(exclude=exclude)
        self.dosfset = DOSFeaturizers(exclude=exclude)
        self.ignore_errors = ignore_errors
        self.drop_featurized_col = drop_featurized_col
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
        if self.drop_featurized_col:
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

        if self.drop_featurized_col:
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
        if self.drop_featurized_col:
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
        if self.drop_featurized_col:
            return df.drop([self._pre_screen_col(col_id)], axis=1)
        else:
            return df
