from warnings import warn

import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
import matminer.featurizers.dos as dosf
import matminer.featurizers.bandstructure as bf
from matminer.featurizers.base import MultipleFeaturizer
from matminer.utils.conversions import composition_to_oxidcomposition, \
    structure_to_oxidstructure
from matbench.utils.utils import MatbenchError
from pymatgen import Composition, Structure
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.dos import CompleteDos


class FeaturizerSet:
    """
    An abstract class for defining sets of featurizers and the methods they
    must implement.

    Each set returned is a list of matminer featurizer objects.

    Args:
        exclude ([str]): The class names of the featurizers which should be
            excluded.
    """

    def __init__(self, exclude=None):
        self.exclude = [] if exclude is None else exclude

    def best(self):
        """
        A set of featurizers that generally gives informative features without
        excessive featurization time.
        """
        raise NotImplementedError("This featurizer set must return a set of "
                                  "best featurizers")

    def all(self):
        """
        All featurizers available in matminer for this featurization type.
        """
        raise NotImplementedError("This featurizer set must return a set of "
                                  "all featurizers")


class CompositionFeaturizers(FeaturizerSet):
    """
    Lists of composition featurizers, depending on requirements.

    Args:
        exclude ([str]): The class names of the featurizers which should be
            excluded.
    """

    @property
    def fast(self):
        """
        Generally fast featurizers.
        """
        featzers = [cf.AtomicOrbitals(),
                    cf.ElementProperty.from_preset("magpie"),
                    cf.ElementProperty.from_preset("matminer"),
                    cf.ElementFraction(),
                    cf.Stoichiometry(),
                    cf.TMetalFraction()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def need_oxi(self):
        """
        Fast if compositions are already decorated with oxidation states, slow
        otherwise.
        """
        featzers = [cf.CationProperty.from_preset(preset_name='deml'),
                    cf.OxidationStates.from_preset(preset_name='deml'),
                    cf.ElectronAffinity(),
                    cf.ElectronegativityDiff(),
                    cf.YangSolidSolution(),
                    cf.IonProperty()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def slow(self):
        """
        Generally slow featurizers under most conditions.
        """
        featzers = [cf.Miedema(),
                    # much slower than the rest
                    cf.AtomicPackingEfficiency(),
                    # requires mpid present
                    cf.CohesiveEnergy()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def all(self):
        """
        All composition featurizers.
        """
        return self.fast + self.need_oxi + self.slow

    @property
    def best(self):
        return self.fast


class StructureFeaturizers(FeaturizerSet):
    """
    Lists of structure featurizers, depending on requirements.
    """

    @property
    def matrix(self):
        """
        Structure featurizers returning matrices in each column. Not useful
        for vectorized representations of crystal structures.
        """
        featzers = [sf.RadialDistributionFunction(),  # returns dict
                    sf.CoulombMatrix(),  # returns a matrix
                    sf.SineCoulombMatrix(),  # returns a matrix
                    sf.OrbitalFieldMatrix(),  # returns a matrix
                    sf.MinimumRelativeDistances()]  # returns a list
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def fast(self):
        """
        Structure featurizers which are generally fast.
        """
        featzers = [sf.DensityFeatures(),
                    sf.GlobalSymmetryFeatures(),
                    sf.EwaldEnergy()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def many_features(self):
        featzers = [sf.BagofBonds(),
                    sf.PartialRadialDistributionFunction(),
                    sf.BondFractions]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def need_fit(self):
        """
        Structure featurizers which must be .fit before featurizing.
        Alternatively, use .fit_featurize_dataframe.
        """
        featzers = [sf.PartialRadialDistributionFunction(),
                    sf.BondFractions(),
                    sf.BagofBonds()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def slow(self):
        """
        Structure featurizers which are generally slow.
        """
        featzers = [
            sf.SiteStatsFingerprint.from_preset('CrystalNNFingerprint_ops'),
            sf.ChemicalOrdering(),
            sf.StructuralHeterogeneity(),
            sf.MaximumPackingEfficiency(),
            sf.XRDPowderPattern()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def all(self):
        return self.fast + self.slow + self.need_fit

    @property
    def best(self):
        featzers = self.fast + [sf.BondFractions()] + self.slow
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]


class DOSFeaturizers(FeaturizerSet):
    """
    Lists of DOS featurizers, depending on requirements
    """

    @property
    def all(self):
        featzers = [dosf.DOSFeaturizer(),
                    dosf.DopingFermi(),
                    dosf.Hybridization()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def best(self):
        return self.all


class BSFeaturizers(FeaturizerSet):
    """
    Lists of bandstructure featurizers, depending on requirements.
    """

    @property
    def all(self):
        featzers = [bf.BandFeaturizer(), bf.BranchPointEnergy()]
        return [i for i in featzers if i.__class__.__name__ not in self.exclude]

    @property
    def best(self):
        return self.all


class Featurize(object):
    """
    Takes in a dataframe and generate features from preset columns such as
    "formula", "structure", "bandstructure", "dos", etc. One may use
    the featurize_columns method to featurize via all available featurizers
    with default setting or selectively call featurizer methods.
    Usage examples:
        featurizer = Featurize()
            df = featurizer.featurize_columns(df) # all features of all types
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
                 n_jobs=None):

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

    def featurize_columns(self, df, input_cols=None, **kwargs):
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
        df = self._prescreen_df(df)
        input_cols = input_cols or ["formula", "structure"]
        for idx, column in enumerate(input_cols):
            featurizer = getattr(self, "featurize_{}".format(column), None)
            if featurizer is not None:
                if idx > 0:
                    col_id = self._pre_screen_col(column)
                else:
                    col_id = column
                df = featurizer(df, col_id=col_id, **kwargs)
            elif column not in df:
                raise MatbenchError('no "{}" in the data!')
            else:
                warn('No method available to featurize "{}"'.format(column))
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
            df[compcol] = composition_to_oxidcomposition(df[compcol])

        if isinstance(featurizers, str):
            featurizers = getattr(self.cfset, featurizers)

        featzer = MultipleFeaturizer(featurizers)
        if self.n_jobs:
            featzer.set_n_jobs(n_jobs=self.n_jobs)
        df = featzer.fit_featurize_dataframe(df, compcol,
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
            structure_to_oxidstructure(df[col_id], inplace=True)
        if isinstance(featurizers, str):
            featurizers = getattr(self.sfset, featurizers)
        featzer = MultipleFeaturizer(featurizers)
        if self.n_jobs:
            featzer.set_n_jobs(n_jobs=self.n_jobs)
        df = featzer.fit_featurize_dataframe(df, col_id,
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
                methods passed through featurize_columns

        Returns (pandas.DataFrame):
            Dataframe with dos features added.
        """
        df = self._prescreen_df(df=df, inplace=inplace, col_id=col_id)
        if isinstance(df[col_id][0], dict):
            df[col_id] = df[col_id].apply(CompleteDos.from_dict)
        if isinstance(featurizers, str):
            featurizers = getattr(self.dosfset, featurizers)
        featzer = MultipleFeaturizer(featurizers)
        if self.n_jobs:
            featzer.set_n_jobs(n_jobs=self.n_jobs)
        df = featzer.fit_featurize_dataframe(df, col_id,
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
                methods passed through featurize_columns

        Returns (pandas.DataFrame):
            Dataframe with bandstructure features added.
        """
        df = self._prescreen_df(df=df, inplace=inplace, col_id=col_id)
        if isinstance(df[col_id][0], dict):
            df[col_id] = df[col_id].apply(BandStructure.from_dict)
        if isinstance(featurizers, str):
            featurizers = getattr(self.bsfset, featurizers)
        featzer = MultipleFeaturizer(featurizers)
        if self.n_jobs:
            featzer.set_n_jobs(n_jobs=self.n_jobs)
        df = featzer.fit_featurize_dataframe(df, col_id,
                                             ignore_errors=self.ignore_errors,
                                             multiindex=self.multiindex)
        if self.drop_featurized_col:
            return df.drop([self._pre_screen_col(col_id)], axis=1)
        else:
            return df


