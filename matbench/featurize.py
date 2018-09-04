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

    def __init__(self, ignore_cols=None, preset_name="matminer",
                 ignore_errors=True, drop_featurized_col=True, exclude=None,
                 multiindex=False, n_jobs=None):
        self.ignore_cols = ignore_cols or []
        self.all_featurizers = AllFeaturizers(preset_name=preset_name,
                                              exclude=exclude)
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

    def featurize_columns(self, df=None, input_cols=None, **kwargs):
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

    def featurize_formula(self, df=None, featurizers="all", col_id="formula",
                          compcol="composition", guess_oxidstates=False,
                          inplace=True, asindex=True, **kwargs):
        """
        Featurizes based on formula or composition (pymatgen Composition).

        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.featurizer] or "all"):
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
        if featurizers == 'all':
            featurizers = self.all_featurizers.composition(**kwargs)
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

    def featurize_structure(self, df=None, featurizers="all", col_id="structure",
                            inplace=True, guess_oxidstates=True, **kwargs):
        """
        Featurizes based on crystal structure (pymatgen Structure object)

        Args:
            df (pandas.DataFrame):
            col_id (str): column name containing pymatgen Structure
        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.featurizer] or "all"):
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
        if featurizers == "all":
            featurizers = self.all_featurizers.structure(**kwargs)
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

    def featurize_dos(self, df=None, featurizers="all", col_id="dos",
                      inplace=True, **kwargs):
        """
        Featurizes based on density of state (pymatgen CompleteDos object)

        Args:
            df (pandas.DataFrame):
            col_id (str): column name containing pymatgen Dos (or CompleteDos)
        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.featurizer] or "all"):
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
        if featurizers == "all":
            featurizers = self.all_featurizers.dos()
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

    def featurize_bandstructure(self, df=None, featurizers="all",
                                col_id="bandstructure", inplace=True, **kwargs):
        """
        Featurizes based on density of state (pymatgen BandStructure object)

        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.featurizer] or "all"):
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
        if featurizers == "all":
            featurizers = self.all_featurizers.bandstructure()
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


class AllFeaturizers(object):
    """
    Contains all available featurizers that take various types of inputs such
    as "composition" or "formula", "structure", "dos", "band structure", etc.
    The point of this class is to have one place where available featurizers
    with default inputs are updated synced with matminer.

    Args:
        preset_name (str): some featurizers take in this argument
        exclude ([str]): list of the featurizer names to be excluded. Note
            that these names are str (e.g. "ElementProperty") and not the class
    """
    def __init__(self, preset_name="matminer", exclude=None):
        self.preset_name = preset_name
        self.exclude = exclude or []

    def composition(self, preset_name=None, need_oxidstates=False,
                    include_slow=False):
        """
        All composition-based featurizers with default arguments.

        Args:
            preset_name (str): some featurizers take in this argument
            need_oxidstates (bool): whether to return those featurizers that
                require oxidation states decorated Composition
            include_slow (bool): Whether to include relatively slow featurizers.
                We have not found evidence that these featurizers improve the
                prediction scores of the current datasets.

        Returns ([matminer featurizer classes]):
        """
        preset_name = preset_name or self.preset_name
        featzers = [
            cf.ElementProperty.from_preset(preset_name=preset_name),
            cf.AtomicOrbitals(),
            cf.BandCenter(),
            cf.Stoichiometry(),
            cf.ValenceOrbital(),
            cf.TMetalFraction(),
            cf.ElementFraction(),
        ]
        if need_oxidstates:
            featzers += [
                cf.CationProperty.from_preset(preset_name='deml'),
                cf.OxidationStates.from_preset(preset_name='deml'),
                cf.ElectronAffinity(),
                cf.ElectronegativityDiff(),
                cf.YangSolidSolution(),
            ]
        if include_slow:
            featzers += [
                cf.IonProperty(),
                cf.Miedema(),
                cf.AtomicPackingEfficiency(),  # much slower than the rest
                cf.CohesiveEnergy(), # an entry must be found in materialsproject.org
            ]
        names = [c.__class__.__name__ for c in featzers]
        return [f for i,f in enumerate(featzers) if names[i] not in self.exclude]

    def structure(self, preset_name="CrystalNNFingerprint_ops", need_fit=True):
        """
        All structure-based featurizers with default arguments that don't
        require the fit method to be called first.

        Args:
            preset_name (str): some featurizers take in this argument
            need_fit (bool): whether to include structure featurizers that
                require the calling of fit method first

        Returns ([matminer featurizer classes]):
        """
        preset_name = preset_name or self.preset_name
        featzers = [
            sf.DensityFeatures(),
            sf.GlobalSymmetryFeatures(),
            sf.Dimensionality(),
            sf.RadialDistributionFunction(),  # returns dict!
            sf.CoulombMatrix(),  # returns a matrix!
            sf.SineCoulombMatrix(),  # returns a matrix!
            sf.OrbitalFieldMatrix(),  # returns a matrix!
            sf.MinimumRelativeDistances(),  # returns a list
            sf.StructuralHeterogeneity(),
            sf.MaximumPackingEfficiency(),
            sf.ChemicalOrdering(),
            sf.XRDPowderPattern(),
            sf.SiteStatsFingerprint.from_preset(preset=preset_name),

            # these need oxidation states present in Structure:
            sf.ElectronicRadialDistributionFunction(),  # returns dict!
            sf.EwaldEnergy()
        ]
        if need_fit:
            featzers += [
            sf.PartialRadialDistributionFunction(),
            sf.BondFractions(),
            sf.BagofBonds()
        ]
        names = [c.__class__.__name__ for c in featzers]
        return [f for i, f in enumerate(featzers) if names[i] not in self.exclude]

    def dos(self):
        """
        All dos-based featurizers with default arguments.

        Returns ([matminer featurizer classes]):
        """
        featzers = [
            dosf.DOSFeaturizer(),
            dosf.DopingFermi(),
            dosf.Hybridization()
        ]
        names = [c.__class__.__name__ for c in featzers]
        return [f for i,f in enumerate(featzers) if names[i] not in self.exclude]

    def bandstructure(self):
        """
        All bandstructure-based featurizers with default arguments.

        Returns ([matminer featurizer classes]):
        """
        featzers = [
            bf.BandFeaturizer(),
            bf.BranchPointEnergy()
        ]
        names = [c.__class__.__name__ for c in featzers]
        return [f for i,f in enumerate(featzers) if names[i] not in self.exclude]
