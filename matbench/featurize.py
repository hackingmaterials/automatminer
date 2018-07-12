import pandas as pd

from matbench.data.generate import generate_mp
from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
import matminer.featurizers.dos as dosf
import matminer.featurizers.bandstructure as bf
from matminer.utils.conversions import composition_to_oxidcomposition, \
    structure_to_oxidstructure
from pymatgen import Composition, Structure
from matbench.data.load import load_castelli_perovskites
from matbench.utils.utils import MatbenchError
from warnings import warn

from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.dos import CompleteDos


class Featurize(object):
    """
    Takes in a dataframe and generate features from preset columns such as
    "formula", "structure", "bandstructure", "dos", etc. One may use
    the featurize_columns method to featurize via all available featurizers
    with default setting or selectively call featurizer methods.
    Usage examples:
        featurizer = Featurize(df)
            df = featurizer.featurize_columns() # all features of all types
        or:
            df = featurizer.featurize_formula() # all formula-related feature
        or:
            df = featurizer.

    Args:
        df (pandas.DataFrame): the input data containing at least one of preset
            inputs (e.g. "formula")
        ignore_cols ([str]): if set, these columns are excluded
        preset_name (str): some featurizers (w/ from_preset) take in this arg
        ignore_errors (bool): whether to ignore exceptions raised when
            featurize_dataframe is called
    """
    def __init__(self, df, ignore_cols=None, preset_name="matminer",
                 ignore_errors=True):
        self.df = df if ignore_cols is None else df.drop(ignore_cols, axis=1)
        self.all_featurizers = AllFeaturizers(preset_name=preset_name)
        self.ignore_errors = ignore_errors


    def _preprocess_df(self, df, inplace=True, col_id=None):
        if df is None:
            df = self.df.copy(deep=True)
        if not inplace:
            df = df.copy(deep=True)
        if col_id and col_id not in df:
            raise MatbenchError("'{}' column must be in data!".format(col_id))
        return df


    def featurize_columns(self, df=None, input_cols=None):
        """
        Featurizes the dataframe based on input_columns.

        Args:
            input_cols ([str]): columns used for featurization (e.g. "structure"),
                set to None to try all preset columns.

        Returns (pandas.DataFrame):
            self.df w/ new features added via featurizering input_cols
        """
        df = self._preprocess_df(df)
        input_cols = input_cols or ["formula"]
        for column in input_cols:
            featurizer = getattr(self, "featurize_{}".format(column), None)
            if featurizer is not None:
                df = featurizer(df)
            elif column not in df:
                raise MatbenchError('no "{}" in the data!')
            else:
                warn('No method available to featurize "{}"'.format(column))
        return df


    def featurize_formula(self, df=None, featurizers="all", col_id="formula",
                          compcol="composition", guess_oxidstates=True,
                          inplace=True):
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

        Returns (pandas.DataFrame):
            Dataframe with compositional features added.
        """
        df = self._preprocess_df(df=df, inplace=inplace)
        if compcol not in df:
            df[compcol] = df[col_id].apply(Composition)
        if guess_oxidstates:
            df[compcol] = composition_to_oxidcomposition(df[compcol])
        if featurizers=='all':
            featurizers = self.all_featurizers.composition()
        df = MultipleFeaturizer(featurizers).featurize_dataframe(
            df, col_id=compcol, ignore_errors=self.ignore_errors)
        return df


    def featurize_structure(self, df=None, featurizers="all",
                            fit_featurizers="all", col_id="structure",
                            inplace=True, guess_oxidstates=True):
        """
        Featurizes based on crystal structure (pymatgen Structure object)

        Args:
            df (pandas.DataFrame):
            col_id (str): column name containing pymatgen Structure
        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.featurizer] or "all"):
            fit_featurizers ([matminer.featurizer] or "all"): those featurizers
                that require the fit method to be called first.
            col_id (str): actual column name to be used as structure
            inplace (bool): whether to modify the input df
            guess_oxidstates (bool): whether to guess elements oxidation states
                in the structure which is required for some featurizers such as
                EwaldEnergy, ElectronicRadialDistributionFunction. Set to
                False if oxidation states already available in the structure.

        Returns (pandas.DataFrame):
            Dataframe with structure features added.
        """
        df = self._preprocess_df(df=df, inplace=inplace, col_id=col_id)
        if isinstance(df[col_id][0], dict):
            df[col_id] = df[col_id].apply(Structure.from_dict)
        if guess_oxidstates:
            structure_to_oxidstructure(df[col_id], inplace=True)
        if featurizers == "all":
            featurizers = self.all_featurizers.structure()
        df = MultipleFeaturizer(featurizers).featurize_dataframe(
            df, col_id=col_id, ignore_errors=self.ignore_errors)
        if fit_featurizers=="all":
            featurizers = self.all_featurizers.fit_structure()
        for featzer in featurizers:
            featzer.fit(df[col_id])
            df = featzer.featurize_dataframe(df,
                                             col_id=col_id,
                                             ignore_errors=self.ignore_errors)
        return df


    def featurize_dos(self, df=None, featurizers="all", col_id="dos",
                      inplace=True):
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

        Returns (pandas.DataFrame):
            Dataframe with dos features added.
        """
        df = self._preprocess_df(df=df, inplace=inplace, col_id=col_id)
        if isinstance(df[col_id][0], dict):
            df[col_id] = df[col_id].apply(CompleteDos.from_dict)
        if featurizers == "all":
            featurizers = self.all_featurizers.dos()
        df = MultipleFeaturizer(featurizers).featurize_dataframe(
            df, col_id=col_id, ignore_errors=self.ignore_errors)
        return df


    def featurize_bandstructure(self, df=None, featurizers="all",
                                col_id="bandstructure", inplace=True):
        """
        Featurizes based on density of state (pymatgen BandStructure object)

        Args:
            df (pandas.DataFrame):
            col_id (str): column name containing pymatgen BandStructure
        Args:
            df (pandas.DataFrame): input data
            featurizers ([matminer.featurizer] or "all"):
            col_id (str): actual column name to be used as bandstructure
            inplace (bool): whether to modify the input df

        Returns (pandas.DataFrame):
            Dataframe with bandstructure features added.
        """
        df = self._preprocess_df(df=df, inplace=inplace, col_id=col_id)
        if isinstance(df[col_id][0], dict):
            df[col_id] = df[col_id].apply(BandStructure.from_dict)
        if featurizers == "all":
            featurizers = self.all_featurizers.bandstructure()
        df = MultipleFeaturizer(featurizers).featurize_dataframe(
            df, col_id=col_id, ignore_errors=self.ignore_errors)
        return df



class AllFeaturizers(object):
    """
    Contains all available featurizers that take various types of inputs such
    as "composition" or "formula", "structure", "dos", "band structure", etc.
    The point of this class is to have one place where available featurizers
    with default inputs are updated synced with matminer.

    Args:
        preset_name (str): some featurizers take in this argument
    """
    def __init__(self, preset_name="matminer"):
        self.preset_name = preset_name

    def composition(self, preset_name=None, extras=False):
        """
        All composition-based featurizers with default arguments.

        Args:
            preset_name (str): some featurizers take in this argument
            extras (bool): Include "niche" composition featurizers

        Returns ([matminer featurizer classes]):

        """
        preset_name = preset_name or self.preset_name
        featzers =  [
            cf.ElementProperty.from_preset(preset_name=preset_name),
            cf.AtomicOrbitals(),
            cf.BandCenter(),
            cf.IonProperty(),
            cf.Stoichiometry(),
            cf.ValenceOrbital(),
            # cf.ElementFraction(), # too many features?
            cf.TMetalFraction(),
            # cf.CohesiveEnergy(), # an entry must be found in materialsproject.org
            # TODO-Qi: what is the requirement for elements? wasn't clear at the top of class's documentation
            # cf.Miedema(),
            # cf.YangSolidSolution(),
            cf.AtomicPackingEfficiency(), # much slower than the rest

            # these need oxidation states present in Composition:
            cf.CationProperty.from_preset(preset_name='deml'),
            cf.OxidationStates.from_preset(preset_name='deml'),
            cf.ElectronAffinity(),
            cf.ElectronegativityDiff(),
        ]

        if extras:
            featzers.append([cf.ElementFraction(), cf.Miedema(), cf.YangSolidSolution()])

        return featzers


    def structure(self, preset_name="CrystalNNFingerprint_ops"):
        """
        All structure-based featurizers with default arguments that don't
        require the fit method to be called first.

        Args:
            preset_name (str): some featurizers take in this argument

        Returns ([matminer featurizer classes]):
        """
        preset_name = preset_name or self.preset_name
        return [
            sf.DensityFeatures(),
            sf.GlobalSymmetryFeatures(),
            sf.Dimensionality(),
            sf.RadialDistributionFunction(), # returns dict!
            sf.CoulombMatrix(), # returns a matrix!
            sf.SineCoulombMatrix(), # returns a matrix!
            sf.OrbitalFieldMatrix(), # returns a matrix!
            sf.MinimumRelativeDistances(), # returns a list
            sf.StructuralHeterogeneity(),
            sf.MaximumPackingEfficiency(),
            sf.ChemicalOrdering(),
            sf.XRDPowderPattern(),
            sf.SiteStatsFingerprint.from_preset(preset=preset_name),

            # these need oxidation states present in Structure:
            sf.ElectronicRadialDistributionFunction(),
            sf.EwaldEnergy(accuracy=4), #TODO: remove this accuracy=4 when new version of matminer is released
        ]


    def fit_structure(self):
        """
        Structure-based featurizers with default arguments that require the
        fit method to be called first.

        Returns ([matminer featurizer classes]):
        """
        return [
            # sf.PartialRadialDistributionFunction(), # got the error AssertionError: 13200 columns passed, passed data had 13260 columns
            sf.BondFractions(),
            sf.BagofBonds()
        ]


    def dos(self):
        """
        All dos-based featurizers with default arguments.

        Returns ([matminer featurizer classes]):
        """
        return [
            dosf.DOSFeaturizer(),
            dosf.DopingFermi(),
            dosf.BandEdge()
        ]


    def bandstructure(self):
        """
        All bandstructure-based featurizers with default arguments.

        Returns ([matminer featurizer classes]):
        """
        return [
            bf.BandFeaturizer(),
            bf.BranchPointEnergy()
        ]