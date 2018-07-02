from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
from matminer.utils.conversions import composition_to_oxidcomposition, \
    structure_to_oxidstructure
from pymatgen import Composition, Structure
from matbench.data.load import load_castelli_perovskites
from matbench.utils.utils import MatbenchError
from warnings import warn


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


    def featurize_columns(self, df=None, input_cols=None):
        """
        Featurizes the dataframe based on input_columns.

        Args:
            input_cols ([str]): columns used for featurization (e.g. "structure"),
                set to None to try all preset columns.

        Returns (pandas.DataFrame):
            self.df w/ new features added via featurizering input_cols
        """
        if df is None:
            df = self.df.copy(deep=True)
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
        if df is None:
            df = self.df.copy(deep=True)
        if not inplace:
            df = df.copy(deep=True)
        if compcol not in df:
            df[compcol] = df[col_id].apply(Composition)
        if guess_oxidstates:
            df[compcol] = composition_to_oxidcomposition(df[compcol])
        if featurizers=='all':
            featurizer = MultipleFeaturizer(self.all_featurizers.composition)
        else:
            featurizer = MultipleFeaturizer(featurizers)
        df = featurizer.featurize_dataframe(df,
                                            col_id=compcol,
                                            ignore_errors=self.ignore_errors)
        return df


    def featurize_structure(self, df=None, col_id="structure", inplace=True,
                            guess_oxidstates=True):
        """
        Featurizes based on crystal structure (pymatgen Structure object)

        Args:
            df (pandas.DataFrame):
            col_id (str): column name containing pymatgen Structure

        Returns (pandas.DataFrame):
            Dataframe with structural features added.
        """
        if df is None:
            df = self.df.copy(deep=True)
        if not inplace:
            df = df.copy(deep=True)
        if col_id not in df:
            raise MatbenchError("'{}' column must be in data!".format(col_id))
        if isinstance(df[col_id][0], dict):
            df[col_id] = df[col_id].apply(Structure.from_dict)
        if guess_oxidstates:
            structure_to_oxidstructure(df[col_id], inplace=True)
        featurizer = MultipleFeaturizer(self.all_featurizers.structure)
        df = featurizer.featurize_dataframe(df,
                                            col_id=col_id,
                                            ignore_errors=self.ignore_errors)
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

    @property
    def composition(self, preset_name=None):
        """
        All composition-based featurizers with default arguments.

        Args:
            preset_name (str): some featurizers take in this argument

        Returns ([matminer featurizer classes]):

        """
        preset_name = preset_name or self.preset_name
        return [
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
            cf.AtomicPackingEfficiency(), # much slower than the rest so far

            # these need oxidation states present in Composition:
            cf.CationProperty.from_preset(preset_name='deml'),
            cf.OxidationStates.from_preset(preset_name='deml'),
            cf.ElectronAffinity(),
            cf.ElectronegativityDiff(),
        ]


    @property
    def structure(self, preset_name="CrystalNNFingerprint_ops"):
        """
        All structure-based featurizers with default arguments.

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
            sf.SiteStatsFingerprint.from_preset(preset=preset_name),

            # these need oxidation states present in Structure:
            sf.ElectronicRadialDistributionFunction(),
            sf.EwaldEnergy(accuracy=12),
            # sf.EwaldEnergy(),
            sf.StructuralHeterogeneity(),
            sf.MaximumPackingEfficiency(),
            sf.ChemicalOrdering(),
            sf.XRDPowderPattern(),

            # these require calling fit first:
            # sf.PartialRadialDistributionFunction()
            # sf.BondFractions(),
            # sf.BagofBonds()

            # TODO: add more featurizers here
        ]

    # TODO: add dos, band_structure, etc featurizers


if __name__ == "__main__":
    df_init = load_castelli_perovskites()[:5]
    featurizer = Featurize(df_init, ignore_errors=False)
    df = featurizer.featurize_structure(df_init)
    df.to_csv('test.csv')
    print(df)

    print('The original df')
    print(featurizer.df)