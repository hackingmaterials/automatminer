from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf
import matminer.featurizers.structure as sf
from pymatgen import Composition
from matbench.data.load import load_double_perovskites_gap
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
    """
    def __init__(self, df, ignore_cols=None, preset_name="matminer"):
        self.df = df if ignore_cols is None else df.drop(ignore_cols, axis=1)
        self.all_featurizers = AllFeaturizers(preset_name=preset_name)

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

    #TODO: -AF see if the only use of featurize_* methods is to be called in featurize_columns, think about defining them outside of the class
    def featurize_formula(self, df=None, featurizers='all', col_id="formula",
                          compcol="composition", ignore_errors=True):
        if df is None:
            df = self.df.copy(deep=True)
        if compcol not in df:
            df[compcol] = df[col_id].apply(Composition)
        if featurizers=='all':
            featurizer = MultipleFeaturizer(self.all_featurizers.composition)
        else:
            featurizer = MultipleFeaturizer(featurizers)
        df = featurizer.featurize_dataframe(df,
                                            col_id=compcol,
                                            ignore_errors=ignore_errors)
        return df


    def featurize_structure(self, df=None, col_id="structure", preset_name="ops"):
        if df is None:
            df = self.df.copy(deep=True)
        featurizer = MultipleFeaturizer([
            sf.SiteStatsFingerprint(
                site_featurizer=sf.CrystalSiteFingerprint.from_preset(
                    preset=preset_name), stats=('mean', 'std_dev', 'minimum','maximum')
            ),
            sf.DensityFeatures(),
            sf.GlobalSymmetryFeatures()
        ])
        df = featurizer.featurize_dataframe(df, col_id=col_id)
        return df


class AllFeaturizers(object):

    def __init__(self, preset_name="matminer"):
        self.preset_name = preset_name

    @property
    def composition(self, preset_name=None):
        preset_name = preset_name or self.preset_name
        return [
                cf.ElementProperty.from_preset(preset_name=preset_name),
                cf.IonProperty()
            ]


if __name__ == "__main__":
    df_init = load_double_perovskites_gap(return_lumo=False)
    featurizer = Featurize(df_init,
                     ignore_cols=['a_1', 'a_2', 'b_1', 'b_2'])
    df = featurizer.featurize_columns()
    # df = featurizer.featurize_formula(df_init, featurizers='all')

    print(df.head())
    print('The original df')
    print(featurizer.df)