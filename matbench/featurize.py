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
    "formula", "structure", "bandstructure", "dos", etc.

    Args:
        df (pandas.DataFrame): the input data containing at least one of preset
            inputs (e.g. "formula")
        ignore_cols ([str]): if set, these columns are excluded
    """
    def __init__(self, df, ignore_cols=None):
        self.df = df.drop(ignore_cols, axis=1)


    def featurize_columns(self, input_cols=None):
        """
        Featurizes the dataframe based on input_columns.

        Args:
            input_cols ([str]): columns used for featurization (e.g. "structure"),
                set to None to try all preset columns.

        Returns (pandas.DataFrame):
            self.df w/ new features added via featurizering input_cols
        """
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
    @staticmethod
    def featurize_formula(df, col_id="formula",
                          preset_name="matminer", compcol="composition"):
        if compcol not in df:
            df[compcol] = df[col_id].apply(Composition)
        featurizer = MultipleFeaturizer([
            cf.ElementProperty.from_preset(preset_name=preset_name),
            cf.IonProperty()
        ])
        df = featurizer.featurize_dataframe(df, col_id=compcol)
        df = df.drop([compcol], axis=1)
        return df


    @staticmethod
    def featurize_structure(df, col_id="structure", preset_name="ops"):
        featurizer = MultipleFeaturizer([
            sf.SiteStatsFingerprint(
                site_featurizer=sf.CrystalSiteFingerprint.from_preset(
                    preset=preset_name), stats=('mean', 'std_dev', 'minimum','maximum')
            ),
            sf.DensityFeatures(),
            sf.GlobalSymmetryFeatures()
        ])
        df = featurizer.featurize_dataframe(col_id=col_id)
        return df



if __name__ == "__main__":
    df_init = load_double_perovskites_gap(return_lumo=False)
    prep = Featurize(df_init,
                       ignore_cols=['A1', 'A2', 'B1', 'B2'])
    df = prep.featurize_columns()
    print(df.head())