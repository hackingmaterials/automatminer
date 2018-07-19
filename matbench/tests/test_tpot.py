import unittest

from matbench.automl.tpot_utils import TpotAutoml
from matbench.data.load import load_double_perovskites_gap
from matbench.featurize import Featurize
from matbench.preprocess import PreProcess
from matminer.featurizers.composition import ElementProperty, TMetalFraction
from sklearn.model_selection import train_test_split

class TestTpotAutoml(unittest.TestCase):

    def test_tpot(self, limit=200):
        target_col = 'gap gllbsc'
        # load and featurize:
        df_init = load_double_perovskites_gap(return_lumo=False)[:limit]
        featzer = Featurize(df_init, ignore_cols=['a_1', 'b_1', 'a_2', 'b_2'])
        df_feats = featzer.featurize_formula(featurizers=[
            ElementProperty.from_preset(preset_name='matminer'),
            TMetalFraction()])
        # preprocessing of the data
        prep = PreProcess(max_colnull=0.1)
        df = prep.handle_nulls(df_feats)
        # train/test split (development is within tpot crossvalidation)
        X_train, X_test, y_train, y_test = \
            train_test_split(df.drop(target_col, axis=1).as_matrix(),
            df[target_col], train_size=0.75, test_size=0.25)

        tpot = TpotAutoml(model_type='regressor',
                          generations=1,
                          population_size=25,
                          scoring='r2',
                          random_state=23)
        self.assertTrue(tpot.scoring_function=='r2')
        tpot.fit(X_train, y_train)

        selected_models = tpot.get_selected_models()
        self.assertTrue(tpot.greater_score_is_better)


if __name__ == '__main__':
    unittest.main()