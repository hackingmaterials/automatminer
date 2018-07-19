import unittest

from matbench.automl.tpot_utils import TpotAutoml
from matbench.data.load import load_double_perovskites_gap
from matbench.featurize import Featurize
from matbench.preprocess import PreProcess
from matminer.featurizers.composition import ElementProperty, TMetalFraction
from sklearn.model_selection import train_test_split

class TestTpotAutoml(unittest.TestCase):

    def setUp(self):
        self.RS = 27

    def test_tpot(self, limit=500):
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
            df[target_col], train_size=0.75, test_size=0.25, random_state=self.RS)

        tpot = TpotAutoml(model_type='regressor',
                          generations=1,
                          population_size=25,
                          scoring='r2',
                          random_state=self.RS)
        self.assertTrue(tpot.scoring_function=='r2')
        tpot.fit(X_train, y_train)

        top_scores = tpot.get_top_models(return_scores=True)
        print(top_scores)
        self.assertTrue(tpot.greater_score_is_better)
        self.assertAlmostEqual(top_scores['XGBRegressor'], 0.8622, 3)
        self.assertAlmostEqual(top_scores['ExtraTreesRegressor'], 0.7933, 3)
        self.assertAlmostEqual(top_scores['DecisionTreeRegressor'], 0.7709, 3)
        self.assertAlmostEqual(top_scores['LassoLarsCV'], 0.7058, 3)
        self.assertAlmostEqual(top_scores['RandomForestRegressor'], 0.7495, 3)
        self.assertAlmostEqual(top_scores['GradientBoostingRegressor'], 0.7352, 3)
        self.assertAlmostEqual(top_scores['ElasticNetCV'], 0.7124, 3)
        self.assertAlmostEqual(top_scores['KNeighborsRegressor'], 0.4808, 2)
        self.assertAlmostEqual(top_scores['LinearSVR'], 0.5, 1)
        test_score = tpot.score(X_test, y_test)
        self.assertAlmostEqual(test_score, 0.8707, places=3)


if __name__ == '__main__':
    unittest.main()