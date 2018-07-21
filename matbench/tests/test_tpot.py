import unittest

from matbench.automl.tpot_utils import TpotAutoml, ErrorAnalysis
from matbench.data.load import load_double_perovskites_gap, \
    load_glass_formation
from matbench.featurize import Featurize
from matbench.preprocess import PreProcess
from matminer.featurizers.composition import ElementProperty, TMetalFraction, \
    Stoichiometry
from sklearn.model_selection import train_test_split

__author__ = 'Alireza Faghaninia <alireza.faghaninia@gmail.com>'

class TestTpotAutoml(unittest.TestCase):

    def setUp(self):
        self.RS = 27

    def test_tpot_regression(self, limit=500):
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

        tpot = TpotAutoml(mode='regressor',
                          generations=1,
                          population_size=25,
                          scoring='r2',
                          random_state=self.RS,
                          feature_names=df.drop(target_col, axis=1).columns)
        self.assertTrue(tpot.scoring_function=='r2')
        tpot.fit(X_train, y_train)
        top_scores = tpot.get_top_models(return_scores=True)
        self.assertTrue(tpot.greater_score_is_better)
        self.assertAlmostEqual(top_scores['XGBRegressor'], 0.8622, 1)
        self.assertAlmostEqual(top_scores['ExtraTreesRegressor'], 0.7933, 1)
        self.assertAlmostEqual(top_scores['DecisionTreeRegressor'], 0.7709, 1)
        self.assertAlmostEqual(top_scores['LassoLarsCV'], 0.7058, 1)
        self.assertAlmostEqual(top_scores['RandomForestRegressor'], 0.7495, 1)
        self.assertAlmostEqual(top_scores['GradientBoostingRegressor'], 0.7352, 1)
        self.assertAlmostEqual(top_scores['ElasticNetCV'], 0.7124, 1)
        self.assertAlmostEqual(top_scores['KNeighborsRegressor'], 0.4808, 1)
        self.assertAlmostEqual(top_scores['LinearSVR'], 0.5, 1)
        test_score = tpot.score(X_test, y_test)
        self.assertAlmostEqual(test_score, 0.8707, places=1)

        # test error analysis:
        ea = ErrorAnalysis(tpot, X_train, y_train, X_test, y_test,
                           mode='regression', target=target_col,
                           features=df.drop(target_col, axis=1).columns,
                           test_samples_index=y_test.index, random_state=self.RS)
        df_errors = ea.get_data_for_error_analysis()
        self.assertTrue((df_errors['{}_true'.format(target_col)]!=\
                         df_errors['{}_predicted'.format(target_col)]).all())


    def test_tpot_classification(self, limit=500):
        target_col = 'gfa'
        # load and featurize:
        df_init = load_glass_formation(phase='binary')[:limit]
        featzer = Featurize(df_init)
        df_feats = featzer.featurize_formula(featurizers=[
            ElementProperty.from_preset(preset_name='matminer'),
            Stoichiometry()])
        # preprocessing of the data
        prep = PreProcess(max_colnull=0.1)
        df = prep.handle_nulls(df_feats)
        # train/test split (development is within tpot crossvalidation)
        X_train, X_test, y_train, y_test = \
            train_test_split(df.drop(target_col, axis=1).as_matrix(),
            df[target_col], train_size=0.75, test_size=0.25, random_state=self.RS)

        tpot = TpotAutoml(mode='classify',
                          generations=1,
                          population_size=25,
                          scoring='f1_weighted',
                          random_state=self.RS,
                          feature_names=df.drop(target_col, axis=1).columns)
        self.assertTrue(tpot.scoring_function=='f1_weighted')
        tpot.fit(X_train, y_train)
        top_scores = tpot.get_top_models(return_scores=True)

        self.assertAlmostEqual(top_scores['DecisionTreeClassifier'], 0.91, 1)
        self.assertAlmostEqual(top_scores['RandomForestClassifier'], 0.89, 1)
        self.assertAlmostEqual(top_scores['GradientBoostingClassifier'], 0.88, 1)
        self.assertAlmostEqual(top_scores['XGBClassifier'], 0.87, 1)
        self.assertAlmostEqual(top_scores['ExtraTreesClassifier'], 0.86, 1)
        self.assertAlmostEqual(top_scores['BernoulliNB'], 0.84, 1)
        self.assertAlmostEqual(top_scores['KNeighborsClassifier'], 0.84, 1)
        self.assertAlmostEqual(top_scores['LogisticRegression'], 0.84, 1)
        self.assertAlmostEqual(top_scores['LinearSVC'], 0.84, 1)
        self.assertAlmostEqual(top_scores['GaussianNB'], 0.66, 1)

        # test error analysis:
        ea = ErrorAnalysis(tpot, X_train, y_train, X_test, y_test,
                           mode='classification', target=target_col,
                           features=df.drop(target_col, axis=1).columns,
                           test_samples_index=y_test.index, random_state=self.RS)
        df_errors = ea.get_data_for_error_analysis()
        self.assertTrue((df_errors['{}_true'.format(target_col)] !=\
                         df_errors['{}_predicted'.format(target_col)]).all())


if __name__ == '__main__':
    unittest.main()