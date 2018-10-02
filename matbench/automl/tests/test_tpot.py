import unittest
from collections import OrderedDict

import numpy as np
from tpot import TPOTClassifier
from matbench.automl.tpot_utils import TpotAutoml
from matbench.automl.tpot_configs.classifier import classifier_config_dict_mb
from matbench.automl.tpot_configs.regressor import regressor_config_dict_mb
from matbench.analysis.analysis import Analysis
from matbench.data.load import load_double_perovskites_gap, \
    load_glass_binary
from matbench.featurization.core import Featurize
from matbench.preprocessing.core import Preprocess
from matminer.featurizers.composition import ElementProperty, TMetalFraction, \
    Stoichiometry
from sklearn.model_selection import train_test_split

__author__ = 'Alireza Faghaninia <alireza.faghaninia@gmail.com>, ' \
             'Qi Wang <wqthu11@gmail.com>'


class TestTpotAutoml(unittest.TestCase):

    def setUp(self):
        self.RS = 27

    def test_tpot_regression(self, limit=500):
        target = 'gap gllbsc'
        # load and featurize:
        df_init = load_double_perovskites_gap(return_lumo=False)[:limit]
        featzer = Featurize(ignore_cols=['a_1', 'b_1', 'a_2', 'b_2'])
        df_feats = featzer.featurize_formula(df_init, featurizers=[
            ElementProperty.from_preset(preset_name='matminer'),
            TMetalFraction()])
        # preprocessing of the data
        prep = Preprocess()
        df = prep.handle_na(df_feats, max_na_frac=0.1)
        feats0 = set(df.columns)
        df = prep.prune_correlated_features(df, target, R_max=0.95)
        self.assertEqual(len(feats0 - set(df.columns)), 17)
        # train/test split (train/dev splot done within tpot crossvalidation)
        X_train, X_test, y_train, y_test = \
            train_test_split(df.drop(target, axis=1).values,
            df[target], train_size=0.75, test_size=0.25, random_state=self.RS)

        tpot = TpotAutoml(mode='regressor',
                          generations=1,
                          population_size=25,
                          scoring='r2',
                          random_state=self.RS,
                          feature_names=df.drop(target, axis=1).columns,
                          n_jobs=1)
        # self.assertTrue(tpot.scoring_function=='r2')
        tpot.fit(X_train, y_train)
        top_scores = tpot.get_top_models(return_scores=True)

        # test customed config_dict
        # .config_dict is changed to ._config_dict in 0.9.5 tpot version
        self.assertTrue(tpot.config_dict == regressor_config_dict_mb)

        self.assertTrue(tpot.greater_score_is_better)

        top_scores_keys = list(top_scores.keys())
        config_keys = [x.split('.')[-1]
                       for x in regressor_config_dict_mb.keys()]
        self.assertEqual(set(top_scores_keys).issubset(config_keys), True)
        self.assertLessEqual(top_scores[top_scores_keys[0]], 1)
        self.assertGreaterEqual(top_scores[top_scores_keys[-1]], 0)

        # test error analysis:
        ea = Analysis(tpot, X_train, y_train, X_test, y_test,
                      mode='regression', target=target,
                      features=df.drop(target, axis=1).columns,
                      test_samples_index=y_test.index,
                      random_state=self.RS)
        df_errors = ea.get_data_for_error_analysis()
        self.assertTrue((df_errors['{}_true'.format(target)]!=\
                         df_errors['{}_predicted'.format(target)]).all())
        rmse = np.sqrt(np.mean((tpot.predict(X_test) - y_test) ** 2))
        self.assertTrue((
            ea.false_positives['{}_predicted'.format(target)]>= \
            ea.false_positives['{}_true'.format(target)] + rmse).all())

        self.assertTrue((
            ea.false_negatives['{}_predicted'.format(target)]<= \
            ea.false_negatives['{}_true'.format(target)] - rmse).all())

        # test feature importance
        feature_importance = ea.get_feature_importance(sort=True)
        self.assertEqual(list(feature_importance.items())[0][0],
                         'transition metal fraction')
        self.assertAlmostEqual(feature_importance['range melting_point'], 0.1, 1)

    def test_tpot_classification(self, limit=500):
        target = 'gfa'
        # load and featurize:
        df_init = load_glass_binary()[:limit]
        featzer = Featurize()
        df_feats = featzer.featurize_formula(df_init, featurizers=[
            ElementProperty.from_preset(preset_name='matminer'),
            Stoichiometry()])
        # preprocessing of the data
        prep = Preprocess()
        df = prep.handle_na(df_feats, max_na_frac=0.1)
        feats0 = set(df.columns)
        df = prep.prune_correlated_features(df, target, R_max=0.95)
        self.assertEqual(len(feats0 - set(df.columns)), 49)
        # train/test split (development is within tpot crossvalidation)
        X_train, X_test, y_train, y_test = \
            train_test_split(df.drop(target, axis=1).values,
            df[target], train_size=0.75, test_size=0.25, random_state=self.RS)

        tpot = TpotAutoml(mode='classify',
                          generations=1,
                          population_size=25,
                          scoring='f1_weighted',
                          random_state=self.RS,
                          feature_names=df.drop(target, axis=1).columns,
                          n_jobs=1)
        tpot.fit(X_train, y_train)
        top_scores = tpot.get_top_models(return_scores=True)

        # test customed config_dict
        self.assertTrue(tpot.config_dict == classifier_config_dict_mb)
        self.assertTrue(tpot.greater_score_is_better)

        top_scores_keys = list(top_scores.keys())
        config_keys = [x.split('.')[-1]
                       for x in classifier_config_dict_mb.keys()]
        self.assertEqual(set(top_scores_keys).issubset(config_keys), True)
        self.assertLessEqual(top_scores[top_scores_keys[0]], 1)
        self.assertGreaterEqual(top_scores[top_scores_keys[-1]], 0.4)

        # test analysis:
        ea = Analysis(tpot, X_train, y_train, X_test, y_test,
                      mode='classification', target=target,
                      features=df.drop(target, axis=1).columns,
                      test_samples_index=y_test.index,
                      random_state=self.RS)
        df_errors = ea.get_data_for_error_analysis()
        self.assertTrue((df_errors['{}_true'.format(target)] !=\
                         df_errors['{}_predicted'.format(target)]).all())
        self.assertTrue(not ea.false_negatives['gfa_predicted'].all() and \
                        ea.false_negatives['gfa_true'].all())
        self.assertTrue(ea.false_positives['gfa_predicted'].all() and \
                        not ea.false_positives['gfa_true'].all())

        # test feature importance
        ea.get_feature_importance(sort=True)
        self.assertTrue(isinstance(ea.feature_importance, OrderedDict))
        # feature_importance = list(ea.feature_importance.items())
        # self.assertEqual('mean block',  feature_importance[0][0])
        # self.assertAlmostEqual(feature_importance[0][1], 0.6, 1)

    def test_customed_configs(self):
        tpot_obj = TPOTClassifier(config_dict=classifier_config_dict_mb)
        # This is not in the 0.9.3 tpot version
        # tpot_obj._fit_init()

        self.assertTrue(isinstance(tpot_obj.config_dict, dict))
        self.assertTrue(tpot_obj.config_dict == classifier_config_dict_mb)


if __name__ == '__main__':
    unittest.main()