import os
import unittest
from collections import OrderedDict

import numpy as np
import pandas as pd
from tpot import TPOTClassifier
from matminer.featurizers.composition import AtomicOrbitals, ElementFraction
from mslearn.automl.tpot_configs.classifier import classifier_config_dict_mb
from mslearn.automl.tpot_configs.regressor import regressor_config_dict_mb
from mslearn.featurization.core import AutoFeaturizer
from sklearn.metrics import r2_score, f1_score

from mslearn.automl.adaptors import TPOTAdaptor
from mslearn.utils.utils import MatbenchError

__author__ = ['Alex Dunn <ardunn@lbl.gov']


class TestTPOTAdaptor(unittest.TestCase):
    def setUp(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        self.train_df = pd.read_csv(basedir + "/mini_training_df_automl.csv",
                                    index_col=0)
        self.test_df = pd.read_csv(basedir + "/mini_test_df_automl.csv",
                                   index_col=0)
        self.common_tpot_kwargs = {"max_time_mins": 3, "max_eval_time_mins": 1}

    def test_regression(self):
        target_key = "K_VRH"
        tpot = TPOTAdaptor(mode="regression", **self.common_tpot_kwargs)
        tpot.fit(self.train_df, target_key)
        test_w_predictions = tpot.predict(self.test_df, target_key)
        y_true = test_w_predictions[target_key]
        y_test = test_w_predictions[target_key + " predicted"]
        self.assertTrue(r2_score(y_true, y_test) > 0.75)

    def test_classification(self):
        tpot = TPOTAdaptor(mode="classification", **self.common_tpot_kwargs)
        max_kvrh = 50
        classifier_key = "K_VRH > {}?".format(max_kvrh)
        train_df = self.train_df.rename(columns={"K_VRH": classifier_key})
        train_df[classifier_key] = train_df[classifier_key] > max_kvrh
        tpot.fit(train_df, classifier_key)
        test_w_predictions = tpot.predict(self.test_df, classifier_key)
        y_true = test_w_predictions[classifier_key]
        y_test = test_w_predictions[classifier_key + " predicted"]
        self.assertTrue(f1_score(y_true, y_test) > 0.75)

    def test_training_only(self):
        tpot = TPOTAdaptor(mode="regression", **self.common_tpot_kwargs)
        target_key = "K_VRH"
        train_w_predictions = tpot.fit_transform(self.train_df, target_key)
        y_true = train_w_predictions[target_key]
        y_test = train_w_predictions[target_key + " predicted"]
        self.assertTrue(f1_score(y_true, y_test) > 0.85)

    def test_feature_mismatching(self):
        tpot = TPOTAdaptor(mode="regression", **self.common_tpot_kwargs)
        target_key = "K_VRH"
        df1 = self.test_df
        df2 = self.train_df.rename(columns={'mean X': "some other feature"})
        tpot.fit(df1, target_key)
        with self.assertRaises(MatbenchError):
            tpot.predict(df2, target_key)


if __name__ == '__main__':
    unittest.main()
