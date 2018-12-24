import os
import unittest

import pandas as pd
from sklearn.metrics import r2_score, f1_score

from automatminer.configs import get_debug_config
from automatminer.automl.adaptors import TPOTAdaptor
from automatminer.utils.package_tools import AutomatminerError

__author__ = ['Qi Wang <qwang3@lbl.gov>', 'Alex Dunn <ardunn@lbl.gov>']


@unittest.skipIf("CI" in os.environ.keys(), "Test too intensive for CircleCI.")
class TestTPOTAdaptor(unittest.TestCase):
    def setUp(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_csv(basedir + "/mini_automl_df.csv", index_col=0)
        self.train_df = df.copy(deep=True).iloc[:450]
        self.test_df = df.copy(deep=True).iloc[451:]

    def test_regression(self):
        target_key = "K_VRH"
        debug_config = get_debug_config()
        tpot = TPOTAdaptor(**debug_config)
        tpot.fit(self.train_df, target_key)
        test_w_predictions = tpot.predict(self.test_df, target_key)
        y_true = test_w_predictions[target_key]
        y_test = test_w_predictions[target_key + " predicted"]
        self.assertTrue(r2_score(y_true, y_test) > 0.75)

    def test_classification(self):
        debug_config = get_debug_config()
        tpot = TPOTAdaptor(**debug_config)
        max_kvrh = 50
        classifier_key = "K_VRH > {}?".format(max_kvrh)
        train_df = self.train_df.rename(columns={"K_VRH": classifier_key})
        test_df = self.test_df.rename(columns={"K_VRH": classifier_key})
        train_df[classifier_key] = train_df[classifier_key] > max_kvrh
        test_df[classifier_key] = test_df[classifier_key] > max_kvrh
        tpot.fit(train_df, classifier_key)
        print(tpot.mode)
        test_w_predictions = tpot.predict(test_df, classifier_key)
        y_true = test_w_predictions[classifier_key]
        y_test = test_w_predictions[classifier_key + " predicted"]
        self.assertTrue(f1_score(y_true, y_test) > 0.75)

    def test_training_only(self):
        debug_config = get_debug_config()
        tpot = TPOTAdaptor(**debug_config)
        target_key = "K_VRH"
        train_w_predictions = tpot.fit_transform(self.train_df, target_key)
        y_true = train_w_predictions[target_key]
        y_test = train_w_predictions[target_key + " predicted"]
        self.assertTrue(r2_score(y_true, y_test) > 0.85)

    def test_feature_mismatching(self):
        debug_config = get_debug_config()
        tpot = TPOTAdaptor(**debug_config)
        target_key = "K_VRH"
        df1 = self.train_df
        df2 = self.test_df.rename(columns={'mean X': "some other feature"})
        tpot.fit(df1, target_key)
        with self.assertRaises(AutomatminerError):
            tpot.predict(df2, target_key)


if __name__ == '__main__':
    unittest.main()
