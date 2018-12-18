import os
import unittest

import pandas as pd
from sklearn.metrics import r2_score, f1_score
from automatminer.automl.adaptors import NeuralNetworkAdaptor
from automatminer.utils.package_tools import AutomatminerError

__author__ = ['Qi Wang <qwang3@lbl.gov>', 'Alex Dunn <ardunn@lbl.gov>', 'Samy Cherfaoui <SCherfaoui@lbl.gov>']

class TestNNAdaptor(unittest.TestCase):
    def setUp(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_csv(basedir + "/mini_automl_df.csv", index_col=0)
        self.train_df = df.copy(deep=True).iloc[:450]
        self.test_df = df.copy(deep=True).iloc[451:]

    def test_regression(self):
        target_key = "K_VRH"
        nn = NeuralNetworkAdaptor()
        nn.fit(self.train_df, target_key)
        test_w_predictions = nn.predict(self.test_df, target_key)
        y_true = test_w_predictions[target_key]
        y_test = test_w_predictions[target_key + " predicted"]
        self.assertTrue(r2_score(y_true, y_test) > 0.6)

    def test_classification(self):
        nn = NeuralNetworkAdaptor()
        max_kvrh = 50
        classifier_key = "K_VRH > {}?".format(max_kvrh)
        train_df = self.train_df.rename(columns={"K_VRH": classifier_key})
        test_df = self.test_df.rename(columns={"K_VRH": classifier_key})
        train_df[classifier_key] = train_df[classifier_key] > max_kvrh
        test_df[classifier_key] = test_df[classifier_key] > max_kvrh
        nn.fit(train_df, classifier_key)
        print(nn.mode)
        test_w_predictions = nn.predict(test_df, classifier_key)
        y_true = test_w_predictions[classifier_key]
        y_test = test_w_predictions[classifier_key + " predicted"]
        self.assertTrue(f1_score(y_true, y_test) > 0.7)

    def test_training_only(self):
        target_key = "K_VRH"
        nn = NeuralNetworkAdaptor()
        nn.fit(self.train_df, target_key)
        test_w_predictions = nn.predict(self.train_df, target_key)
        y_true = test_w_predictions[target_key]
        y_test = test_w_predictions[target_key + " predicted"]
        self.assertTrue(r2_score(y_true, y_test) > 0.6)

    def test_feature_mismatching(self):
        nn = NeuralNetworkAdaptor()
        target_key = "K_VRH"
        df1 = self.train_df
        df2 = self.test_df.rename(columns={'mean X': "some other feature"})
        nn.fit(df1, target_key)
        with self.assertRaises(AutomatminerError):
            nn.predict(df2, target_key)


if __name__ == '__main__':
    unittest.main()
