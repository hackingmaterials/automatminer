import os
import unittest

import pandas as pd
from sklearn.metrics import r2_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline

from automatminer.presets import get_preset_config
from automatminer.automl.adaptors import SinglePipelineAdaptor
from automatminer.utils.pkg import AutomatminerError

__author__ = ['Qi Wang <qwang3@lbl.gov>', 'Alex Dunn <ardunn@lbl.gov>']


@unittest.skipIf(int(os.environ.get("SKIP_INTENSIVE", 0)),
                 "Test too intensive for CircleCI commit builds.")
class TestTPOTAdaptor(unittest.TestCase):
    def setUp(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_csv(basedir + "/mini_automl_df.csv", index_col=0)
        self.train_df = df.copy(deep=True).iloc[:450]
        self.test_df = df.copy(deep=True).iloc[451:]
        self.tpot = get_preset_config("debug")["learner"]

    def test_regression(self):
        target_key = "K_VRH"
        self.tpot.fit(self.train_df, target_key)
        test_w_predictions = self.tpot.predict(self.test_df, target_key)
        y_true = test_w_predictions[target_key]
        y_test = test_w_predictions[target_key + " predicted"]
        self.assertGreater(r2_score(y_true, y_test), 0.75)

    def test_classification(self):
        max_kvrh = 50
        classifier_key = "K_VRH > {}?".format(max_kvrh)
        train_df = self.train_df.rename(columns={"K_VRH": classifier_key})
        test_df = self.test_df.rename(columns={"K_VRH": classifier_key})
        train_df[classifier_key] = train_df[classifier_key] > max_kvrh
        test_df[classifier_key] = test_df[classifier_key] > max_kvrh
        self.tpot.fit(train_df, classifier_key)
        test_w_predictions = self.tpot.predict(test_df, classifier_key)
        y_true = test_w_predictions[classifier_key]
        y_test = test_w_predictions[classifier_key + " predicted"]
        self.assertGreater(f1_score(y_true, y_test), 0.75)

    def test_training_only(self):
        target_key = "K_VRH"
        train_w_predictions = self.tpot.fit_transform(self.train_df, target_key)
        y_true = train_w_predictions[target_key]
        y_test = train_w_predictions[target_key + " predicted"]
        self.assertTrue(r2_score(y_true, y_test) > 0.85)

    def test_feature_mismatching(self):
        target_key = "K_VRH"
        df1 = self.train_df
        df2 = self.test_df.rename(columns={'mean X': "some other feature"})
        self.tpot.fit(df1, target_key)
        with self.assertRaises(AutomatminerError):
            self.tpot.predict(df2, target_key)


class TestSinglePipelineAdaptor(unittest.TestCase):
    def setUp(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_csv(basedir + "/mini_automl_df.csv", index_col=0)
        self.train_df = df.copy(deep=True).iloc[:450]
        self.test_df = df.copy(deep=True).iloc[451:]

    def test_Pipeline(self):
        modelr = Pipeline([('scaler', StandardScaler()),
                           ('rfr', RandomForestRegressor())])
        modelc = Pipeline([('scaler', StandardScaler()),
                           ('rfr', RandomForestClassifier())])
        learner = SinglePipelineAdaptor(regressor=modelr, classifier=modelc)
        target_key = "K_VRH"
        learner.fit(self.train_df, target_key)
        test_w_predictions = learner.predict(self.test_df, target_key)
        y_true = test_w_predictions[target_key]
        y_test = test_w_predictions[target_key + " predicted"]
        print(r2_score(y_true, y_test))
        self.assertTrue(r2_score(y_true, y_test) > 0.75)

    def test_BaseEstimator(self):
        learner = SinglePipelineAdaptor(regressor=RandomForestRegressor(),
                                        classifier=RandomForestClassifier())
        target_key = "K_VRH"
        learner.fit(self.train_df, target_key)
        test_w_predictions = learner.predict(self.test_df, target_key)
        y_true = test_w_predictions[target_key]
        y_test = test_w_predictions[target_key + " predicted"]
        self.assertGreater(r2_score(y_true, y_test), 0.75)

    def test_feature_mismatching(self):
        learner = SinglePipelineAdaptor(regressor=RandomForestRegressor(),
                                        classifier=RandomForestClassifier())
        target_key = "K_VRH"
        df1 = self.train_df
        df2 = self.test_df.rename(columns={'mean X': "some other feature"})
        learner.fit(df1, target_key)
        with self.assertRaises(AutomatminerError):
            learner.predict(df2, target_key)

    def test_BaseEstimator_classification(self):
        learner = SinglePipelineAdaptor(regressor=RandomForestRegressor(),
                                        classifier=RandomForestClassifier())
        # Prepare dataset for classification
        train_df = self.train_df
        test_df = self.test_df
        for df in [train_df, test_df]:
            df["K_VRH"] = df["K_VRH"] > 150
            df.rename(columns={"K_VRH": "K_VRH > 50"}, inplace=True)

        print(train_df["K_VRH > 50"].value_counts())
        print(test_df["K_VRH > 50"].value_counts())

        target_key = "K_VRH > 50"
        learner.fit(self.train_df, target_key)
        test_w_predictions = learner.predict(self.test_df, target_key)
        y_true = test_w_predictions[target_key]
        y_test = test_w_predictions[target_key + " predicted"]
        print(f1_score(y_true, y_test))
        self.assertGreater(f1_score(y_true, y_test), 0.65)


if __name__ == '__main__':
    unittest.main()
