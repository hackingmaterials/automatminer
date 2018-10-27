"""
This file should house the highest-level methods of matbench, including the
classes which take in a dataframe and target key and product useful analytics.
"""
import datetime
import logging

from sklearn.model_selection import train_test_split

from matbench.featurization.core import Featurization
from matbench.preprocessing.core import Preprocesser
from matbench.automl.tpot_utils import TpotAutoml

# todo: this is a WIP - AD


class PredictionPipeline:
    """
    Accepts a dataset and runs a matbench pipeline on it.
    Also contains all data needed for generating reports and understanding
    a matbench pipeline that was executed.
    """
    def __init__(self, time_limit=None, persistence_lvl=2, name=None):
        self.time_limit = time_limit
        self.persistence_lvl = persistence_lvl
        self.pipetype = None
        self.name = name

        if self.persistence_lvl > 1:
            now = datetime.datetime.now().isoformat()
            input_df_name = "input_df_{}.json".format(now)

        self.best_model = None

    def fit(self, df, target_key, test_frac=None):
        """
        Benchmarks an AutoML pipeline including featurization, feature
        reduction, and model selection.

        Returns:
            A MatbenchPipeline object
        """
        self.pipetype = "benchmark"

        # TODO: featurizer selection should go here
        #self.featurizers = FeaturizerSelection()

        f = Featurization()
        df = f.auto_featurize(df)

        if self.persistence_lvl > 1:
            df.to_json("{}_featurized_df_{}.json".format(self.name, datetime.datetime.now().isoformat()))

        p = Preprocesser()
        df = p.preprocess(df, target_key, scale=True, max_na_frac=0.01)

        if self.persistence_lvl > 1:
            df.to_json("{}_preprocessed_df_{}.json".format(self.name, datetime.datetime.now().isoformat()))

        #todo: need to account for when validation fraction is zero

        tpot = TpotAutoml(mode="regression",
                          max_time_mins=360,
                          scoring='neg_mean_absolute_error',
                          feature_names=df.drop(target_key, axis=1).columns,
                          n_jobs=8,
                          verbosity=2)

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(target_key, axis=1).values, df[target_key],
            test_size=test_frac)

        tpot.fit(X_train, y_train)
        self.validation_score = tpot.score(X_test, y_test)
        self.top_models = tpot.get_top_models(return_scores=True)

        # todo: should save tpot model here
        # self.tpot_top_predictor = ??
        return self

    def predict(self, predict_df):
        """
        Using pipeline created with fit, makes predictions with an automated pipeline.

        Args:
            predict_df: The dataframe containing

        Returns:
            A MatbenchPipeline object

        """
        self.pipetype = "prediction"
        pass


if __name__ == "__main__":
    from matbench.data.load import load_jarvis_dft
    df = load_jarvis_dft()["structure", "e_form"]
    from sklearn.linear_model import LinearRegression
    # mbp = PredictionPipeline(df, target_key="e_form")