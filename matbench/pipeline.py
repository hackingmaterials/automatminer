"""
This file should house the highest-level methods of matbench, including the
classes which take in a dataframe and target key and product useful analytics.
"""
import datetime
import logging

from matbench.featurization.core import Featurization
from matbench.preprocessing.core import Preprocessing
from matbench.utils.utils import setup_custom_logger
from matbench.automl.tpot_utils import TpotAutoml
from sklearn.model_selection import train_test_split


class PredictionPipeline:
    """
    Accepts a dataset and runs a matbench pipeline on it.
    Also contains all data needed for generating reports and understanding
    a matbench pipeline that was executed.
    """
    def __init__(self, df, target_key, time_limit=None,
                 test_frac=0.1, aux_keys=None, logpath=".",
                 loglvl=logging.INFO, persistence_lvl=2, name=None):
        self.train_df = prescreen_df(df)
        self.target_key = target_key
        self.time_limit = time_limit
        self.test_frac = test_frac
        self.aux_keys = aux_keys
        self.persistence_lvl = persistence_lvl
        self.logger = setup_custom_logger(filepath=logpath, level=loglvl)
        self.pipetype = None
        self.name = name

        if self.persistence_lvl > 1:
            now = datetime.datetime.now().isoformat()
            input_df_name = "input_df_{}.json".format(now)
            self.logger.log("Saving input df as {}".format(input_df_name))
            self.train_df.to_json()

        self.best_model

    def fit(self):
        """
        Benchmarks an AutoML pipeline including featurization, feature
        reduction, and model selection.

        Returns:
            A MatbenchPipeline object
        """
        self.logger.log("Beginning benchmarking.\n")
        self.pipetype = "benchmark"

        # TODO: featurizer selection should go here

        f = Featurization()
        df = f.auto_featurize(self.train_df)

        if self.persistence_lvl > 1:
            df.to_json("{}_featurized_df_{}.json".format(self.name, datetime.datetime.now().isoformat()))

        p = Preprocessing()
        df = p.preprocess(df, "e_form", scale=True, max_na_frac=0.01)

        if self.persistence_lvl > 1:
            df.to_json("{}_preprocessed_df_{}.json".format(self.name, datetime.datetime.now().isoformat()))

        #todo: need to account for when validation fraction is zero

        tpot = TpotAutoml(mode="regression",
                          max_time_mins=360,
                          scoring='neg_mean_absolute_error',
                          feature_names=df.drop(self.target_key, axis=1).columns,
                          n_jobs=8,
                          verbosity=2)

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(self.target_key, axis=1).values, df[self.target_key],
            test_size=self.test_frac)

        tpot.fit(X_train, y_train)
        validation_score = tpot.score(X_test, y_test)

        # todo: should save tpot model here
        return {"top_model_scores": tpot.get_top_models(return_scores=True),
                "best_model_validation_score": validation_score}

    def predict(self, predict_df):
        """
        Using the data in df, makes predictions with an automated pipeline.

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