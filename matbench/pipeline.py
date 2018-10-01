"""
This file should house the highest-level methods of matbench, including the
classes which take in a dataframe and target key and product useful analytics.
"""
import datetime
import logging

from matbench.featurization.core import Featurize
from matbench.utils.utils import setup_custom_logger


class PredictionPipeline:
    """
    Accepts a dataset and runs a matbench pipeline on it.
    Also contains all data needed for generating reports and understanding
    a matbench pipeline that was executed.

    Args:
        df:
        target_key:
        time_limit:
        test_frac:
        aux_keys:
        persistence_lvl (int): Determines which files will be saved. 0 means
            nothing will be saved. 1 means final model will be saved. 2 means
            all intermediate dataframes will be saved.
    """

    def __init__(self, df, target_key, time_limit=None,
                 test_frac=None, aux_keys=None, logpath=".",
                 loglvl=logging.INFO, persistence_lvl=2):
        self.train_df = prescreen_df(df)
        self.target_key = target_key
        self.time_limit = time_limit
        self.test_frac = test_frac
        self.aux_keys = aux_keys
        self.persistence_lvl = persistence_lvl
        self.logger = setup_custom_logger(filepath=logpath, level=loglvl)
        self.pipetype = None

        if self.persistence_lvl > 1:
            now = datetime.datetime.now().isoformat()
            input_df_name = "input_df_{}.json".format(now)
            self.logger.log("Saving input df as {}".format(input_df_name))
            self.train_df.to_json()

    def benchmark(self):
        """
        Benchmarks an AutoML pipeline including featurization, feature
        reduction, and model selection.

        Returns:
            A MatbenchPipeline object
        """
        self.logger.log("Beginning benchmarking.\n")
        self.pipetype = "benchmark"

        # TODO: meta learning should go here

        f = Featurize()
        df = f.auto_featurize(self.train_df)

        if self.persistence_lvl > 1:
            now = datetime.datetime.now().isoformat()
            df.to_json("featurized_df_{}.json".format(now))

        pass

    def predict(self, predict_df):
        """
        Using the data in df, makes predictions with an automated pipeline.

        Args:
            predict_df: The dataframe containing

        Returns:
            A MatbenchPipeline object

        """
        self.pipetype = "prediction"


def prescreen_df(df, must_include_cols=None, cant_include_cols=None):
    return df
