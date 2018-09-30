"""
This file should house the highest-level methods of matbench, including the
classes which take in a dataframe and target key and product useful analytics.
"""


class PredictionEngine:
    """
    Benchmark takes in a dataset, runs a matbench pipeline on it, and returns a
    model. Post-pipeline, it summarizes (or evaluates w/ validation set) the
    performance of the model and creates a webpage w/ the results.
    """

    def __init__(self, df, target_key, time_limit=None,
                 test_frac=None, aux_keys=None, persistence_lvl=2):
        """

        Args:
            df: The dataframe to be featurized
            target_key:
            aux_keys:
            save_intermediate:
        """
        self.train_df = df
        self.pipe = MatBenchPipeline(persistence_lvl)
        self.target_key = target_key
        self.time_limit = time_limit
        self.test_frac = test_frac
        self.aux_keys = aux_keys
        pass

    def benchmark(self):
        """
        Checks to make sure the dataframe is ready for featurization.
            - Is the target key in the dataframe?
            - Are auxiliary keys in the dataframe?
            - Is df actually a dataframe?
            - Are featurization keys available in the dataframe?
            - Are the columns for composition, formula, structure, the correct
                types?

        Returns:

        """
        pass

    def predict(self, predict_df):
        pass




class MatBenchPipeline:
    """
    Holds all high-level data for
    """
    pass


