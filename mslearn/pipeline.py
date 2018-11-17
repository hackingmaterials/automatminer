"""
The highest level classes for pipelines.
"""
from collections import Iterable
from pprint import pformat
import pickle

import numpy as np

from mslearn.base import LoggableMixin, DataframeTransformer
from mslearn.featurization import AutoFeaturizer
from mslearn.preprocessing import DataCleaner, FeatureReducer
from mslearn.automl.adaptors import TPOTAdaptor
from mslearn.utils.ml_tools import regression_or_classification
from mslearn.utils.package_tools import check_fitted, set_fitted, \
    return_attrs_recursively


performance_config = {}
default_config = {"learner": TPOTAdaptor(max_time_mins=120),
                  "reducer": FeatureReducer(),
                  "autofeaturizer": AutoFeaturizer(),
                  "cleaner": DataCleaner()}
fast_config = {"learner": TPOTAdaptor(max_time_mins=30, population_size=50),
               "reducer": FeatureReducer(reducers=('corr', 'tree')),
               "autofeaturizer": AutoFeaturizer(),
               "cleaner": DataCleaner()}
debug_config = {"learner": TPOTAdaptor(max_time_mins=1, population_size=10),
                "reducer": FeatureReducer(reducers=('corr',)),
                "autofeaturizer": AutoFeaturizer(),
                "cleaner": DataCleaner()}


class MatPipe(DataframeTransformer, LoggableMixin):
    """
    Establish an ML pipeline for transforming compositions, structures,
    bandstructures, and DOS objects into machine-learned properties.

    The pipeline includes:
        - featurization
        - ml-preprocessing
        - automl model fitting and creation

    If you are using MatPipe for benchmarking, use the "benchmark" method.

    If you have some training data and want to use MatPipe for production
    predictions (e.g., predicting material properties for which you have
    no data) use "fit" and "predict".

    The pipeline is transferable. So it can be fit on one dataset and used
    to predict the properties of another. Furthermore, the entire pipeline and
    all constituent objects can be summarized in text with "digest".

    ----------------------------------------------------------------------------
    Note: This pipeline should function the same regardless of which
    "component" classes it is made out of. E.g., he steps for each method should
    remain the same whether using the TPOTAdaptor class as the learner or
    using an AutoKerasAdaptor class as the learner.
    ----------------------------------------------------------------------------

    Examples:
        # A benchmarking experiment, where all property values are known
        pipe = MatPipe()
        test_predictions = pipe.benchmark(df, "target_property")

        # Creating a pipe with data containing known properties, then predicting
        # on new materials
        pipe = MatPipe()
        pipe.fit(training_df, "target_property")
        predictions = pipe.predict(unknown_df, "target_property")

    Args:
        persistence_lvl (int): Persistence level of 0 saves nothing. 1 saves
            intermediate dataframes and final dataframes. 2 saves all dataframes
            and all objects used to create the pipeline, and auto-saves a digest
        autofeaturizer (AutoFeaturizer): The autofeaturizer object used to
            automatically decorate the dataframe with descriptors.
        cleaner (DataCleaner): The data cleaner object used to get a
            featurized dataframe in ml-ready form.
        reducer (FeatureReducer): The feature reducer object used to
            select the best features from a "clean" dataframe.
        learner (AutoMLAdaptor): The auto ml adaptor object used to
            actually run a auto-ml pipeline on the clean, reduced, featurized
            dataframe.

    Attributes:
        The following attributes are set during fitting. Each has their own set
        of attributes which defines more specifically how the pipeline works.

        is_fit (bool): If True, the matpipe is fit. The matpipe should be
            fit before being used to predict data.
    """

    def __init__(self, logger=True, autofeaturizer=None,
                 cleaner=None, reducer=None, learner=None):

        self._logger = self.get_logger(logger)
        self.autofeaturizer = autofeaturizer if autofeaturizer else \
            default_config['autofeaturizer']
        self.cleaner = cleaner if cleaner else default_config["cleaner"]
        self.reducer = reducer if reducer else default_config["reducer"]
        self.learner = learner if learner else default_config["learner"]

        self.autofeaturizer._logger = self.get_logger(logger)
        self.cleaner._logger = self.get_logger(logger)
        self.reducer._logger = self.get_logger(logger)
        self.learner._logger = self.get_logger(logger)

        self.pre_fit_df = None
        self.post_fit_df = None
        self.is_fit = False
        self.ml_type = self.learner.mode

    @set_fitted
    def fit(self, df, target):
        """
        Fit a matpipe to a dataframe. Once fit, can be used to predict out of
        sample data.

        The dataframe should contain columns having some materials data:
            - compositions
            - structures
            - bandstructures
            - density of states
            - user-defined features

        Any combination of these data is ok.

        Args:
            df (pandas.DataFrame): Pipe will be fit to this dataframe.
            target (str): The column in the dataframe containing the target
                property of interest

        Returns:
            MatPipe (self)

        """
        self.pre_fit_df = df
        self.ml_type = regression_or_classification(df[target])

        # Fit transformers on training data
        self.logger.info("Fitting MatPipe pipeline to data.")
        df = self.autofeaturizer.fit_transform(df, target)
        df = self.cleaner.fit_transform(df, target)
        df = self.reducer.fit_transform(df, target)
        self.learner.fit(df, target)
        self.logger.info("MatPipe successfully fit.")
        self.post_fit_df = df
        return self

    @check_fitted
    def predict(self, df, target):
        """
        Predict a target property of a set of materials.

        The dataframe should have the same target property as the dataframe
        used for fitting. The dataframe should also have the same materials
        property types at the dataframe used for fitting (e.g., if you fit a
        matpipe to a df containing composition, your prediction df should have
        a column for composition).

        Args:
            df (pandas.DataFrame): Pipe will be fit to this dataframe.
            target (str): The column in the dataframe containing the target
                property of interest

        Returns:
            (pandas.DataFrame): The dataframe with target property predictions.
        """
        self.logger.info("Beginning MatPipe prediction using fitted pipeline.")
        df = self.autofeaturizer.transform(df, target)
        df = self.cleaner.transform(df, target)
        df = self.reducer.transform(df, target)
        predictions = self.learner.predict(df, target)
        self.logger.info("MatPipe prediction completed.")
        return predictions

    @set_fitted
    def benchmark(self, df, target, test_spec=0.2):
        """
        If the target property is known for all data, perform an ML benchmark
        using MatPipe. Used for getting an idea of how well AutoML can predict
        a certain target property.

        This method featurizes and cleans the entire dataframe, then splits
        the data for training and testing. FeatureReducer and TPOT models are
        fit on the training data. Finally, these fitted models are used to
        predict the properties of the test df. This scheme allows for rigorous
        ML model evaluation, as the feature selection and model fitting are
        determined without any knowledge of the validation/test set.

        To use a random validation set for model validation, pass in a nonzero
        validation fraction as a float. The returned df will have the validation
        predictions.

        To use a CV-only validation, use a validation frac. of 0. The original
        df will be returned having predictions made on all training data. This
        should ONLY be used to evaluate the training error!

        To use a fixed validation set, pass in the index (must be .iloc-able in
        pandas) as the validation argument.

        Whether using CV-only or validation, both will create CV information
        in the MatPipe.learner.best_models variable.

        Args:
            df (pandas.DataFrame): The dataframe for benchmarking. Must contain
            target (str): The column name to use as the ml target property.
            test_spec (float or listlike): Specifies how to do test/evaluation.
                If the test spec is a float, it specifies the fraction of the
                dataframe to be randomly selected for testing (must be a
                number between 0-1). test_spec=0 means a CV-only validation.
                If test_spec is a list/ndarray, it is the iloc indexes of the
                dataframe to use for testing. This option is useful if you
                are comparing multiple techniques and want to use the same
                test or validation fraction across benchmarks.

        Returns:
            testdf (pandas.DataFrame): A dataframe containing original test data
                and predicted data. If test_spec is set to 0, test df
                will contain PREDICTIONS MADE ON TRAINING DATA. This should be
                used to evaluate the training error only!

        """
        # Fit transformers on all data
        self.logger.info("Featurizing and cleaning {} samples from the entire"
                         " dataframe.".format(df.shape[0]))
        df = self.autofeaturizer.fit_transform(df, target)
        df = self.cleaner.fit_transform(df, target)

        # Split data for steps where combined transform could otherwise over-fit
        # or leak data from validation set into training set.
        if isinstance(test_spec, Iterable):
            traindf = df.iloc[~np.asarray(test_spec)]
            testdf = df.iloc[np.asarray(test_spec)]
        else:
            testdf, traindf = np.split(df.sample(frac=1),
                                       [int(test_spec * len(df))])
        self.logger.info("Dataframe split into training and testing fractions"
                         " having {} and {} samples.".format(traindf.shape[0],
                                                             testdf.shape[0]))

        # Use transformers on separate training and testing dfs
        self.logger.info("Performing feature reduction and model selection on "
                         "the {}-sample training set.".format(traindf.shape[0]))
        traindf = self.reducer.fit_transform(traindf, target)
        self.learner.fit(traindf, target)

        if isinstance(test_spec, Iterable) or test_spec != 0:
            self.logger.info(
                "Using pipe fitted on training data to predict target {} on "
                "{}-sample validation dataset".format(target, testdf.shape[0]))
            testdf = self.reducer.transform(testdf, target)
            testdf = self.learner.predict(testdf, target)
            return testdf
        else:
            self.logger.warning("Validation fraction set to zero. Using "
                                "cross-validation-only benchmarking...")
            traindf = self.learner.predict(traindf, target)
            return traindf

    @check_fitted
    def digest(self, filename=None):
        """
        Save a text digest (summary) of the fitted pipeline. Similar to the log
        but contains more detail in a structured format.

        Args:
            filename (str): The filename.
            fmt (str): The format to save the pipeline in. Valid choices are
                "json", "txt".

        Returns:
            digeststr (str): The formatted pipeline digest.
        """
        digeststr = pformat(return_attrs_recursively(self))
        if filename:
            with open(filename, "w") as f:
                f.write(digeststr)
        return digeststr

    @check_fitted
    def save(self, filename="matpipe.p"):
        """
        Pickles and saves a pipeline. Direct pickling will not work as some
        AutoML backends can't serialize.

        Note that the saved object should only be used for prediction, and
        should not be refit. The automl backend is removed and replaced with
        the best pipeline, so the other evaluated pipelines may not be saved!

        Args:
            filename (str): The filename the pipe should be saved as.

        Returns:
            None
        """
        self.learner._backend = self.learner.backend.fitted_pipeline_
        for obj in [self, self.learner, self.reducer, self.cleaner,
                    self.autofeaturizer]:
            obj._logger = None

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename, logger=True):
        """
        Loads a matpipe that was saved.

        Args:
            filename (str): The pickled matpipe object (should have been saved
                using save).
            logger (bool or logging.Logger): The logger to use for the loaded
                matpipe.

        Returns:
            pipe (MatPipe): A MatPipe object.
        """
        with open(filename, 'rb') as f:
            pipe = pickle.load(f)

        for obj in [pipe, pipe.learner, pipe.reducer, pipe.cleaner,
                    pipe.autofeaturizer]:
            obj._logger = cls.get_logger(logger)

        pipe.logger.info("Loaded MatPipe from file {}.".format(filename))
        pipe.logger.warning("Only use this model to make predictions (do not "
                            "retrain!). Backend was serialzed as only the top "
                            "model, not the full automl backend. ")
        return pipe


def MatPipePerform(**kwargs):
    return MatPipe(**kwargs, **performance_config)


def MatPipeFast(**kwargs):
    return MatPipe(**kwargs, **fast_config)


from fireworks import FireTaskBase, Firework, explicit_serialize, LaunchPad


if __name__ == "__main__":
    # from sklearn.metrics import mean_squared_error
    # from matminer.datasets.dataset_retrieval import load_dataset
    #
    # hugedf = load_dataset("elastic_tensor_2015").rename(
    #     columns={"formula": "composition"})[["composition", "K_VRH"]]
    #
    # validation_ix = [1, 2, 3, 4, 5, 7, 12]
    # df = hugedf.iloc[:100]
    # df2 = hugedf.iloc[101:150]
    # target = "K_VRH"
    #
    # mp = MatPipe(**debug_config)
    # df = mp.benchmark(df, target, test_spec=0.25)

    lp = LaunchPad(name="automatminer")
    lp.add_wf(Firework(CustomTask()))

