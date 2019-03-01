"""
The highest level classes for pipelines.
"""
import os
import pickle
from pprint import pformat

from automatminer.base import LoggableMixin, DFTransformer
from automatminer.presets import get_preset_config
from automatminer.utils.ml import regression_or_classification
from automatminer.utils.pkg import check_fitted, set_fitted, \
    return_attrs_recursively, AutomatminerError


class MatPipe(DFTransformer, LoggableMixin):
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

    The pipeline is transferrable. So it can be fit on one dataset and used
    to predict the properties of another. Furthermore, the entire pipeline and
    all constituent objects can be summarized in text with "digest".

    ----------------------------------------------------------------------------
    Note: This pipeline should function the same regardless of which
    "component" classes it is made out of. E.g. the steps for each method should
    remain the same whether using the TPOTAdaptor class as the learner or
    using an AutoKerasAdaptor class as the learner. To use a preset config,
    import a config from automatminer.configs and do MatPipe(**config).
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
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default automatminer logger will
            be used. If set to False, then no logging will occur.
        log_level (int): The log level. For example logging.DEBUG or 2.
        autofeaturizer (AutoFeaturizer): The autofeaturizer object used to
            automatically decorate the dataframe with descriptors.
        cleaner (DataCleaner): The data cleaner object used to get a
            featurized dataframe in ml-ready form.
        reducer (FeatureReducer): The feature reducer object used to
            select the best features from a "clean" dataframe.
        learner (DFMLAdaptor): The auto ml adaptor object used to
            actually run a auto-ml pipeline on the clean, reduced, featurized
            dataframe.

    Attributes:
        The following attributes are set during fitting. Each has their own set
        of attributes which defines more specifically how the pipeline works.

        is_fit (bool): If True, the matpipe is fit. The matpipe should be
            fit before being used to predict data.
    """

    def __init__(self, autofeaturizer=None, cleaner=None, reducer=None,
                 learner=None, logger=True, log_level=None):
        transformers = [autofeaturizer, cleaner, reducer, learner]
        if not all(transformers):
            if any(transformers):
                raise AutomatminerError("Please specify all dataframe"
                                        "transformers (autofeaturizer, learner,"
                                        "reducer, and cleaner), or none (to use"
                                        "default).")
            else:
                config = get_preset_config("production")
                autofeaturizer = config["autofeaturizer"]
                cleaner = config["cleaner"]
                reducer = config["reducer"]
                learner = config["learner"]

        self._logger = self.get_logger(logger, level=log_level)
        self.autofeaturizer = autofeaturizer
        self.cleaner = cleaner
        self.reducer = reducer
        self.learner = learner
        self.autofeaturizer._logger = self.get_logger(logger)
        self.cleaner._logger = self.get_logger(logger)
        self.reducer._logger = self.get_logger(logger)
        self.learner._logger = self.get_logger(logger)
        self.pre_fit_df = None
        self.post_fit_df = None
        self.is_fit = False
        self.ml_type = None

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
        self.logger.info("Problem type is: {}".format(self.ml_type))

        # Fit transformers on training data
        self.logger.info("Fitting MatPipe pipeline to data.")
        df = self.autofeaturizer.fit_transform(df, target)
        df = self.cleaner.fit_transform(df, target)
        df = self.reducer.fit_transform(df, target)
        self.learner.fit(df, target)
        self.logger.info("MatPipe successfully fit.")
        self.post_fit_df = df
        return self

    def transform(self, df, target, **transform_kwargs):
        return self.predict(df, target, **transform_kwargs)

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
    def benchmark(self, df, target, kfold, fold_subset=None, cache=False):
        """
        If the target property is known for all data, perform an ML benchmark
        using MatPipe. Used for getting an idea of how well AutoML can predict
        a certain target property.

        MatPipe benchmarks with a nested cross validation, meaning it makes
        k validation/test splits, where all model selection is done on the train
        /validation set (a typical CV). When the model is done validating, it is
        used to predict the previously unseen test set data. This process is
        repeated for each of the k folds, which (1) mitigates the benchmark from
        biasing the model based to the selection of test set and (2) better
        estimates the generalization error than a single validation/test split.

        tl;dr: Put in a dataset and kfold scheme for nested CV, get out the
        predicted test sets.

        Note: MatPipes after benchmarking have been fit on the last fold, not
        the entire dataset. To use your entire dataset for prediction, use the
        MatPipe fit and predict methods.

        Args:
            df (pandas.DataFrame): The dataframe for benchmarking. Must contain
            target (str): The column name to use as the ml target property.
            kfold (sklearn KFold or StratifiedKFold: The cross validation split
                object to use for nested cross validation. Used to index the
                dataframe with .iloc, NOT .loc.
            fold_subset ([int]): A subset of the folds in kfold to evaluate (by
                index). For example, to run only the 3rd train/validation/test
                split of the kfold, set fold_subset = [2]. To use the first and
                fourth, set fold_subset = [0, 3].
            cache (bool): If True, pre-featurizes the entire dataframe
                (including test data!) and caches it before iterating over
                folds. Do NOT use if you are using fittable featurizers whose
                feature labels are based on their input! Doing so may "leak"
                information from the testing set to the training set and will
                over-represent your benchmark. Enabling this for featurizers
                which are not fittable is completely safe. Note that your
                autofeaturizer must have a cache_src defined if allow_caching is
                enabled (do this either through the AutoFeaturizer class or
                using the cache_src argument to get_preset_config.

        Returns:
            results ([pd.DataFrame]): Dataframes containing each fold's
                known targets, as well as their independently predicted targets.
        """
        cache_src = self.autofeaturizer.cache_src
        if cache_src and cache:
            if os.path.exists(cache_src):
                self.logger.warning(
                    "Cache src {} already found! Ensure this featurized data "
                    "matches the df being benchmarked.".format(cache_src))
            self.logger.warning("Running pre-featurization for caching.")
            self.autofeaturizer.fit_transform(df, target)
        elif cache_src and not cache:
            raise AutomatminerError(
                "Caching was enabled in AutoFeaturizer but not in benchmark. "
                "Either disable caching in AutoFeaturizer or enable it by "
                "passing cache=True to benchmark.")
        elif cache and not cache_src:
            raise AutomatminerError(
                "MatPipe cache is enabled, but no cache_src was defined in "
                "autofeaturizer. Pass the cache_src argument to AutoFeaturizer "
                "or use the cache_src get_preset_config powerup.")
        else:
            self.logger.debug("No caching being used in AutoFeaturizer or "
                              "benchmark.")

        if not fold_subset:
            fold_subset = list(range(kfold.n_splits))

        self.logger.warning("Beginning benchmark.")
        results = []
        fold = 0
        for _, test_ix in kfold.split(X=df, y=df[target]):
            if fold in fold_subset:
                self.logger.info("Training on fold index {}".format(fold))
                # Split and identify test set
                test = df.iloc[test_ix]
                train = df[~df.index.isin(test.index)]
                self.fit(train, target)
                self.logger.info("Predicting fold index {}".format(fold))
                test = self.predict(test, target)
                results.append(test)
            fold += 1
        return results

    @check_fitted
    def digest(self, filename=None):
        """
        Save a text digest (summary) of the fitted pipeline. Similar to the log
        but contains more detail in a structured format.

        Args:
            filename (str): The filename.

        Returns:
            digeststr (str): The formatted pipeline digest.
        """
        digeststr = pformat(return_attrs_recursively(self))
        if filename:
            with open(filename, "w") as f:
                f.write(digeststr)
        return digeststr

    @check_fitted
    def save(self, filename="mat.pipe"):
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
        temp_backend = self.learner.backend
        self.learner._backend = self.learner.best_pipeline
        for obj in [self, self.learner, self.reducer, self.cleaner,
                    self.autofeaturizer]:
            obj._logger = None
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        self.learner._backend = temp_backend

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
