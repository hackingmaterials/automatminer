"""
The highest level classes for pipelines.
"""
import os
import pickle
from typing import Dict

import pandas as pd
from automatminer import __name__ as amm_name
from automatminer.base import DFTransformer
from automatminer.presets import get_preset_config
from automatminer.utils.log import initialize_logger
from automatminer.utils.ml import regression_or_classification
from automatminer.utils.pkg import (
    AutomatminerError,
    VersionError,
    check_fitted,
    get_version,
    return_attrs_recursively,
    save_dict_to_file,
    set_fitted,
)

logger = initialize_logger(logger_name=amm_name)


class MatPipe(DFTransformer):
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

    The pipeline is transferrable. It can be fit on one dataset and used
    to predict the properties of another. The entire pipeline and
    all constituent objects can be summarized (via "summarize") or inspected
    (via "inspect") in human readable formats.

    ----------------------------------------------------------------------------
    Note: This pipeline should function the same regardless of which
    "component" classes it is made out of. E.g. the steps for each method should
    remain the same whether using the TPOTAdaptor class as the learner or
    using an SinglePipelineAdaptor class as the learner. To use a preset config,
    use MatPipe.from_preset(preset)
    ----------------------------------------------------------------------------

    Examples:
        # A benchmarking experiment, where all property values are known
        pipe = MatPipe()
        test_predictions = pipe.benchmark(df, "target_property")

        # Creating a pipe with data containing known properties, then predicting
        # on new materials
        pipe = MatPipe()
        pipe.fit(training_df, "target_property")
        predictions = pipe.predict(unknown_df)

        # Getting a MatPipe from preset
        pipe = MatPipe.from_preset("debug")

    Args:
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
        version (str): The automatminer version used for serialization and
            deserialization.

        The following attributes are set during fitting. Each has their own set
        of attributes which defines more specifically how the pipeline works.

        pre_fit_df (pd.DataFrame): The dataframe on which the pipeline was fit.
        post_fit_df (pd.DataFrame): The dataframe transformed into the ML-ready
            form.
        ml_type (str): Specifies regression or classification.
        target (str): The name of the column where target values are held.
    """

    def __init__(
        self, autofeaturizer=None, cleaner=None, reducer=None, learner=None
    ):
        transformers = [autofeaturizer, cleaner, reducer, learner]
        if not all(transformers):
            if any(transformers):
                raise AutomatminerError(
                    "Please specify all dataframe"
                    "transformers (autofeaturizer, learner,"
                    "reducer, and cleaner), or none (to use"
                    "default)."
                )
            else:
                config = get_preset_config("express")
                autofeaturizer = config["autofeaturizer"]
                cleaner = config["cleaner"]
                reducer = config["reducer"]
                learner = config["learner"]

        self.autofeaturizer = autofeaturizer
        self.cleaner = cleaner
        self.reducer = reducer
        self.learner = learner
        self.pre_fit_df = None
        self.post_fit_df = None
        self.ml_type = None
        self.target = None
        self.version = get_version()
        super(MatPipe, self).__init__()

    @staticmethod
    def from_preset(preset: str = "express", **powerups):
        """
        Get a preset MatPipe from a string using
        automatminer.presets.get_preset_config

        See get_preeset_config for more inspect.

        Args:
            preset (str): The preset configuration to use.
                Current presets are:
                 - production
                 - express (recommended for most problems)
                 - express_single (no AutoML, XGBoost only)
                 - heavy
                 - debug
                 - debug_single (no AutoML, XGBoost only)
            powerups (kwargs): General upgrades/changes to apply.
                Current powerups are:
                 - cache_src (str): The cache source if you want to save
                    features.
                 - n_jobs (int): The number of parallel process to use when
                    running.
        """
        config = get_preset_config(preset, **powerups)
        return MatPipe(**config)

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
        logger.info("Problem type is: {}".format(self.ml_type))

        # Fit transformers on training data
        logger.info("Fitting MatPipe pipeline to data.")
        df = self.autofeaturizer.fit_transform(df, target)
        df = self.cleaner.fit_transform(df, target)
        df = self.reducer.fit_transform(df, target)
        self.learner.fit(df, target)
        logger.info("MatPipe successfully fit.")
        self.post_fit_df = df
        self.target = target
        return self

    def transform(self, df, **transform_kwargs):
        return self.predict(df, **transform_kwargs)

    @check_fitted
    def predict(self, df, ignore=None, output_col=None):
        """
        Predict a target property of a set of materials.

        The dataframe should have the same target property as the dataframe
        used for fitting. The dataframe should also have the same materials
        property types at the dataframe used for fitting (e.g., if you fit a
        matpipe to a df containing composition, your prediction df should have
        a column for composition). If you used custom features, make sure those
        are included in your prediction df as well.

        Args:
            df (pandas.DataFrame): Pipe will be fit to this dataframe.
            ignore ([str], None): Select which columns to ignore.
                These columns will not be used for learning/prediction, but will
                simply be appended back to the predicted df at the end of
                prediction REGARDLESS of the pipeline configuration.

                This will not stop samples from being dropped. If
                columns not present in the fitting are not ignored, they will
                be automatically dropped. Similarly, if the AutoFeaturizer
                is not configured to preserve inputs and they are not ignored,
                they will be automatically dropped. Ignoring columns supercedes
                all inner operations.

                Select columns using:
                - [str]: String names of columns to ignore.
                - None: input columns will be automatically dropped if they are
                    inputs. User defined features will be preserved if usable
                    as ML input.

        Returns:
            (pandas.DataFrame): The dataframe with target property predictions.
        """
        if ignore:
            logger.warning(
                f"MatPipe will ignore and append (after prediction) the "
                f"following columns: \n{ignore}"
            )
            ignore_df = df[list(ignore)]
            df = df.drop(columns=ignore_df)
        else:
            ignore_df = pd.DataFrame()

        logger.info("Beginning MatPipe prediction using fitted pipeline.")
        df = self.autofeaturizer.transform(
            df, self.target, prevent_cache_overwrite=True
        )
        df = self.cleaner.transform(df, self.target)
        df = self.reducer.transform(df, self.target)
        predictions = self.learner.predict(df, self.target, output_col=output_col)
        logger.info("MatPipe prediction completed.")
        merged_df = predictions.join(ignore_df, how="left")
        return merged_df

    @set_fitted
    def benchmark(
        self, df, target, kfold, fold_subset=None, cache=False, ignore=None
    ):
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
            ignore ([str], None): Ignore columns during prediction for each
                outer fold. See .predict --> ignore argument for more details.

        Returns:
            results ([pd.DataFrame]): Dataframes containing each fold's
                known targets, as well as their independently predicted targets.
        """
        cache_src = self.autofeaturizer.cache_src
        if cache_src and cache:
            if os.path.exists(cache_src):
                logger.warning(
                    "Cache src {} already found! Ensure this featurized data "
                    "matches the df being benchmarked.".format(cache_src)
                )
            logger.warning("Running pre-featurization for caching.")
            self.autofeaturizer.fit_transform(df, target)
        elif cache_src and not cache:
            raise AutomatminerError(
                "Caching was enabled in AutoFeaturizer but not in benchmark. "
                "Either disable caching in AutoFeaturizer or enable it by "
                "passing cache=True to benchmark."
            )
        elif cache and not cache_src:
            raise AutomatminerError(
                "MatPipe cache is enabled, but no cache_src was defined in "
                "autofeaturizer. Pass the cache_src argument to AutoFeaturizer "
                "or use the cache_src get_preset_config powerup."
            )
        else:
            logger.debug("No caching being used in AutoFeaturizer or " "benchmark.")

        if not fold_subset:
            fold_subset = list(range(kfold.n_splits))

        logger.warning("Beginning benchmark.")
        results = []
        fold = 0
        for _, test_ix in kfold.split(X=df, y=df[target]):
            if fold in fold_subset:
                logger.info("Training on fold index {}".format(fold))
                # Split, identify, and randomize test set
                test = df.iloc[test_ix].sample(frac=1)
                train = df[~df.index.isin(test.index)].sample(frac=1)
                self.fit(train, target)
                logger.info("Predicting fold index {}".format(fold))
                test = self.predict(test, ignore=ignore)
                results.append(test)
            fold += 1
        return results

    @check_fitted
    def inspect(self, filename=None) -> Dict[str, str]:
        """
        Get all details of the pipeline in human-readable format.

        For a shorter human-readable representation, use MatPipe.summarize().

        Args:
            filename (str): An optional  '.txt', '.yaml', '.yml', or '.json'
                filename to use for saving the pipeline inspect.

        Returns:
            (dict): A dict of strings in human readable format. Contains a
                string representation of every object and user parameter.
        """
        attrs = return_attrs_recursively(self)
        if filename:
            save_dict_to_file(attrs, filename)
        return attrs

    @check_fitted
    def summarize(self, filename=None) -> Dict[str, str]:
        """
        Get an executive summary of the most important parts of the pipeline.
        Useful for understanding the pipeline at a high level.

        For a more detailed human-readable representation, use MatPipe.inspect.

        Args:
            filename (str): An optional  '.txt', '.yaml', '.yml', or '.json'
                filename to use for saving the pipeline summarize.

        Returns:
            (dict): A dict of strings in human readable format. Contains a
                string representation of every object and user parameter.
        """
        cleaner_attrs = [
            "encoder",
            "feature_na_method",
            "na_method_fit",
            "na_method_transform",
            "drop_na_targets",
        ]
        cleaner_data = {
            attr: str(getattr(self.cleaner, attr)) for attr in cleaner_attrs
        }

        reducer_attrs = ["reducers", "reducer_params"]
        reducer_data = {
            attr: str(getattr(self.reducer, attr)) for attr in reducer_attrs
        }

        attrs = {
            "featurizers": self.autofeaturizer.featurizers,
            "ml_model": str(self.learner.best_pipeline),
            "feature_reduction": reducer_data,
            "data_cleaning": cleaner_data,
            "features": self.learner.features,
        }
        if filename:
            save_dict_to_file(attrs, filename)
        return attrs

    @check_fitted
    def save(self, filename="mat.pipe"):
        """
        Pickles and saves a pipeline. Direct pickling will not work as some
        AutoML backends can't serialize.

        Note that the saved object should only be used for prediction, and
        should not be refit. The AutoML backend is removed and replaced with
        the best pipeline, so the other evaluated pipelines may not be saved!

        Args:
            filename (str): The filename the pipe should be saved as.

        Returns:
            None
        """
        self.learner.serialize()
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        # Reassign live memory objects for further use in this object
        self.learner.deserialize()

    @staticmethod
    def load(filename, supress_version_mismatch=False):
        """
        Loads a MatPipe that was saved.

        Args:
            filename (str): The pickled MatPipe object (should have been saved
                using save).
            supress_version_mismatch (bool): If False, throws an error when
                there is a version mismatch between a serialized MatPipe and the
                current Automatminer version. If True, suppresses this error.

        Returns:
            pipe (MatPipe): A MatPipe object.
        """
        with open(filename, "rb") as f:
            pipe = pickle.load(f)

        if pipe.version != get_version() and not supress_version_mismatch:
            raise VersionError("Version mismatch")

        logger.info("Loaded MatPipe from file {}.".format(filename))
        if hasattr(pipe.learner, "from_serialized"):
            if pipe.learner.from_serialized:
                logger.warning(
                    "Only use this model to make predictions (do not "
                    "retrain!). Backend was serialzed as only the top model, "
                    "not the full automl backend. "
                )
        return pipe
