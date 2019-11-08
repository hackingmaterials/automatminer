"""
Adaptor classes for using AutoML packages in a Matbench pipeline.

Current adaptor classes are:

    TPOTAdaptor: Uses the backend from the automl project TPOT, which can be
        found at https://github.com/EpistasisLab/tpot
"""
import logging
from collections import OrderedDict

from tpot import TPOTClassifier, TPOTRegressor

from automatminer.automl.config.tpot_configs import (
    TPOT_CLASSIFIER_CONFIG,
    TPOT_REGRESSOR_CONFIG,
)
from automatminer.utils.pkg import set_fitted, check_fitted
from automatminer.utils.ml import is_greater_better, regression_or_classification
from automatminer.utils.log import log_progress, AMM_LOG_FIT_STR
from automatminer.utils.ml import AMM_CLF_NAME, AMM_REG_NAME
from automatminer.automl.base import DFMLAdaptor

__authors__ = [
    "Alex Dunn <ardunn@lbl.gov" "Alireza Faghaninia <alireza.faghaninia@gmail.com>",
    "Qi Wang <wqthu11@gmail.com>",
    "Daniel Dopp <dbdopp@lbl.gov>",
]

_adaptor_tmp_backend = None
logger = logging.getLogger(__name__)


class TPOTAdaptor(DFMLAdaptor):
    """
    A dataframe adaptor for the TPOT classifiers and regressors.

    Args:
        tpot_kwargs: All kwargs accepted by a TPOTRegressor/TPOTClassifier
            or TPOTBase object.

            Note that for example, you can limit the models that TPOT explores
            by setting config_dict directly. For example, if you want to only
            use random forest:
        config_dict = {
            'sklearn.ensemble.RandomForestRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'bootstrap': [True, False]
                },
            }
    Attributes:
        The following unique attributes are set during fitting.

        mode (str): Either AMM_REG_NAME (regression) or AMM_CLF_NAME
            (classification)
        best_models (OrderedDict): The best model names and their scores.
        backend (TPOTBase): The TPOT object interface used for ML training.
        models (OrderedDict): The raw sklearn-style models output by TPOT.

        from_serialized (bool): Whether the backend is loaded from a serialized
            instance. If True, the previous full TPOT data will not be available
            due to pickling problems.
    """

    def __init__(self, **tpot_kwargs):
        tpot_kwargs["cv"] = tpot_kwargs.get("cv", 5)
        tpot_kwargs["n_jobs"] = tpot_kwargs.get("n_jobs", -1)
        tpot_kwargs["verbosity"] = tpot_kwargs.get("verbosity", 3)
        tpot_kwargs["memory"] = tpot_kwargs.get("memory", "auto")

        self.mode = None
        self.tpot_kwargs = tpot_kwargs
        self.models = None
        self.random_state = tpot_kwargs.get("random_state", None)
        self.greater_score_is_better = None

        self._fitted_target = None
        self._backend = None
        self._features = None

        self.from_serialized = False
        self._best_models = None
        super(DFMLAdaptor, self).__init__()

    @log_progress(logger, AMM_LOG_FIT_STR)
    @set_fitted
    def fit(self, df, target, **fit_kwargs):
        """
        Train a TPOTRegressor or TPOTClassifier by fitting on a dataframe.

        Args:
            df (pandas.DataFrame): The df to be used for training.
            target (str): The key used to identify the machine learning target.
            **fit_kwargs: Keyword arguments to be passed to the TPOT backend.
                These arguments must be valid arguments to the TPOTBase class.

        Returns:
            TPOTAdaptor (self)

        """
        # Prevent goofy pandas casting by casting to native
        y = df[target].values
        X = df.drop(columns=target).values

        # Determine learning type based on whether classification or regression
        self.mode = regression_or_classification(df[target])

        mltype_str = "Classifier" if self.mode == AMM_CLF_NAME else "Regressor"
        self.tpot_kwargs["template"] = self.tpot_kwargs.get(
            "template", "Selector-Transformer-{}".format(mltype_str)
        )

        if self.mode == AMM_CLF_NAME:
            self.tpot_kwargs["config_dict"] = self.tpot_kwargs.get(
                "config_dict", TPOT_CLASSIFIER_CONFIG
            )
            if "scoring" not in self.tpot_kwargs:
                self.tpot_kwargs["scoring"] = "balanced_accuracy"
            self._backend = TPOTClassifier(**self.tpot_kwargs)
        elif self.mode == AMM_REG_NAME:
            self.tpot_kwargs["config_dict"] = self.tpot_kwargs.get(
                "config_dict", TPOT_REGRESSOR_CONFIG
            )
            if "scoring" not in self.tpot_kwargs:
                self.tpot_kwargs["scoring"] = "neg_mean_absolute_error"
            self._backend = TPOTRegressor(**self.tpot_kwargs)
        else:
            raise ValueError(
                "Learning type {} not recognized as a valid mode "
                "for {}".format(self.mode, self.__class__.__name__)
            )
        self._features = df.drop(columns=target).columns.tolist()
        self._fitted_target = target
        self._backend = self._backend.fit(X, y, **fit_kwargs)
        return self

    @property
    @check_fitted
    def best_models(self):
        """
        The best models found by TPOT, in order of descending performance.

        If you want a pipeline you can use to make predtions, use the
        best_pipeline.

        Performance is evaluated based on the TPOT scoring. This can be changed
        by passing a "scoring" kwarg into the __init__ method.

        Returns:
            best_models_and_scores (dict): Keys are names of models. Values
                are the best internal cv scores of that model with the
                best hyperparameter combination found.

        """

        if self.from_serialized:
            return self._best_models
        else:
            self.greater_score_is_better = is_greater_better(
                self.backend.scoring_function
            )

            # Get list of evaluated model names, cast to set and back
            # to get unique model names, instantiate ordered model dictionary
            evaluated_models = []
            for key in self.backend.evaluated_individuals_.keys():
                evaluated_models.append(key.split("(")[0])
                # evaluated_models.append(key)

            model_names = list(set(evaluated_models))
            models = OrderedDict({model: [] for model in model_names})

            # This makes a dict of model names mapped to all runs of that model
            for key, val in self.backend.evaluated_individuals_.items():
                models[key.split("(")[0]].append(val)

            # For each base model type sort the runs by best score
            for model_name in model_names:
                models[model_name].sort(
                    key=lambda x: x["internal_cv_score"],
                    reverse=self.greater_score_is_better,
                )

            # Gets a simplified dict of the model to only its best run
            # Sort the best individual models by type to best models overall
            best_models = OrderedDict(
                sorted(
                    {model: models[model][0] for model in models}.items(),
                    key=lambda x: x[1]["internal_cv_score"],
                    reverse=self.greater_score_is_better,
                )
            )

            # Mapping of top models to just their score
            scores = {
                model: best_models[model]["internal_cv_score"]
                for model in best_models
            }

            # Sorted dict of top models just mapped to their top scores
            best_models_and_scores = OrderedDict(
                sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=self.greater_score_is_better,
                )
            )
            self.models = models
            return best_models_and_scores

    @property
    @check_fitted
    def backend(self):
        return self._backend

    @property
    @check_fitted
    def best_pipeline(self):
        if self.from_serialized:
            # The TPOT backend is replaced by the best pipeline.
            return self._backend
        else:
            return self._backend.fitted_pipeline_

    @property
    @check_fitted
    def features(self):
        return self._features

    @property
    @check_fitted
    def fitted_target(self):
        return self._fitted_target

    @check_fitted
    def serialize(self) -> None:
        """
        Avoid TPOT pickling issues. Used by MatPipe during save.

        Returns:
            (self): A deepcopy of this object, with some modifications to make
                it serializable.

        """
        if not self.from_serialized:
            global _adaptor_tmp_backend
            _adaptor_tmp_backend = self._backend
            # Necessary for getting best models post serialization
            self._best_models = self.best_models
            self._backend = self.best_pipeline
            self.from_serialized = True

    @check_fitted
    def deserialize(self) -> None:
        """
        Get the original TPOTAdaptor image back after serializing, with
        (relatively) contained scope.

        Returns:
            None
        """
        if not self.from_serialized:
            global _adaptor_tmp_backend
            self._backend = _adaptor_tmp_backend
            _adaptor_tmp_backend = None
            self.from_serialized = False


class SinglePipelineAdaptor(DFMLAdaptor):
    """
    For running single models or pipelines in a MatPipe pipeline using the same
    syntax as the AutoML adaptors.

    This adaptor should be able to fit into a MatPipe in similar fashion to
    TPOTAdaptor.

    Args:
        regressor (sklearn Pipeline or BaseEstimator-like): The object you want
            to use for machine learning regression. Must implement
            fit/predict/transform methods analagously to BaseEstimator, but does
            not need to be a BaseEstimator or Pipeline.
        classifier (sklearn Pipeline or BaseEstimator-like): The object you want
            to use for machine learning classification.

    Attributes:
        The following unique attributes are set during fitting.

        mode (str): Either AMM_REG_NAME (regression) or AMM_CLF_NAME
            (classification)
    """

    def __init__(self, regressor, classifier):
        self.mode = None
        self._regressor = regressor
        self._classifier = classifier
        self._features = None
        self._fitted_target = None
        self._best_pipeline = None

    @log_progress(logger, AMM_LOG_FIT_STR)
    @set_fitted
    def fit(self, df, target, **fit_kwargs):

        # Determine learning type based on whether classification or regression
        self.mode = regression_or_classification(df[target])

        if self.mode == AMM_CLF_NAME:
            self._best_pipeline = self._classifier
        elif self.mode == AMM_REG_NAME:
            self._best_pipeline = self._regressor
        else:
            raise ValueError(
                "Learning type {} not recognized as a valid mode "
                "for {}".format(self.mode, self.__class__.__name__)
            )

        # Prevent goofy pandas casting by casting to native
        y = df[target].values.tolist()
        X = df.drop(columns=target).values.tolist()
        self._features = df.drop(columns=target).columns.tolist()
        self._fitted_target = target
        self._best_pipeline.fit(X, y)

    @property
    @check_fitted
    def backend(self):
        return self.best_pipeline

    @property
    @check_fitted
    def best_pipeline(self):
        return self._best_pipeline

    @property
    @check_fitted
    def features(self):
        return self._features

    @property
    @check_fitted
    def fitted_target(self):
        return self._fitted_target
