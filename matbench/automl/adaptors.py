"""
Adaptor classes for using AutoML packages in a Matbench pipeline.

Current adaptor classes are:

    TPOTAdaptor: Uses the backend from the automl project TPOT, which can be
        found at https://github.com/EpistasisLab/tpot
"""
from collections import OrderedDict

import numpy as np
from sklearn.exceptions import NotFittedError
from tpot import TPOTClassifier, TPOTRegressor

from matbench.automl.tpot_configs.classifier import classifier_config_dict_mb
from matbench.automl.tpot_configs.regressor import regressor_config_dict_mb
from matbench.utils.utils import is_greater_better, MatbenchError
from matbench.base import AutoMLAdaptor

__authors__ = ['Alex Dunn <ardunn@lbl.gov'
               'Alireza Faghaninia <alireza.faghaninia@gmail.com>',
               'Qi Wang <wqthu11@gmail.com>',
               'Daniel Dopp <dbdopp@lbl.gov>']

_classifier_modes = {'classifier', 'classification', 'classify'}

_regressor_modes = {'regressor', 'regression', 'regress'}


def TPOTAutoML(mode, **kwargs):
    """
    Returns a class wrapped on TPOTClassifier or TPOTRegressor (differentiated
    via mode argument) but with additional visualization and
    post-processing methods for easier analysis.

    Args:
        mode (str): determines TPOTClassifier or TPOTRegressor to be used
            For example "Classification" or "regressor" are valid options.

        feature_names ([str]): list of feature/column names that is optionally
            passed for post-training analyses.

        **kwargs: keyword arguments accepted by TpotWrapper which have a few
            more arguments in addition to TPOTClassifier or TPOTRegressor
            For example: scoring='r2'; see TpotWrapper and TPOT documentation
            for more details.

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

    Returns (instantiated TpotWrapper class):
        TpotWrapper that has all methods of TPOTClassifier and TPOTRegressor as
        well as additional analysis methods.
    """
    if mode.lower() not in _classifier_modes \
            and mode.lower() not in _regressor_modes:
        raise ValueError('Unsupported mode: "{}"'.format(mode))

    return _tpot_class_wrapper(mode, **kwargs)


def _tpot_class_wrapper(mode, **kwargs):
    """
    Internal function to instantiate and return the child of the right class
    inherited from the two choices that TPOT package provides: TPOTClassifier
    and TPOTRegressor. The difference is that this new class has additional
    analysis and visualization methods.

    Args:
        mode (str): mode specifier that selects between classifier or regressor

        **kwargs: keyword arguments related to TPOTClassifier or TPOTRegressor

    Returns (class instance): instantiated TPOTAdaptor
    """

    class TPOTWrapper(
        TPOTClassifier if mode in _classifier_modes else TPOTRegressor, ):

        def __init__(self, **kwargs):
            self.features = None
            self.target = None
            self.models = None
            self.is_fit = False
            self.top_models = OrderedDict()
            self.top_models_scores = OrderedDict()
            self.feature_names = kwargs.pop('feature_names', None)
            self.random_state = kwargs.get('random_state', None)

            self.greater_score_is_better = is_greater_better(
                self.scoring_function
            )

            if self.random_state is not None:
                np.random.seed(self.random_state)

            kwargs['cv'] = kwargs.get('cv', 5)
            kwargs['n_jobs'] = kwargs.get('n_jobs', -1)
            super(TPOTWrapper, self).__init__(**kwargs)

        def get_top_models(self, return_scores=True):
            """
            Get a dictionary of top performing run for each sklearn model that
            was tried in TPOT. Must be called after the fit method. It also
            populates the instance variable "models" to a dictionary of all
            models tried and all their run history.

            Args:
                return_scores (bool): whether to return the score of the top
                    (selected) models (True) or their full parameters.

            Returns (dict):
                Top performing run for each sklearn model
            """

            # Update greater is better attribute as scoring function may have
            # changed between instantiation and call
            if not self.is_fit:
                raise RuntimeError("Error, the model has not yet been fit")

            self.greater_score_is_better = is_greater_better(
                self.scoring_function
            )

            # Get list of evaluated model names, cast to set and back
            # to get unique model names, instantiate ordered model dictionary
            evaluated_models = [key.split('(')[0]
                                for key in self.evaluated_individuals_.keys()]
            model_names = list(set(evaluated_models))
            models = OrderedDict({model: [] for model in model_names})

            # This makes a dict of model names mapped to all runs of that model
            for key, val in self.evaluated_individuals_.items():
                models[key.split('(')[0]].append(val)

            # For each base model type sort the runs by best score
            for model_name in model_names:
                models[model_name].sort(
                    key=lambda x: x['internal_cv_score'],
                    reverse=self.greater_score_is_better
                )

            # Gets a simplified dict of the model to only its best run
            top_models = {model: models[model][0] for model in models}

            # Sort the best individual models by type to best models overall
            self.top_models = OrderedDict(
                sorted(top_models.items(),
                       key=lambda x: x[1]['internal_cv_score'],
                       reverse=self.greater_score_is_better)
            )

            # Mapping of top models to just their score
            scores = {model: self.top_models[model]['internal_cv_score']
                      for model in self.top_models}

            # Sorted dict of top models just mapped to their top scores
            self.top_models_scores = OrderedDict(
                sorted(scores.items(),
                       key=lambda x: x[1],
                       reverse=self.greater_score_is_better)
            )

            self.models = models
            if return_scores:
                return self.top_models_scores
            else:
                return self.top_models

        def fit(self, features, target, **kwargs):
            """
            Wrapper function that is identical to the fit method of
            TPOTClassifier or TPOTRegressor. The purpose is to store the
            feature and target and use it in other methods of TPOTAutoML

            Args:
                please see the documentation of TPOT for a full description.

            Returns:
                please see the documentation of TPOT for a full description.
            """
            self.is_fit = True
            self.features = features
            self.target = target
            super(TPOTWrapper, self).fit(features, target, **kwargs)

    return TPOTWrapper(**kwargs)


class TPOTAdaptor(AutoMLAdaptor):
    """
    A dataframe adaptor for the TPOT classifiers and regressors.

    Args:
        mode (str): Either "regression" or "classification".
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
        These attributes are set during fitting.

        is_fit (bool): If True, the adaptor and backend are fit to a dataset.
        models

    """

    def __init__(self, mode, **tpot_kwargs):
        if mode.lower() not in _classifier_modes \
                and mode.lower() not in _regressor_modes:
            raise ValueError('Unsupported mode: "{}"'.format(mode))

        self._features = None
        self.target = None
        self.models = None
        self.is_fit = False
        self.random_state = tpot_kwargs.get('random_state', None)

        if mode in _classifier_modes:
            self.mode = 'classification'
            tpot_kwargs['config_dict'] = tpot_kwargs.get('config_dict',
                                                         classifier_config_dict_mb)
        else:
            self.mode = 'regression'
            tpot_kwargs['config_dict'] = tpot_kwargs.get('config_dict',
                                                         regressor_config_dict_mb)

        tpot_kwargs['cv'] = tpot_kwargs.get('cv', 5)
        tpot_kwargs['n_jobs'] = tpot_kwargs.get('n_jobs', -1)

        self.tpot_kwargs = tpot_kwargs
        self._backend = TPOTRegressor(**tpot_kwargs) if mode in _regressor_modes \
            else TPOTClassifier(**tpot_kwargs)

    def fit(self, df, target, **fit_kwargs):
        """
        Train a TPOTRegressor or TPOTClassifier by fitting on a dataframe.

        Args:
            df (pandas.DataFrame): The df to be used for training.
            target (str): The key used to identify the machine learning target.
            **fit_kwargs: Keyword arguments to be passed to the TPOT backend.
                These arguments must be valid arguments to the TPOTBase class.

        Returns:

        """
        # Prevent goofy pandas casting by casting to native
        y = df[target].values.tolist()
        X = df.drop(columns=target).values.tolist()
        self._features = df.drop(columns=target).columns.tolist()
        self._ml_data = {"X": X, "y": y}
        self.is_fit = True
        self.fitted_target = target
        return self._backend.fit(X, y, **fit_kwargs)

    @property
    def _best_models(self):
        """
        The best models found by TPOT, in order of descending performance.

        Performance is evaluated based on the TPOT scoring. This can be changed
        by passing a "scoring" kwarg into the __init__ method.

        Returns:
            best_models_and_scores (dict): Keys are names of models. Values
                are the best internal cv scores of that model with the
                best hyperparameter combination found.

        """
        if not self.is_fit:
            raise NotFittedError("Error, the model has not yet been fit")

        self.greater_score_is_better = is_greater_better(
            self.backend.scoring_function
        )

        # Get list of evaluated model names, cast to set and back
        # to get unique model names, instantiate ordered model dictionary
        evaluated_models = [key.split('(')[0]
                            for key in
                            self.backend.evaluated_individuals_.keys()]
        model_names = list(set(evaluated_models))
        models = OrderedDict({model: [] for model in model_names})

        # This makes a dict of model names mapped to all runs of that model
        for key, val in self.backend.evaluated_individuals_.items():
            models[key.split('(')[0]].append(val)

        # For each base model type sort the runs by best score
        for model_name in model_names:
            models[model_name].sort(
                key=lambda x: x['internal_cv_score'],
                reverse=self.greater_score_is_better
            )

        # Gets a simplified dict of the model to only its best run
        # Sort the best individual models by type to best models overall
        best_models = OrderedDict(
            sorted({model: models[model][0] for model in models}.items(),
                   key=lambda x: x[1]['internal_cv_score'],
                   reverse=self.greater_score_is_better))

        # Mapping of top models to just their score
        scores = {model: best_models[model]['internal_cv_score']
                  for model in best_models}

        # Sorted dict of top models just mapped to their top scores
        best_models_and_scores = OrderedDict(
            sorted(scores.items(),
                   key=lambda x: x[1],
                   reverse=self.greater_score_is_better))

        self.models = models
        return best_models_and_scores

    def predict(self, df, target):
        """
        Predict the target property of materials given a df of features.

        The predictions are appended to the dataframe in a column called:
            "{target} predicted"

        Args:
            df (pandas.DataFrame): Contains all features needed for ML (i.e.,
                all features contained in the training dataframe.
            target (str): The property to be predicted. Should match the target
                used for fitting. May or may not be present in the argument
                dataframe.

        Returns:
            (pandas.DataFrame): The argument dataframe plus a column containing
                the predictions of the target.

        """
        # todo: We should have the ability to ensembelize predictions based on
        # todo: the top models (including one model type with mutliple
        # todo: combinations of model params).
        if target != self.fitted_target:
            raise MatbenchError("Argument dataframe target {} is different from"
                                " the fitted dataframe target! {}"
                                "".format(target, self.fitted_target))
        elif not self.is_fit:
            raise NotFittedError("The TPOT models have not been fit!")
        elif not all([f in df.columns for f in self._features]):
            not_in_model = [f for f in self._features if f not in df.columns]
            not_in_df = [f for f in df.columns if f not in self._features]
            raise MatbenchError("Features used to build model are different "
                                "from df columns! Features located in model "
                                "not located in df: \n{} \n Features located "
                                "in df not in model: \n{}".format(not_in_df,
                                                                  not_in_model))
        else:
            X = df[self._features].values         # rectify feature order
            y_pred = self._backend.predict(X)
            df[target + " predicted"] = y_pred
            return df



if __name__ == "__main__":
    from matminer.datasets.dataset_retrieval import load_dataset
    from matbench.featurization import AutoFeaturizer
    from matbench.preprocessing import DataCleaner

    df = load_dataset("elastic_tensor_2015").rename(columns={"formula": "composition"})[["composition",  "K_VRH"]]
    dfp = df.iloc[60:90]
    df = df.iloc[:500]
    target = "K_VRH"

    af = AutoFeaturizer()
    af.fit(df, target)
    df = af.transform(df, target)

    dfp = dfp.drop(columns=[target])
    dfp = af.transform(dfp, target)

    dc = DataCleaner()
    df = dc.fit_transform(df, target=target)
    dfp = dc.transform(dfp, target=target)

    # df.to_csv("mini_training_df_automl.csv")
    # dfp.to_csv("mini_validation_df_automl.csv")


    print(target in dfp.columns)

    tpotw = TPOTAdaptor("regression", max_time_mins=2)
    tpotw.fit(df, target)
    dfp = tpotw.predict(dfp, target)
    print(dfp)
