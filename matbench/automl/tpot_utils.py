from collections import OrderedDict

import numpy as np
from tpot import TPOTClassifier, TPOTRegressor

from matbench.utils.utils import is_greater_better

__author__ = 'Alireza Faghaninia <alireza.faghaninia@gmail.com>' \
             'Daniel Dopp <dbdopp@lbl.gov>'

"""
-AF: 
TPOT (https://github.com/EpistasisLab/tpot) is an academic open-sourced 
package that seems to be suitable overall to matbench though there doesn't 
seem to be an explicit support for feature importance.

It uses genetic algorithm to find the estimator with the set of parameters that 
returns the best score (supports many scoring metrics)

    pros:
        + easy install: "pip install tpot"; written in python
        + easily accessible history of models tried and their accuracy 
          (in evaluated_individuals_)
        + automatic global optimization already implemented
        + seems more organized than automl
        + writes sample pipeline to file to resume/modify analysis
    cons:
        - no feature importance as far as I can tell
        - I have had some difficulties using it in Pycharm as opposed to 
          Terminal and Jupyter notebooks
"""

_classifier_modes = {'classifier', 'classification', 'classify'}

_regressor_modes = {'regressor', 'regression', 'regress'}


def TpotAutoml(mode, feature_names=None, **kwargs):
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
    kwargs['feature_names'] = feature_names

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

    Returns (class instance): instantiated TpotWrapper
    """
    if mode in _classifier_modes:
        tpot_class = TPOTClassifier
    else:
        tpot_class = TPOTRegressor

    class TpotWrapper(tpot_class):

        def __init__(self, **kwargs):
            self.features = None
            self.target = None
            self.models = None
            self.has_been_fit = False
            self.top_models = OrderedDict()
            self.top_models_scores = OrderedDict()
            self.feature_names = kwargs.pop('feature_names', None)
            self.random_state = kwargs.get('random_state', None)

            self.greater_score_is_better = is_greater_better(
                self.scoring_function
            )

            if self.random_state is not None:
                np.random.seed(self.random_state)

            if mode in _classifier_modes:
                self.mode = 'classification'
            else:
                self.mode = 'regression'

            kwargs['cv'] = kwargs.get('cv', 5)
            kwargs['n_jobs'] = kwargs.get('n_jobs', -1)
            super(TpotWrapper, self).__init__(**kwargs)

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
            if not self.has_been_fit:
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
            feature and target and use it in other methods of TpotAutoml

            Args:
                please see the documentation of TPOT for a full description.

            Returns:
                please see the documentation of TPOT for a full description.
            """
            self.has_been_fit = True
            self.features = features
            self.target = target
            super(TpotWrapper, self).fit(features, target, **kwargs)

    return TpotWrapper(**kwargs)
