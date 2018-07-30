import numpy as np
from collections import OrderedDict
from matbench.utils.utils import is_greater_better
from tpot import TPOTClassifier, TPOTRegressor

__author__ = 'Alireza Faghaninia <alireza.faghaninia@gmail.com>'

"""
-AF: 
    TPOT (https://github.com/EpistasisLab/tpot) is an academic open-sourced 
package that seems to be suitable overall to
matbench though there doesn't seem to be an explicit support for feature importance.
It uses genetic algorithm to find the estimator with the set of parameters that 
returns the best score (supports many scoring metrics)
    pros:
        + easy install: "pip install tpot"; written in python
        + easily accessible history of models tried and their accuracy in evaluated_individuals_
        + automatic global optimization already implemented
        + seems more organized than automl
        + writes sample pipeline to file to resume/modify analysis
    cons:
        - no feature importance as far as I can tell
        - I have had some difficulties using it in Pycharm as opposed to Terminal and Jupyter notebooks
"""

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
    if mode.lower() in ['classifier', 'classification', 'classify']:
        return _tpot_class_wrapper(TPOTClassifier, **kwargs)
    elif mode.lower() in ['regressor', 'regression', 'regress']:
        return _tpot_class_wrapper(TPOTRegressor, **kwargs)
    else:
        raise ValueError('Unsupported mode: "{}"'.format(mode))


def _tpot_class_wrapper(tpot_class, **kwargs):
    """
    Internal function to instantiate and return the child of the right class
    inherited from the two choices that TPOT package provides: TPOTClassifier
    and TPOTRegressor. The difference is that this new class has additional
    analyis and visualization methods.
    Args:
        tpot_class (class object): TPOTClassifier or TPOTRegressor
        **kwargs: keyword arguments related to TPOTClassifier or TPOTRegressor

    Returns (class instance): instantiated TpotWrapper
    """
    class TpotWrapper(tpot_class):

        def __init__(self, **kwargs):
            self.models  = None
            self.top_models = OrderedDict()
            self.top_models_scores = OrderedDict()
            self.feature_names = kwargs.pop('feature_names', None)
            if tpot_class.__name__ == 'TPOTClassifier':
                self.mode = 'classification'
            elif tpot_class.__name__ == 'TPOTRegressor':
                self.mode = 'regression'
            self.random_state = kwargs.get('random_state', None)
            if self.random_state is not None:
                np.random.seed(self.random_state)

            kwargs['cv'] = kwargs.get('cv', 5)
            kwargs['n_jobs'] = kwargs.get('n_jobs', -1)
            super(tpot_class, self).__init__(**kwargs)


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
            self.greater_score_is_better = is_greater_better(self.scoring_function)
            model_names = list(set([key.split('(')[0] for key in
                                          self.evaluated_individuals_.keys()]))
            models = OrderedDict({model: [] for model in model_names})
            for k in self.evaluated_individuals_:
                models[k.split('(')[0]].append(self.evaluated_individuals_[k])
            for model_name in model_names:
                models[model_name]=sorted(models[model_name],
                                          key=lambda x: x['internal_cv_score'],
                                          reverse=self.greater_score_is_better)
                self.models = models
                top_models = {model: models[model][0] for model in models}
                self.top_models = OrderedDict(
                    sorted(top_models.items(),
                           key=lambda x:x[1]['internal_cv_score'],
                           reverse=self.greater_score_is_better))
                scores = {model: self.top_models[model]['internal_cv_score']\
                          for model in self.top_models}
                self.top_models_scores = OrderedDict(sorted(
                    scores.items(), key=lambda x: x[1],
                    reverse=self.greater_score_is_better))
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
            self.features = features
            self.target = target
            super(tpot_class, self).fit(features, target, **kwargs)
    return TpotWrapper(**kwargs)
