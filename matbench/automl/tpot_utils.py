import numpy as np
import pandas as pd
from collections import OrderedDict
from matbench.utils.utils import MatbenchError, is_greater_better
from sklearn.model_selection import train_test_split
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


class ErrorAnalysis(object):
    """
    Evaluating the errors and uncertainty of a given machine learning model,
    bias, variance and tools assisting in manual inspection of errors and areas
    of improvements with that maximize impact. We also take advantage of methods
    already available in lime ml evaluation package: https://github.com/marcotcr/lime

    mode options are "regression" or "classification"
    """
    def __init__(self, model, X_train, y_train, X_test, y_test, mode, target=None,
                 features=None, test_samples_index=None, random_state=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.mode = mode
        self.target = target
        self.features= features
        self.test_samples_index = test_samples_index
        self.random_state = random_state


    def from_dataframe_iid(self, df, target, mode,
                           train_size=0.75, test_size=0.25, random_state=None):
        """
        A helper function to be called only if 1) the model is not fit yet and
        train/test split hasn't been done AND 2) the user does not want to
        assign so many arguments necessary in class instantiation.

        * Note that if this method is called, the model is again fit to X_train
        to avoid data leakage. If the model is already trained, do not call this
        method.

        Args:
            df:
            target:
            mode:
            train_size:
            test_size:
            random_state:

        Returns:

        """
        y = np.array(df[target])
        X = df.drop(target, axis=1).as_matrix()

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                             train_size=train_size,
                             test_size=test_size,
                             random_state=random_state)
        # re-train the model as it has been fit to part of the\ current X_test!
        self.model.fit(X_train, y_train)
        super (ErrorAnalysis, self).__init__(X_train, y_train, X_test, y_test,
                                             mode, target=target,
                   features=df.columns, test_samples_index=y_test.index)

    def get_data_for_error_analysis(self, X_test=None, y_test=None, nmax=100):
        """
        Returns points with the wrong labels in case of a classification
        problem or high error in case of a regression problem. This can be
        used for further manual error analysis.

        * Note that this method must be called after the fit.

        Args:
            X_test (nxm numpy matrix where n is numer of samples and m is
                the number of features)
            y_test (nx1 numpy array): target labels/values
            nmax (int): maximum number of bad predictions returned

        Returns (pandas.DataFrame):
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.as_matrix()
        if X_test.shape[1] != len(list(self.features)):
            raise MatbenchError('The number of columns/features of X_test '
                                'do NOT match with the original features')
        predictions = self.model.predict(X_test)
        if self.mode == 'regression':
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        wrong_pred_idx = []
        for idx, pred in enumerate(predictions):
            if pred != y_test[idx]:
                if self.mode == 'classification':
                    wrong_pred_idx.append(idx)
                elif self.mode == 'regression':
                    if abs(pred - y_test[idx]) >= rmse:
                        wrong_pred_idx.append(idx)
        if len(wrong_pred_idx) > nmax:
            wrong_pred_idx = np.random.choice(wrong_pred_idx, nmax,
                                              replace=False)
        df = pd.DataFrame(X_test, columns=self.features,
                          index=self.test_samples_index)

        if isinstance(y_test, pd.Series):
            y_name = y_test.name
            y_test = np.array(y_test)
        else:
            y_name = 'target'
        df['{}_true'.format(y_name)] = y_test
        df['{}_predicted'.format(y_name)] = predictions
        df = df.iloc[wrong_pred_idx]
        return df