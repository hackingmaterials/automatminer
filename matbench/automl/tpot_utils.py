from matbench.data.load import load_double_perovskites_gap
from matbench.featurize import Featurize
from matbench.preprocess import PreProcess
from matbench.utils.utils import MatbenchError, is_greater_better
from matminer.featurizers.composition import ElementProperty, TMetalFraction
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split


"""
-AF: 
    TPOT is an academic open-sourced package that seems to be suitable overall to
matbench though there doesn't seem to be an explicit support for feature importance.
It uses a combination of GP and genetic algorithm to find the estimator with the
set of parameters that returns the best scoring (supports many scoring metrics)
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

def TpotAutoml(model_type, **kwargs):
    """
    Returns a class wrapped on TPOTClassifier or TPOTRegressor (differentiated
    via model_type argument) but with additional visualization and
    post-processing methods for easier analysis.

    Args:
        model_type (str): determines TPOTClassifier or TPOTRegressor to be used
            For example "Classification" or "regressor" are valid options.
        **kwargs: keyword arguments accepted by TpotWrapper which have a few
            more arguments in addition to TPOTClassifier or TPOTRegressor
            For example: scoring='r2'; see TpotWrapper and TPOT documentation
            for more details.

    Returns (instantiated TpotWrapper class):
        TpotWrapper that has all methods of TPOTClassifier and TPOTRegressor as
        well as additional analysis methods.
    """
    if model_type.lower() in ['classifier', 'classification', 'classify']:
        return _tpot_class_wrapper(TPOTClassifier, **kwargs)
    elif model_type.lower() in ['regressor', 'regression', 'regress']:
        return _tpot_class_wrapper(TPOTRegressor, **kwargs)
    else:
        raise ValueError('Unsupported model_type: "{}"'.format(model_type))


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
            models = {model: [] for model in model_names}
            for k in self.evaluated_individuals_:
                models[k.split('(')[0]].append(self.evaluated_individuals_[k])
            for model_name in model_names:
                models[model_name]=sorted(models[model_name],
                                          key=lambda x: x['internal_cv_score'],
                                          reverse=self.greater_score_is_better)
                self.models = models
                self.top_models = {model: models[model][0] for model in models}
                self.top_models_scores = {model: self.top_models[
                    model]['internal_cv_score'] for model in self.top_models}
            if return_scores:
                return self.top_models_scores
            else:
                return self.top_models


    return TpotWrapper(**kwargs)


if __name__ == '__main__':
    # matbench-type example
    limit = 200
    target_col = 'gap gllbsc'
    df_init = load_double_perovskites_gap(return_lumo=False)[:limit]

    featzer = Featurize(df_init, ignore_cols=['a_1', 'b_1', 'a_2', 'b_2'])
    df_feats = featzer.featurize_formula(featurizers=[
        ElementProperty.from_preset(preset_name='matminer'), TMetalFraction()])

    prep = PreProcess(max_colnull=0.1)
    df = prep.handle_nulls(df_feats)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(target_col, axis=1).as_matrix(),
        df[target_col], train_size=0.75, test_size=0.25)


    tpot = TpotAutoml(model_type='regressor', generations=1, population_size=25,
                      verbosity=0, scoring='r2', random_state=23)
    print(tpot.scoring_function)

    tpot.fit(X_train, y_train)
    print(tpot.get_top_models())
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_iris_pipeline.py')

