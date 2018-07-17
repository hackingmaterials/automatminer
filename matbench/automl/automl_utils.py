from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset

"""
Test of automl package as a candidate for main "auto machine learning" engine
for matbench.

-AF: here are some major advantages and disadvantages I found:
    pros:
        + easy to install
        + easy to get it to work and print some results
        + already have an internal implementation of feature importance.:
            - actually the "Importance" part comes from sklearn feature_importance_
            when available. However, they do have this other analysis where they
            go through each feature and see how their variance affect the variance
            of the predictions that could potentially used to define a custom
            "importance" or sensitivity.
    cons:
        - lack of documentation and organization
        - results are printed or to ".dill" file. Difficult to handle the results
    
    Might not be the best option.
"""


df_train, df_test = get_boston_dataset()

column_descriptions = {
    'MEDV': 'output',
    'CHAS': 'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
# ml_predictor.set_params_and_defaults(df_train, compare_all_models=True)

ml_predictor.train(df_train)
score = ml_predictor.score(df_test, df_test.MEDV)

print(score)
# ml_predictor.save()