from matbench.data.load import load_double_perovskites_gap
from matbench.featurize import Featurize
from matbench.preprocess import PreProcess
from matminer.featurizers.composition import ElementProperty, TMetalFraction
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

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

# irist simple example
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
#     iris.target.astype(np.float64), train_size=0.75, test_size=0.25)
# tpot = TPOTClassifier(generations=2, population_size=50, verbosity=2)

# matbench-type example
limit = 200
target_col = 'gap gllbsc'
df_init = load_double_perovskites_gap(return_lumo=False)[:limit]

featzer = Featurize(df_init, ignore_cols=['a_1', 'b_1', 'a_2', 'b_2'])
df_feats = featzer.featurize_formula(featurizers=[
    ElementProperty.from_preset(preset_name='matminer'), TMetalFraction()])

prep = PreProcess(df_feats, target_col=target_col)
df = prep.preprocess()

X_train, X_test, y_train, y_test = train_test_split(df.drop(target_col, axis=1).as_matrix(),
    df[target_col], train_size=0.75, test_size=0.25)


tpot = TPOTRegressor(generations=2, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')

# very good method for us; keeps track of the score of different algorithms:
print(tpot.evaluated_individuals_)