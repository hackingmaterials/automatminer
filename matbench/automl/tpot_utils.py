from tpot import TPOTClassifier
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

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=2, population_size=50, verbosity=2)
tpot.scoring_function = "accuracy"
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')

# very good method for us; keeps track of the score of different algorithms:
print(tpot.evaluated_individuals_)