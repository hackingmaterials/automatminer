# -*- encoding: utf-8 -*-
import sys
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics.classification import type_of_target

try:
    import autosklearn.classification
    import autosklearn.regression
    from autosklearn.automl import BaseAutoML
    from autosklearn.constants import *
except ImportError:
    sys.stderr.write('Please install auto-sklearn first!')
    sys.exit(1)
# from glass_learning.glass_learning.utils.check import check_output_path


class AutoSklearnML:
    """
    Perform machine learning based on auto-sklearn for automatic algorithm
    selection and hyperparameter optimization.
    Installation of auto-sklearn (https://github.com/automl/auto-sklearn)
    is needed.

    Args:
    X: array-like or sparse matrix of shape = [n_samples, n_features]
       The input features.
    y: array-like, shape = [n_samples] or [n_samples, n_outputs]
       The target.
    dataset_name: dataset_name : str, optional (default=None)
                  For creating nicer output. If None, a string will be
                  determined by the md5 hash of the dataset.
    output_folder:
    tmp_folder:
    """

    def __init__(self,
                 X, y,
                 output_folder,
                 tmp_folder,
                 delete_tmp_folder=False,
                 metric=None,
                 dataset_name=None,
                 time_left_for_this_task=3600,
                 per_run_time_limit=1800,
                 ml_memory_limit=3072,
                 resampling_strategy='holdout',
                 ensemble_size=1,
                 ensemble_nbest=1,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_preprocessors=None,
                 exclude_preprocessors=None):

        self.X = X
        self.y = y
        self.metric = metric
        self.dataset_name = dataset_name
        self.auto_sklearn_kwargs = \
            {"output_folder": output_folder,
             "tmp_folder": tmp_folder,
             "delete_tmp_folder": delete_tmp_folder,
             "time_left_for_this_task": time_left_for_this_task,
             "per_run_time_limit": per_run_time_limit,
             "ml_memory_limit": ml_memory_limit,
             "resampling_strategy": resampling_strategy,
             "ensemble_size": ensemble_size,
             "ensemble_nbest": ensemble_nbest,
             "include_estimators": include_estimators,
             "exclude_estimators": exclude_estimators,
             "include_preprocessors": include_preprocessors,
             "exclude_preprocessors": exclude_preprocessors}

        self.X_train, self.X_test, self.y_train, self.y_test = \
            sklearn.model_selection.train_test_split(self.X, self.y,
                                                     random_state=1)

    def auto_classification(self):
        auto_classifier = autosklearn.classification.AutoSklearnClassifier(
            **self.auto_sklearn_kwargs)

        auto_classifier.fit(self.X_train.copy(), self.y_train.copy(),
                            metric=self.metric,
                            dataset_name=self.dataset_name)

        auto_classifier.refit(self.X_train.copy(), self.y_train.copy())
        print(auto_classifier.show_models())

        prediction = auto_classifier.predict(self.X_test)
        print("Accuracy score: ",
              sklearn.metrics.accuracy_score(self.y_test, prediction))

    def auto_regression(self):
        auto_regressor = autosklearn.regression.AutoSklearnRegressor(
            **self.auto_sklearn_kwargs)
        auto_regressor.fit(self.X_train, self.y_train,
                           metric=self.metric,
                           dataset_name=self.dataset_name)
        print(auto_regressor.show_models())

        prediction = auto_regressor.predict(self.X_test)
        print("R2 score: ",
              sklearn.metrics.r2_score(self.y_test, prediction))
