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
    from autosklearn.metrics import make_scorer
    from autosklearn.metrics import classification_metrics
except ImportError:
    sys.stderr.write("Please install auto-sklearn first!")
    sys.exit(1)


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
    output_folder:
    tmp_folder:
    dataset_name: dataset_name : str, optional (default=None)
        For creating nicer output. If None, a string will be determined by the
        md5 hash of the dataset.

    """

    def __init__(self,
                 X, y,
                 output_folder,
                 tmp_folder,
                 delete_output_folder_after_terminate=False,
                 delete_tmp_folder_after_terminate=False,
                 dataset_name=None,
                 time_left_for_this_task=3600,
                 per_run_time_limit=1800,
                 ml_memory_limit=3072,
                 resampling_strategy="holdout",
                 ensemble_size=1,
                 ensemble_nbest=1,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_preprocessors=None,
                 exclude_preprocessors=None):

        self.X = X
        self.y = y
        self.dataset_name = dataset_name
        self.auto_sklearn_kwargs = \
            {"output_folder": output_folder,
             "tmp_folder": tmp_folder,
             "delete_output_folder_after_terminate":
                 delete_output_folder_after_terminate,
             "delete_tmp_folder_after_terminate":
                 delete_tmp_folder_after_terminate,
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

    def auto_classification(self, metric=None):
        auto_classifier = autosklearn.classification.AutoSklearnClassifier(
            **self.auto_sklearn_kwargs)
        classification_metric = make_scorer('accuracy',
                                            sklearn.metrics.accuracy_score) \
            if metric is None else AutoSklearnML.get_auto_sklearn_metric(metric)
        auto_classifier.fit(self.X_train.copy(),
                            self.y_train.copy(),
                            metric=classification_metric,
                            dataset_name=self.dataset_name)

        # auto_classifier.refit(self.X_train.copy(), self.y_train.copy())
        print(auto_classifier.show_models())

        prediction = auto_classifier.predict(self.X_test)

        print("{} score:".format(metric),
              classification_metric._score_func(self.y_test, prediction))

    def auto_regression(self, metric=None):
        auto_regressor = autosklearn.regression.AutoSklearnRegressor(
            **self.auto_sklearn_kwargs)
        regression_metric = AutoSklearnML.get_auto_sklearn_metric(metric)
        auto_regressor.fit(self.X_train, self.y_train,
                           metric=regression_metric,
                           dataset_name=self.dataset_name)
        print(auto_regressor.show_models())

        prediction = auto_regressor.predict(self.X_test)
        print("{} score:".format(metric),
              regression_metric._score_func(self.y_test, prediction))

    @staticmethod
    def get_auto_sklearn_metric(metric):
        standard_regression_metrics = \
            {"r2":
                make_scorer('r2', sklearn.metrics.r2_score),
             "mean_squared_error":
                 make_scorer('mean_squared_error',
                             sklearn.metrics.mean_squared_error,
                             greater_is_better=False),
             "mean_absolute_error":
                 make_scorer('mean_absolute_error',
                             sklearn.metrics.mean_absolute_error,
                             greater_is_better=False),
             "median_absolute_error":
                 make_scorer('median_absolute_error',
                             sklearn.metrics.median_absolute_error,
                             greater_is_better=False)
             }

        standard_classification_metrics = \
            {"accuracy":
                make_scorer('accuracy', sklearn.metrics.accuracy_score),
             "balanced_accuracy":
                 make_scorer('balanced_accuracy',
                             classification_metrics.balanced_accuracy),
             "f1":
                 make_scorer('f1', sklearn.metrics.f1_score),
             "roc_auc":
                 make_scorer('roc_auc', sklearn.metrics.roc_auc_score,
                             greater_is_better=True, needs_threshold=True),
             "average_precision":
                 make_scorer('average_precision',
                             sklearn.metrics.average_precision_score,
                             needs_threshold=True),
             "precision":
                 make_scorer('precision', sklearn.metrics.precision_score),
             "recall":
                 make_scorer('recall', sklearn.metrics.recall_score)
             }

        return standard_regression_metrics.get(
            metric, standard_classification_metrics.get(metric))

if __name__ == '__main__':
    from matbench.data.load import load_glass_formation
    from pymatgen.core import Composition
    from matminer.featurizers.composition import ElementProperty

    df = load_glass_formation()
    df['composition'] = df["formula"].apply(lambda x: Composition(x))

    elemprop = ElementProperty.from_preset("matminer")
    df = elemprop.featurize_dataframe(df, col_id="composition")

    feature_cols = elemprop.feature_labels()
    target_col = "gfa"

    automl = AutoSklearnML(X=df[feature_cols],
                           y=df[target_col],
                           output_folder="/tmp/matbench_automl_tmp2",
                           tmp_folder="/tmp/matbench_automl_out2",
                           dataset_name="glasses_ternary",
                           time_left_for_this_task=60,
                           per_run_time_limit=30,
                           )
    automl.auto_classification()
