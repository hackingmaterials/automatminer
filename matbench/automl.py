import warnings
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

try:
    import autosklearn.classification
    import autosklearn.regression
    from autosklearn.metrics import make_scorer
    from autosklearn.metrics import classification_metrics
except ImportError:
    raise ImportError("Please install auto-sklearn first! "
                      "See: https://github.com/automl/auto-sklearn")


class AutoSklearnML:
    """
    Perform machine learning based on auto-sklearn for automatic algorithm
    selection and hyperparameter optimization. Data preprocessing will also
    be performed before machine learning.
    Installation of auto-sklearn (https://github.com/automl/auto-sklearn)
    is needed.

    Args:
        X: array-like or sparse matrix of shape = [n_samples, n_features]
           The input features.
        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
           The target.
        dataset_name: dataset_name (str): optional (default=None)
            For creating nicer output. If None, a string will be determined by
            md5 hash of the dataset.
        time_left_for_this_task (int): optional (default=3600)
            Time limit in seconds for the search of appropriate
            models. By increasing this value, *auto-sklearn* has a higher
            chance of finding better models.
        per_run_time_limit (int): optional (default=360)
            Time limit for a single call to the machine learning model.
            Model fitting will be terminated if the machine learning
            algorithm runs over the time limit. Set this value high enough so
            that typical machine learning algorithms can be fit on the
            training data.
        ml_memory_limit (int): optional (3072)
            Memory limit in MB for the machine learning algorithm.
            `auto-sklearn` will stop fitting the machine learning algorithm if
            it tries to allocate more than `ml_memory_limit` MB.
        ensemble_size (int): optional (default=50)
            Number of models added to the ensemble built by *Ensemble
            selection from libraries of models*. Models are drawn with
            replacement.

        ensemble_nbest (int): optional (default=50)
            Only consider the ``ensemble_nbest`` models when building an
            ensemble. Implements `Model Library Pruning` from `Getting the
            most out of ensemble selection`.
        include_estimators (list): optional (None)
            If None, all possible estimators are used. Otherwise specifies
            set of estimators to use.

        exclude_estimators (list):  optional (None)
            If None, all possible estimators are used. Otherwise specifies
            set of estimators not to use. Incompatible with include_estimators.

        include_preprocessors (list): optional (None)
            If None all possible preprocessors are used. Otherwise specifies
             set of preprocessors to use.

        exclude_preprocessors (list): optional (None)
            If None all possible preprocessors are used. Otherwise specifies set
            of preprocessors not to use. Incompatible with
            include_preprocessors.

        resampling_strategy (string):  optional ('holdout')
            how to handle overfitting, might need 'resampling_strategy_arguments'
            * 'holdout': 67:33 (train:test) split
            * 'holdout-iterative-fit':  67:33 (train:test) split, calls
              iterative fit where possible
            * 'cv': crossvalidation, requires 'folds'

        output_folder : string, optional (None)
            folder to store predictions for optional test set, if ``None``
            automatically use ``/tmp/autosklearn_output_$pid_$random_number``

        tmp_folder : string, optional (None)
            folder to store configuration output and log files, if ``None``
            automatically use ``/tmp/autosklearn_tmp_$pid_$random_number``

        delete_output_folder_after_terminate: bool, optional (False)
            remove output_folder, when finished.

        delete_tmp_folder_after_terminate: string, optional (False)
            remove tmp_folder, when finished.
    """

    def __init__(self,
                 X, y,
                 dataset_name=None,
                 time_left_for_this_task=3600,
                 per_run_time_limit=360,
                 ml_memory_limit=3072,
                 ensemble_size=1,
                 ensemble_nbest=1,
                 include_estimators=None,
                 exclude_estimators=None,
                 include_preprocessors=None,
                 exclude_preprocessors=None,
                 resampling_strategy="holdout",
                 output_folder=None,
                 tmp_folder=None,
                 delete_output_folder_after_terminate=False,
                 delete_tmp_folder_after_terminate=False):

        self.X = X
        self.y = y
        self.dataset_name = dataset_name
        self.auto_sklearn_kwargs = \
            {"time_left_for_this_task": time_left_for_this_task,
             "per_run_time_limit": per_run_time_limit,
             "ml_memory_limit": ml_memory_limit,
             "resampling_strategy": resampling_strategy,
             "ensemble_size": ensemble_size,
             "ensemble_nbest": ensemble_nbest,
             "include_estimators": include_estimators,
             "exclude_estimators": exclude_estimators,
             "include_preprocessors": include_preprocessors,
             "exclude_preprocessors": exclude_preprocessors,
             "output_folder": output_folder,
             "tmp_folder": tmp_folder,
             "delete_output_folder_after_terminate":
                 delete_output_folder_after_terminate,
             "delete_tmp_folder_after_terminate":
                 delete_tmp_folder_after_terminate}

        self.X_train, self.X_test, self.y_train, self.y_test = \
            sklearn.model_selection.train_test_split(self.X, self.y,
                                                     random_state=1)

    def classification(self, metric="accuracy"):
        """
        Perform auto_classification.
        Args:
            metric (str): The evaluation metric of classification.
                 This will be mapped by AutoSklearnML.get_auto_sklearn_metric
                 to an instance of :class:`autosklearn.metrics.Scorer` as
                 created by :meth:`autosklearn.metrics.make_scorer`.
                 Default metric: "accuracy".
                 Other supported metrics: "balanced_accuracy", "f1",
                                          "roc_auc", "average_precision",
                                          "precision", "recall"

        Returns:

        """
        auto_classifier = autosklearn.classification.AutoSklearnClassifier(
            **self.auto_sklearn_kwargs)
        classification_metric = AutoSklearnML.get_classification_metric(metric)
        auto_classifier.fit(self.X_train.copy(),
                            self.y_train.copy(),
                            metric=classification_metric,
                            dataset_name=self.dataset_name)

        # auto_classifier.refit(self.X_train.copy(), self.y_train.copy())
        print(auto_classifier.show_models())

        prediction = auto_classifier.predict(self.X_test)

        print("{} score:".format(metric),
              classification_metric._score_func(self.y_test, prediction))

    def regression(self, metric="r2"):
        """
        Perform auto_regression.
        Args:
            metric (str): The evaluation metric of regression.
                 This will be mapped by AutoSklearnML.get_auto_sklearn_metric
                 to an instance of :class:`autosklearn.metrics.Scorer` as
                 created by :meth:`autosklearn.metrics.make_scorer`.
                 Default metric: "r2".
                 Other supported metrics: "mean_squared_error",
                                          "mean_absolute_error",
                                          "median_absolute_error"

        Returns:

        """
        auto_regressor = autosklearn.regression.AutoSklearnRegressor(
            **self.auto_sklearn_kwargs)
        regression_metric = AutoSklearnML.get_regression_metric(metric)
        auto_regressor.fit(self.X_train, self.y_train,
                           metric=regression_metric,
                           dataset_name=self.dataset_name)
        print(auto_regressor.show_models())

        prediction = auto_regressor.predict(self.X_test)
        print("{} score:".format(metric),
              regression_metric._score_func(self.y_test, prediction))

    @staticmethod
    def get_classification_metric(metric):
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
        classification_metric = standard_classification_metrics.get(metric)
        if metric is not None and classification_metric is None:
            warnings.warn("This metric \"{}\" is not a supported metric for "
                          "classification. The metric will be reset as default "
                          "\"accuracy\".".format(metric))
            classification_metric = standard_classification_metrics.get(
                "accuracy")
        return classification_metric

    @staticmethod
    def get_regression_metric(metric):
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
        regression_metric = standard_regression_metrics.get(metric)
        if metric is not None and regression_metric is None:
            warnings.warn("This metric \"{}\" is not a supported metric for "
                          "regression. The metric will be reset as default "
                          "\"r2\".".format(metric))
            regression_metric = standard_regression_metrics.get("r2")
        return regression_metric


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
                           dataset_name="ternary glass formation",
                           time_left_for_this_task=60,
                           per_run_time_limit=30,
                           output_folder="/tmp/matbench_automl/tmp",
                           tmp_folder="/tmp/matbench_automl/out")
    automl.classification()
