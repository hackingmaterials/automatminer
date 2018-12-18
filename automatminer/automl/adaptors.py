"""
Adaptor classes for using AutoML packages in a Matbench pipeline.

Current adaptor classes are:

    TPOTAdaptor: Uses the backend from the automl project TPOT, which can be
        found at https://github.com/EpistasisLab/tpot
"""
from collections import OrderedDict
from sklearn.exceptions import NotFittedError
from tpot import TPOTClassifier, TPOTRegressor
from automatminer.automl.tpot_configs.classifier import classifier_config_dict_mb
from automatminer.automl.tpot_configs.regressor import regressor_config_dict_mb
from automatminer.utils.package_tools import AutomatminerError, set_fitted, check_fitted
from automatminer.utils.ml_tools import is_greater_better, \
    regression_or_classification
from automatminer.base import AutoMLAdaptor, LoggableMixin
from automatminer.automl.keras_wrapper import NnWrapper

__authors__ = ['Alex Dunn <ardunn@lbl.gov'
               'Alireza Faghaninia <alireza.faghaninia@gmail.com>',
               'Qi Wang <wqthu11@gmail.com>',
               'Daniel Dopp <dbdopp@lbl.gov>',
               'Samy Cherfaoui <SCherfaoui@lbl.gov>']

_classifier_modes = {'classifier', 'classification', 'classify'}

_regressor_modes = {'regressor', 'regression', 'regress'}


class TPOTAdaptor(AutoMLAdaptor, LoggableMixin):
    """
    A dataframe adaptor for the TPOT classifiers and regressors.

    Args:
        tpot_kwargs: All kwargs accepted by a TPOTRegressor/TPOTClassifier
            or TPOTBase object.

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

        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default automatminer logger will be
            used. If set to False, then no logging will occur.

    Attributes:
        The following attributes are set during fitting.

        mode (str): Either "regression" or "classification"
        features (list): The features labels used to develop the ml model.
        ml_data (dict): The raw ml data used for training.
        best_pipeline (sklearn.Pipeline): The best fitted pipeline found.
        best_models (OrderedDict): The best model names and their scores.
        backend (TPOTBase): The TPOT object interface used for ML training.
        is_fit (bool): If True, the adaptor and backend are fit to a dataset.
        models (OrderedDict): The raw sklearn-style models output by TPOT.
        fitted_target (str): The target name in the df used for training.
    """

    def __init__(self, logger=True, **tpot_kwargs):
        tpot_kwargs['cv'] = tpot_kwargs.get('cv', 5)
        tpot_kwargs['n_jobs'] = tpot_kwargs.get('n_jobs', -1)
        tpot_kwargs['verbosity'] = tpot_kwargs.get('verbosity', 2)

        self.mode = None
        self._backend = None
        self.tpot_kwargs = tpot_kwargs
        self.fitted_target = None
        self._features = None
        self.models = None
        self._logger = self.get_logger(logger)
        self.is_fit = False
        self.random_state = tpot_kwargs.get('random_state', None)

    @set_fitted
    def fit(self, df, target, **fit_kwargs):
        """
        Train a TPOTRegressor or TPOTClassifier by fitting on a dataframe.

        Args:
            df (pandas.DataFrame): The df to be used for training.
            target (str): The key used to identify the machine learning target.
            **fit_kwargs: Keyword arguments to be passed to the TPOT backend.
                These arguments must be valid arguments to the TPOTBase class.

        Returns:
            TPOTAdaptor (self)

        """
        # Prevent goofy pandas casting by casting to native
        y = df[target].values.tolist()
        X = df.drop(columns=target).values.tolist()

        # Determine learning type based on whether classification or regression
        self.mode = regression_or_classification(df[target])
        if self.mode == "classification":
            self.tpot_kwargs['config_dict'] = self.tpot_kwargs.get(
                'config_dict', classifier_config_dict_mb)
            self._backend = TPOTClassifier(**self.tpot_kwargs)
        elif self.mode == "regression":
            self.tpot_kwargs['config_dict'] = self.tpot_kwargs.get(
                'config_dict', regressor_config_dict_mb)
            self._backend = TPOTRegressor(**self.tpot_kwargs)
        else:
            raise ValueError("Learning type {} not recognized as a valid mode "
                             "for {}".format(self.mode, self.__class__.__name__))
        self._features = df.drop(columns=target).columns.tolist()
        self._ml_data = {"X": X, "y": y}
        self.fitted_target = target
        self.logger.info("TPOT fitting started.")
        self._backend = self._backend.fit(X, y, **fit_kwargs)
        self.logger.info("TPOT fitting finished.")
        return self

    @property
    @check_fitted
    def best_models(self):
        """
        The best models found by TPOT, in order of descending performance.

        Performance is evaluated based on the TPOT scoring. This can be changed
        by passing a "scoring" kwarg into the __init__ method.

        Returns:
            best_models_and_scores (dict): Keys are names of models. Values
                are the best internal cv scores of that model with the
                best hyperparameter combination found.

        """
        self.greater_score_is_better = is_greater_better(
            self.backend.scoring_function)

        # Get list of evaluated model names, cast to set and back
        # to get unique model names, instantiate ordered model dictionary
        evaluated_models = [key.split('(')[0]
                            for key in
                            self.backend.evaluated_individuals_.keys()]
        model_names = list(set(evaluated_models))
        models = OrderedDict({model: [] for model in model_names})

        # This makes a dict of model names mapped to all runs of that model
        for key, val in self.backend.evaluated_individuals_.items():
            models[key.split('(')[0]].append(val)

        # For each base model type sort the runs by best score
        for model_name in model_names:
            models[model_name].sort(
                key=lambda x: x['internal_cv_score'],
                reverse=self.greater_score_is_better
            )

        # Gets a simplified dict of the model to only its best run
        # Sort the best individual models by type to best models overall
        best_models = OrderedDict(
            sorted({model: models[model][0] for model in models}.items(),
                   key=lambda x: x[1]['internal_cv_score'],
                   reverse=self.greater_score_is_better))

        # Mapping of top models to just their score
        scores = {model: best_models[model]['internal_cv_score']
                  for model in best_models}
        # Sorted dict of top models just mapped to their top scores
        best_models_and_scores = OrderedDict(
            sorted(scores.items(),
                   key=lambda x: x[1],
                   reverse=self.greater_score_is_better))
        self.models = models
        return best_models_and_scores

    @property
    @check_fitted
    def _best_pipeline(self):
        return self._backend.fitted_pipeline_

    @check_fitted
    def predict(self, df, target):
        """
        Predict the target property of materials given a df of features.

        The predictions are appended to the dataframe in a column called:
            "{target} predicted"

        Args:
            df (pandas.DataFrame): Contains all features needed for ML (i.e.,
                all features contained in the training dataframe.
            target (str): The property to be predicted. Should match the target
                used for fitting. May or may not be present in the argument
                dataframe.

        Returns:
            (pandas.DataFrame): The argument dataframe plus a column containing
                the predictions of the target.

        """
        # todo: We should have the ability to ensembelize predictions based on
        # todo: the top models (including one model type with mutliple
        # todo: combinations of model params).
        if target != self.fitted_target:
            raise AutomatminerError("Argument dataframe target {} is different from"
                                " the fitted dataframe target! {}"
                                "".format(target, self.fitted_target))
        elif not all([f in df.columns for f in self._features]):
            not_in_model = [f for f in self._features if f not in df.columns]
            not_in_df = [f for f in df.columns if f not in self._features]
            raise AutomatminerError("Features used to build model are different "
                                "from df columns! Features located in model "
                                "not located in df: \n{} \n Features located "
                                "in df not in model: \n{}".format(not_in_df,
                                                                  not_in_model))
        else:
            X = df[self._features].values  # rectify feature order
            y_pred = self._backend.predict(X)
            df[target + " predicted"] = y_pred
            self.logger.debug("Prediction finished successfully.")
            return df

class NeuralNetworkAdaptor(LoggableMixin):
    """
        A dataframe adaptor for a Keras neural network regressor/classifier.

        Args:
            logger (Logger, bool): A custom logger object to use for logging.
                Alternatively, if set to True, the default automatminer logger will be
                used. If set to False, then no logging will occur.
            hidden_layer_sizes: Number of layers in the neural network
            init: Initialization function to use for neural network
            optimizer: Optimizer used in neural network
            dropout: The dropout rate (used as a form of regularization to reduce overfitting)
            show_accuracy: Boolean to indicate whether accuracy should be shown when fitting/predicting with the model
            batch_spec: Tuple consisting of number of epochs the neural network should go through and batch size (number
                of training examples per epoch
            activation: Activation function to use for neural network. By default, we use relu
            input_noise: Any noise that must be added to the input of the system
            use_maxout: Boolean to indicate if network should be maxout network
            use_maxnorm: Boolean to indicate if maxnorm regularization constraints should be used
            learning_rate: Specifies learning rate decay of ADAM optimizer
            stop_early: True if neural network should stop training after loss starts to increases, false otherwise.
            **kwargs: Additional (optional) args needed for neural network wrapper.

        Attributes:
            The following attributes are set during fitting.

            mode (str): Either "regression" or "classification"
            features (list): The features labels used to develop the ml model.
            ml_data (dict): The raw ml data used for training.
            backend (NnWrapper): The Keras neural network architecture used for ML training.
            is_fit (bool): If True, the adaptor and backend are fit to a dataset.
            fitted_target (str): The target name in the df used for training.
            score (double): Internal CV score of the best model (Stored in backend)
    """
    def __init__(self, logger=True, init="glorot_uniform", optimizer="adam", hidden_layer_sizes=(100,), dropout=0.5, show_accuracy=True, batch_spec=((400, 1024), (100, -1)), activation="relu", input_noise=0., use_maxout=False, use_maxnorm=False, learning_rate=0.001, stop_early=False, **kwargs):
        self.mode = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.show_accuracy = show_accuracy
        self.batch_spec = batch_spec
        self.activation = activation
        self.input_noise = input_noise
        self.use_maxout = use_maxout
        self.use_maxnorm = use_maxnorm
        self.learning_rate = learning_rate
        self.stop_early = stop_early
        self._backend = None
        self.kwargs = kwargs
        self._features = None
        self._logger = self.get_logger(logger)
        self.is_fit = False
        self.init = init
        self.optimizer = optimizer
        self.score = 0

    @set_fitted
    def fit(self, df, target, **fit_kwargs):
        """
        Train a neural network by fitting on a dataframe.

        Args:
            df (pandas.DataFrame): The df to be used for training.
            target (str): The key used to identify the machine learning target.
            **fit_kwargs: Keyword arguments to be passed to the neural network's fit method.
                These arguments must be valid arguments to the Keras neural network fit method.

        Returns:
            NeuralNetworkAdaptor(self)

        """
        y = df[target]
        X = df.drop(columns=target)

        # Determine learning type based on whether classification or regression
        self.mode = regression_or_classification(df[target])
        self._features = df.drop(columns=target).columns.tolist()
        self._ml_data = {"X": X, "y": y}
        self.fitted_target = target
        print(self._logger)
        self._logger.info("Neural network fitting started.")
        self._backend = NnWrapper(self.init, self.optimizer, self.hidden_layer_sizes, self.dropout, self.show_accuracy, self.batch_spec, self.activation, self.input_noise, self.use_maxout, self.use_maxnorm, self.learning_rate, self.stop_early, self.mode)
        self._backend, self.score = self._backend.best_model(X, y, self.mode)
        self._backend.fit(X, y, **fit_kwargs)
        self._logger.info("Neural network fitting finished.")
        return self

    @check_fitted
    def predict(self, df, target):
        """
        Predict the target property of materials given a df of features.

        The predictions are appended to the dataframe in a column called:
            "{target} predicted"

        Args:
            df (pandas.DataFrame): Contains all features needed for ML (i.e.,
                all features contained in the training dataframe.
            target (str): The property to be predicted. Should match the target
                used for fitting. May or may not be present in the argument
                dataframe.

        Returns:
            (pandas.DataFrame): The argument dataframe plus a column containing
                the predictions of the target.

        """
        if target != self.fitted_target:
            raise AutomatminerErrorError("Argument dataframe target {} is different from"
                                " the fitted dataframe target! {}"
                                "".format(target, self.fitted_target))
        elif not self.is_fit:
            raise NotFittedError("The TPOT models have not been fit!")
        elif not all([f in df.columns for f in self._features]):
            not_in_model = [f for f in self._features if f not in df.columns]
            not_in_df = [f for f in df.columns if f not in self._features]
            raise AutomatminerError("Features used to build model are different "
                                "from df columns! Features located in model "
                                "not located in df: \n{} \n Features located "
                                "in df not in model: \n{}".format(not_in_df,
                                                                  not_in_model))
        else:
            X = df[self._features]  # rectify feature order
            y_pred = self._backend.predict(X)
            df[target + " predicted"] = y_pred
            self.logger.debug("Prediction finished successfully.")
            return df

    @property
    @check_fitted
    def best_model(self):
        """
        The best model found by the wrapper. Performance is evaluated based on cross validation scores

        Returns:
            self._backend, self.score (tuple): Contains best architecture returned as well as its cross
            validation score

        """
        return self._backend, self.score

if __name__ == "__main__":
    from matminer.datasets.dataset_retrieval import load_dataset
    from automatminer.featurization import AutoFeaturizer
    from automatminer.preprocessing import DataCleaner, FeatureReducer

    # Load a dataset
    df = load_dataset("elastic_tensor_2015").rename(
        columns={"formula": "composition"})[["composition", "K_VRH"]]
    testdf = df.iloc[501:550]
    traindf = df.iloc[:100]
    target = "K_VRH"

    # Get top-lvel transformers
    autofeater = AutoFeaturizer()
    cleaner = DataCleaner()
    reducer = FeatureReducer()
    learner = NeuralNetworkAdaptor()

    # Fit transformers on training data
    traindf = autofeater.fit_transform(traindf, target)
    traindf = cleaner.fit_transform(traindf, target)
    traindf = reducer.fit_transform(traindf, target)
    learner.fit(traindf, target)

    # Use transformers on testing data
    testdf = autofeater.transform(testdf, target)
    testdf = cleaner.transform(testdf, target)
    testdf = reducer.transform(testdf, target)
    testdf = learner.predict(testdf, target)
    print(testdf)
