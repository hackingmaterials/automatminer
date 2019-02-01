"""
Base classes, mixins, and other inheritables.
"""

import logging

from automatminer.utils.log_tools import initialize_logger, initialize_null_logger

__authors__ = ["Alex Dunn <ardunn@lbl.gov>", "Alex Ganose <aganose@lbl.gov>"]

logger_base_name = "automatminer"


class LoggableMixin:
    """A mixin class for easy logging (or absence of it)."""

    @property
    def logger(self):
        """Get the class lowgger.
        If the logger is None, the logging calls will be redirected to a dummy
        logger that has no output.
        """
        if hasattr(self, "_logger"):
            return self._logger
        else:
            raise AttributeError("Loggable object has no _logger attribute!")

    @staticmethod
    def get_logger(logger, level=None):
        """Set the class logger.
        Args:
            logger (Logger, bool): A custom logger object to use for logging.
                Alternatively, if set to True, the default automatminer logger
                will be used. If set to False, then no logging will occur.
            level (int): The log level. For example logging.DEBUG.
        """
        # need comparison to True and False to avoid overwriting Logger objects
        if logger is True:
            logger = logging.getLogger(logger_base_name)

            if not logger.handlers:
                initialize_logger(logger_base_name, level=level)

        elif logger is False:
            logger = logging.getLogger(logger_base_name + "_null")

            if not logger.handlers:
                initialize_null_logger(logger_base_name)

        logger.setLevel(logging.INFO)
        return logger


class DataframeTransformer:
    """
    A base class to allow easy transformation in the same way as
    TransformerMixin and BaseEstimator in sklearn.

    When implementing a base class adaptor, make sure to use @check_fitted
    and @set_fitted if necessary!
    """
    def fit(self, df, target, **fit_kwargs):
        """
        Fits the transformer to a dataframe, given a target.

        Args:
            df (pandas.DataFrame): The pandas dataframe to be fit.
            target (str): the target string specifying the ML target.
            fit_kwargs: Keyword paramters for fitting

        Returns:
            (DataFrameTransformer) This object (self)

        """
        raise NotImplementedError("{} has no fit method implemented!".format(
            self.__class__.__name__))

    def transform(self, df, target, **transform_kwargs):
        """
        Transforms a dataframe.

        Args:
            df (pandas.DataFrame): The pandas dataframe to be fit.
            target (str): the target string specifying the ML target.
            transform_kwargs: Keyword paramters for transforming

        Returns:
            (pandas.DataFrame): The transformed dataframe.

        """
        raise NotImplementedError("{} has no transform method implemented!".
                                  format(self.__class__.__name__))

    def fit_transform(self, df, target):
        """
        Combines the fitting and transformation of a dataframe.

        Args:
            df (pandas.DataFrame): The pandas dataframe to be fit.
            target (str): the target string specifying the ML target.

        Returns:
            (pandas.DataFrame): The transformed dataframe.

        """
        return self.fit(df, target).transform(df, target)


class AutoMLAdaptor(DataframeTransformer):
    """
    A base class to adapt from an AutoML backend to a sklearn-style fit/predict
    scheme and add a few extensions.

    When implementing a base class adaptor, make sure to use @check_fitted
    and @set_fitted if necessary!
    """
    def transform(self, df, target):
        return self.predict(df, target)

    def predict(self, df, target):
        """
        Using a fitted object, use the best model available to transform a
        dataframe not containing the target to a dataframe containing the
        predicted target values.

        Analagous to DataframeTransformer.transform

        Args:
            df (pandas.DataFrame): The dataframe to-be-predicted
            target: The target metric to be predicted. The output column for
                the data will be "predicted {target}".

        Returns:
            (pandas.DataFrame): The dataframe updated with predictions of the
                target property.

        """
        raise NotImplementedError("{} has no predict method implemented!".
                                  format(self.__class__.__name__))

    @property
    def features(self):
        """
        The features being used for machine learning.

        Returns:
            ([str]): The feature labels
        """
        try:
            return self._features
        except AttributeError:
            raise NotImplementedError("{} has no features attr implemented!".
                                      format(self.__class__.__name__))

    @property
    def ml_data(self):
        """
        The raw ML-data being passed to the backend.

        Returns:
            (dict): At minimum, the raw X and y matrices being used for training.
                May also contain other data.
        """
        try:
            return self._ml_data
        except AttributeError:
            raise NotImplementedError("{} has no ML data attr implemented!".
                                      format(self.__class__.__name__))

    @property
    def best_pipeline(self):
        """
        The best pipeline returned by the automl backend. Should implement fit
        and predict methods and be able to make predictions.

        Returns:
            sklearn.pipeline.Pipeline or BaseEstimator:
        """
        try:
            return self._best_pipeline
        except AttributeError:
            raise NotImplementedError("{} has no best models attr implemented!".
                                      format(self.__class__.__name__))

    @property
    def backend(self):
        """
        The raw, fitted backend object, if it exists.

        Returns:
            Backend object (e.g., TPOTClassifier)

        """
        try:
            return self._backend
        except AttributeError:
            raise NotImplementedError(
                "{} has no backend object attr implemented!".format(
                    self.__class__.__name__))