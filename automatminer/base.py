"""
Base classes, mixins, and other inheritables.
"""
import abc
import logging

from automatminer.utils.log import initialize_logger, \
    initialize_null_logger, AMM_LOGGER_BASENAME

__authors__ = ["Alex Dunn <ardunn@lbl.gov>", "Alex Ganose <aganose@lbl.gov>"]


class LoggableMixin:
    """A mixin class for easy logging (or absence of it)."""

    @property
    def logger(self):
        """Get the class logger.
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
            logger = logging.getLogger(AMM_LOGGER_BASENAME)

            if not logger.handlers:
                initialize_logger(AMM_LOGGER_BASENAME, level=level)

        elif logger is False:
            logger = logging.getLogger(AMM_LOGGER_BASENAME + "_null")

            if not logger.handlers:
                initialize_null_logger(AMM_LOGGER_BASENAME)

        logger.setLevel(logging.INFO)
        return logger

    @property
    def _log_prefix(self):
        return self.__class__.__name__ + ": "


class DFTransformer(abc.ABC):
    """ A base class to allow easy transformation in the same way as
    TransformerMixin and BaseEstimator in sklearn, but for pandas dataframes.

    When implementing a base class adaptor, make sure to use @check_fitted
    and @set_fitted if necessary!
    """

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

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


class DFMLAdaptor(DFTransformer):
    """
    A base class to adapt from an AutoML backend to a sklearn-style fit/predict
    scheme and add a few extensions for pandas dataframes.

    When implementing a base class adaptor, make sure to use @check_fitted
    and @set_fitted if necessary!
    """

    @abc.abstractmethod
    def predict(self, df, target):
        """
        Using a fitted object, use the best model available to transform a
        dataframe not containing the target to a dataframe containing the
        predicted target values.

        Analagous to DFTransformer.transform

        Args:
            df (pandas.DataFrame): The dataframe to-be-predicted
            target: The target metric to be predicted. The output column for
                the data will be "predicted {target}".

        Returns:
            (pandas.DataFrame): The dataframe updated with predictions of the
                target property.

        """
        pass

    def transform(self, df, target):
        return self.predict(df, target)

    @property
    @abc.abstractmethod
    def features(self):
        """
        The features being used for machine learning.

        Returns:
            ([str]): The feature labels
        """
        pass

    @property
    @abc.abstractmethod
    def ml_data(self):
        """
        The raw ML-data being passed to the backend.

        Returns:
            (dict): At minimum, the raw X and y matrices being used to train.
                May also contain other data.
        """
        pass

    @property
    @abc.abstractmethod
    def best_pipeline(self):
        """
        The best pipeline returned by the automl backend. Should implement fit
        and predict methods and be able to make predictions.

        Returns:
            sklearn.pipeline.Pipeline or BaseEstimator:
        """
        pass

    @property
    @abc.abstractmethod
    def backend(self):
        """
        The raw, fitted backend object, if it exists.

        Returns:
            Backend object (e.g., TPOTClassifier)

        """
        pass
