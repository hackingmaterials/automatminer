"""
Base classes, mixins, and other inheritables.
"""
import abc
import logging

from automatminer.utils.log import (
    AMM_LOGGER_BASENAME,
    initialize_logger,
    initialize_null_logger,
)
from sklearn.base import BaseEstimator

__authors__ = ["Alex Dunn <ardunn@lbl.gov>", "Alex Ganose <aganose@lbl.gov>"]


class LoggableMixin:
    """A mixin class for easy logging (or absence of it)."""

    @property
    def logger(self):
        """Get the class logger.
        If the logger is None, the logging calls will be redirected to a dummy
        logger that has no output.
        """
        return self._logger

    @logger.setter
    def logger(self, new_logger):
        """Set a new logger.

        Args:
            new_logger (Logger, bool): A boolean or custom logger object to use
            for logging. Alternatively, if set to True, the default automatminer
            logger will be used. If set to False, then no logging will occur.
        """
        new_logger = self.get_logger(new_logger)
        if not isinstance(new_logger, logging.Logger):
            TypeError("The new logger must be an instance of the logger class.")

        self._logger = new_logger

        if hasattr(self, "autofeaturizer"):
            for x in ["autofeaturizer", "cleaner", "reducer", "learner"]:
                getattr(self, x)._logger = new_logger

    @staticmethod
    def get_logger(logger):
        """Handle boolean logger.
        Args:
            logger (Logger, bool): A custom logger object to use for logging.
                Alternatively, if set to True, the default automatminer logger
                will be used. If set to False, then no logging will occur.
        """
        # need comparison to True and False to avoid overwriting Logger objects
        if logger is True:
            logger = logging.getLogger(AMM_LOGGER_BASENAME)

            if not logger.handlers:
                initialize_logger(AMM_LOGGER_BASENAME)

        elif logger is False:
            logger = logging.getLogger(AMM_LOGGER_BASENAME + "_null")

            if not logger.handlers:
                initialize_null_logger(AMM_LOGGER_BASENAME)

        return logger

    @property
    def _log_prefix(self):
        return self.__class__.__name__ + ": "


class DFTransformer(abc.ABC, BaseEstimator):
    """A base class to allow easy transformation in the same way as
    TransformerMixin and BaseEstimator in sklearn, but for pandas dataframes.

    When implementing a base class adaptor, make sure to use @check_fitted
    and @set_fitted if necessary!
    """

    def __init__(self):
        self.is_fit = False

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
