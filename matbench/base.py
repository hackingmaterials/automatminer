"""
Base classes, mixins, and other inheritables.
"""

import logging
from matbench.utils.utils import initialize_logger, initialize_null_logger

__authors__ = ["Alex Dunn <ardunn@lbl.gov>", "Alex Ganose <aganose@lbl.gov>"]


class LoggableMixin(object):
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

    def get_logger(self, logger):
        """Set the class logger.

        Args:
            logger (Logger, bool): A custom logger object to use for logging.
                Alternatively, if set to True, the default matbench logger will
                be used. If set to False, then no logging will occur.
        """
        # need comparison to True and False to avoid overwriting Logger objects
        if logger is True:
            logger = logging.getLogger(self.__module__.split('.')[0])

            if not logger.handlers:
                initialize_logger()

        elif logger is False:
            logger = logging.getLogger(self.__module__.split('.')[0] + "_null")

            if not logger.handlers:
                initialize_null_logger()

        return logger


class DataframeTransformer:
    """
    A class to allow easy transformation in the same way as TransformerMixin
    and BaseEstimator in sklearn.
    """
    def fit(self, df, target):
        raise NotImplementedError("{} has no fit method implemented!".format(
            self.__class__.__name__))

    def transformer(self, df, target):
        raise NotImplementedError("{} has no transform method implemented!".
                                  format(self.__class__.__name__))

    def fit_transform(self, df, target):
        return self.fit(df, target).transform(df, target)
