"""
Base classes, mixins, and other inheritables.
"""

__authors__ = ["Alex Dunn <ardunn@lbl.gov>"]


class LoggableMixin:
    """
    A mixin class for easy logging (or absence of it).
    """
    def _log(self, lvl, msg):
        """
        Convenience method for logging.

        Args:
            lvl (str): Level of the log message, either "info", "warn", or "debug"
            msg (str): The message for the logger.

        Returns:
            None
        """
        if hasattr(self, "logger"):
            if self.logger is not None:
                if lvl == "warn":
                    self.logger.warning(msg)
                elif lvl == "info":
                    self.logger.info(msg)
                elif lvl == "debug":
                    self.logger.debug(msg)
        else:
            raise AttributeError("Loggable object has no logger attr!")


class DataframeTransformer:
    """
    A base class to allow easy transformation in the same way as
    TransformerMixin and BaseEstimator in sklearn.
    """
    def fit(self, df, target):
        raise NotImplementedError("{} has no fit method implemented!".format(self.__class__.__name__))

    def transform(self, df, target):
        raise NotImplementedError("{} has no transform method implemented!".format(self.__class__.__name__))

    def fit_transform(self, df, target):
        return self.fit(df, target).transform(df, target)


class AutoMLAdaptor:
    """
    A base class to adapt from an AutoML backend to a sklearn-style fit/predict
    scheme and add a few extensions.
    """

    def fit(self, df, target):
        raise NotImplementedError("{} has no fit method implemented!".format(self.__class__.__name__))

    def predict(self, df):
        raise NotImplementedError("{} has no predict method implemented!".format(self.__class__.__name__))

    @property
    def features(self):
        raise NotImplementedError("{} has no features attr implemented!".format(self.__class__.__name__))

    @property
    def best_model(self):
        raise NotImplementedError("{} has no best model attr implemented!".format(self.__class__.__name__))