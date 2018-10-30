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
        """
        Fits the transformer to a dataframe, given a target.

        Args:
            df (pandas.DataFrame): The pandas dataframe to be fit.
            target (str): the target string specifying the ML target.

        Returns:
            (AutoMLAdaptor) This object (self)

        """
        raise NotImplementedError("{} has no fit method implemented!".format(self.__class__.__name__))

    def transform(self, df, target):
        """
        Transforms a dataframe.

        Args:
            df (pandas.DataFrame): The pandas dataframe to be fit.
            target (str): the target string specifying the ML target.

        Returns:
            (pandas.DataFrame): The transformed dataframe.

        """
        raise NotImplementedError("{} has no transform method implemented!".format(self.__class__.__name__))

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
    """

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
        raise NotImplementedError("{} has no predict method implemented!".format(self.__class__.__name__))

    @property
    def features(self):
        """
        The features being used for machine learning.

        Returns:
            ([str]): The feature labels
        """
        raise NotImplementedError("{} has no features attr implemented!".format(self.__class__.__name__))

    @property
    def ml_data(self):
        """
        The raw ML-data being passed to the backend.

        Returns:
            (dict): At minimum, the raw X and y matrices being used for training.
                May also contain other data.
        """
        raise NotImplementedError("{} has no ML data attr implemented!".format(self.__class__.__name__))


    @property
    def best_models(self):
        """
        The best models returned by the AutoML backend.

        Returns:
            (list or OrderedDict}: The best models as determined by the AutoML package.
        """
        raise NotImplementedError("{} has no best models attr implemented!".format(self.__class__.__name__))

    @property
    def backend(self):
        """
        The raw, fitted backend object, if it exists.

        Returns:
            Backend object (e.g., TPOTClassifier)

        """
        raise NotImplementedError("{} has no backend object attr implemented!".format(self.__class__.__name__))