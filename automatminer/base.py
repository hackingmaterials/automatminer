"""
Base classes, mixins, and other inheritables.
"""
import abc
from sklearn.base import BaseEstimator

__authors__ = ["Alex Dunn <ardunn@lbl.gov>", "Alex Ganose <aganose@lbl.gov>"]


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

    @property
    def _log_prefix(self):
        """
        The class's log prefix.

        Without log_prefix:
        2019.10.15 WARNING Some log message.

        with log prefix:
        2019.10.15 WARNING DataCleaner: Some log message.

        Returns:
            (str): The log prefix.

        """
        return self.__class__.__name__ + ": "
