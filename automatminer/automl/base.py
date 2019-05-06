"""
Base classes for automl.
"""

import abc

from automatminer.base import DFTransformer


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
