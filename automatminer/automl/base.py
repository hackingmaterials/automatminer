"""
Base classes for automl.
"""

import abc
from typing import List, Dict

import pandas as pd
from sklearn.base import BaseEstimator

from automatminer.base import DFTransformer
from automatminer.utils.log import AMM_LOG_PREDICT_STR, log_progress
from automatminer.utils.pkg import AutomatminerError, check_fitted


class DFMLAdaptor(DFTransformer):
    """
    A base class to adapt from an AutoML backend to a sklearn-style fit/predict
    scheme and add a few extensions for pandas dataframes.

    When implementing a base class adaptor, make sure to use @check_fitted
    and @set_fitted if necessary!
    """

    @property
    @abc.abstractmethod
    def fitted_target(self) -> str:
        """
        The target (a string) on which the adaptor was fit on.
        Returns:
            (str): The fitted target label.
        """
        pass

    @property
    @abc.abstractmethod
    def features(self) -> List:
        """
        The features being used for machine learning.

        Returns:
            ([str]): The feature labels
        """
        pass

    @property
    @abc.abstractmethod
    def ml_data(self) -> Dict:
        """
        The raw ML-data being passed to the backend.

        Returns:
            (dict): At minimum, the raw X and y matrices being used to train.
                May also contain other data.
        """
        pass

    @property
    @abc.abstractmethod
    def best_pipeline(self) -> BaseEstimator:
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
        The raw, fitted backend object, if it exists. Does not need to directly
        implement fit and predict methods.

        Returns:
            Backend object (e.g., TPOTClassifier)

        """
        pass

    @log_progress(AMM_LOG_PREDICT_STR)
    @check_fitted
    def predict(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Predict the target property of materials given a df of features. This
        base method is widely applicable across many kinds of DFMLAdaptors.

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
            raise AutomatminerError(
                "Argument dataframe target ({}) is different from the fitted "
                "dataframe target! ({})".format(target, self.fitted_target)
            )
        elif not all([f in df.columns for f in self.features]):
            not_in_model = [f for f in self.features if
                            f not in df.columns]
            not_in_df = [f for f in df.columns if f not in self.features]
            raise AutomatminerError(
                "Features used to build model are different from df columns! "
                "Features located in model not located in df: \n{} \n "
                "Features located in df not in model: \n{}"
                "".format(not_in_df, not_in_model))
        else:
            X = df[self.features].values  # rectify feature order
            y_pred = self.backend.predict(X)
            df[target + " predicted"] = y_pred

            log_msg = "Prediction finished successfully."
            try:
                self.logger.info(self._log_prefix + log_msg)
            except AttributeError:
                pass
            return df

    def transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        return self.predict(df, target)

