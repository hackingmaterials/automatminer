"""
Base classes for automl.
"""

import abc
from typing import List
import logging

import numpy as np
import pandas as pd

from automatminer.base import DFTransformer
from automatminer.utils.log import AMM_LOG_PREDICT_STR, log_progress
from automatminer.utils.pkg import AutomatminerError, check_fitted

logger = logging.getLogger(__name__)


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
    def features(self) -> (List, np.ndarray):
        """
        The features being used for machine learning.

        Returns:
            ([str]): The feature labels
        """
        pass

    @property
    @abc.abstractmethod
    def backend(self):
        """
        The AutoML backend object. Does not need to implement any methods for
        compatibility with higher level classes. If no AutoML backend is present
        e.g., SinglePipelineAdaptor, backend = None.

        Does not need to be serializable, as matpipe.save will not save
        backends.
        """
        pass

    @property
    @abc.abstractmethod
    def best_pipeline(self):
        """
        The best ML pipeline found by the backend. Can be any type though
        BaseEstimator is preferred.

            1. MUST implement a .predict method unless DFMLAdaptor.predict is
            overridden!

            2. MUST be serializable!

        Should be as close to the algorithm as possible - i.e., instead of
        calling TPOTClassifier.fit, calls TPOTClassifier.fitted_pipeline_, so
        that examining the true form of models is more straightforward.
        """
        pass

    @check_fitted
    def serialize(self) -> None:
        """
        Assign the adaptor components to be serializable.

        For example, TPOTBase-based backends are not serializable themselves.
        The adaptor attributes need to be reassigned in order to serialize the
        entire pipeline as pickle.

        If the backend serializes without extra effort, there is no need to
        override this method.

        Returns:
            None
        """
        return None

    @check_fitted
    def deserialize(self) -> None:
        """
        Invert the operations in serialize, if necessary. Useful if you are
        going to keep using this pipeline after saving it and want to retain
        the full functionality before the main python process ends.

        If the backend serializes without extra effort, there is no need to
        override this method.

        Returns:
            None
        """
        return None

    @check_fitted
    @log_progress(logger, AMM_LOG_PREDICT_STR)
    def predict(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Predict the target property of materials given a df of features. This
        base method is widely applicanble across different AutoML backends.

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
            not_in_model = [f for f in self.features if f not in df.columns]
            not_in_df = [f for f in df.columns if f not in self.features]
            raise AutomatminerError(
                "Features used to build model are different from df columns! "
                "Features located in model not located in df: \n{} \n "
                "Features located in df not in model: \n{}"
                "".format(not_in_df, not_in_model)
            )
        else:
            X = df[self.features].values  # rectify feature order
            y_pred = self.best_pipeline.predict(X)
            df[target + " predicted"] = y_pred

            log_msg = "Prediction finished successfully."
            try:
                logger.info(self._log_prefix + log_msg)
            except AttributeError:
                pass
            return df

    def transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        return self.predict(df, target)
