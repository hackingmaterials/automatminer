"""
Base classes.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from pandas.api.types import is_numeric_dtype
from skrebate import ReliefF

from matbench.utils.utils import MatbenchError, setup_custom_logger
from matminer.featurizers.base import BaseFeaturizer



class Preprocesser(object):
    """
    Clean and prepare the data for visualization and training.

    Args:
        max_colnull (float): after generating features, drop the columns that
            have null/na rows with more than this ratio. Note that there is an
            important trade-off here. this ratio is high, one may lose more
            features and if it is low one may lose more samples.
        loglevel (int): the level of output; e.g. logging.DEBUG
        logpath (str): the path to the logfile dir, current folder by default.
    """


    def prune_correlated_features(self, df, target_key, R_max=0.95):
        """
        A feature selection method that remove those that are cross correlated
        by more than threshold.

        Args:
            df (pandas.DataFrame): The dataframe containing features, target_key
            target_key (str): the name of the target column/feature
            R_max (0<float<=1): if R is greater than this value, the
                feature that has lower correlation with the target is removed.

        Returns (pandas.DataFrame):
            the dataframe with the highly cross-correlated features removed.
        """
        corr = abs(df.corr())
        corr = corr.sort_values(by=target_key)
        rm_feats = []
        for feature in corr.columns:
            if feature == target_key:
                continue
            for idx, corval in zip(corr.index, corr[feature]):
                if np.isnan(corval):
                    break
                if idx == feature or idx in rm_feats:
                    continue
                else:
                    if corval >= R_max:
                        if corr.loc[idx, target_key] > corr.loc[feature, target_key]:
                            removed_feat = feature
                        else:
                            removed_feat = idx
                        if removed_feat not in rm_feats:
                            rm_feats.append(removed_feat)
                            self.logger.debug('"{}" correlates strongly with '
                                              '"{}"'.format(feature, idx))
                            self.logger.debug(
                                'removing "{}"...'.format(removed_feat))
                        if removed_feat == feature:
                            break
        if len(rm_feats) > 0:
            df = df.drop(rm_feats, axis=1)
            self.logger.info('These {} features were removed due to cross '
                             'correlation with the current features more than '
                             '{}:\n{}'.format(len(rm_feats), R_max, rm_feats))
        return df


class DataCleaner(BaseEstimator, TransformerMixin):

    """

    """
    def __init__(self, scale=False, max_na_frac=0.01, na_method='drop',
                 encode_categories=True, encoder='one-hot', drop_na_targets=True,
                 logger=setup_custom_logger()):
        self.scale = scale
        self.max_na_frac = max_na_frac
        self.na_method = na_method
        self.encoder = encoder
        self.encode_categories = encode_categories
        self.drop_na_targets = drop_na_targets
        self.logger = logger

    def fit(self, df, target):
        pass

    def transform(self, df, target):
        """
        A sequence of data pre-processing steps either through this class or
        sklearn.

        Args:
            df (pandas.DataFrame): Contains features and the target_key
            target (str): The name of the target in the dataframe

        Returns (pandas.DataFrame)
        """
        df = self.handle_na(df, target)
        df_numerical = self.to_numerical(df, target)
        y = df_numerical[target]
        X = df_numerical.drop(target, axis=1)
        X = MinMaxScaler().fit_transform(X) if self.scale else X
        return pd.concat([y, X], axis=1)

    def handle_na(self, df, target):
        """
        First pass for handling cells wtihout values (null or nan). Additional
        preprocessing may be necessary as one column may be filled with
        median while the other with mean or mode, etc.

        Args:
            df (pandas.DataFrame): the incumbent dataframe

        Returns:
            (pandas.DataFrame) The cleaned df
        """
        self.logger.info("Before handling na: {} samples, {} features".format(*df.shape))

        # Drop targets containing na before further processing
        if self.drop_na_targets:
            df = df.dropna(axis=0, how='any', subset=target)

        # Remove features failing the max_na_frac limit
        feats0 = set(df.columns)
        df = df.dropna(axis=1, thresh=int((1 - self.max_na_frac) * len(df)))
        if len(df.columns) < len(feats0):
            feats = set(df.columns)
            n_feats = len(feats0) - len(feats)
            napercent = self.max_na_frac * 100
            feat_names = feats0 - feats
            self.logger.info('These {} features were removed as they '
                             'had more than {}% missing values:\n{}'.format(n_feats, napercent, feat_names))

        # Handle all rows that still contain any nans
        if self.na_method == "drop":
            df = df.dropna(axis=0, how='any')
        else:
            df = df.fillna(method=self.na_method)
        self.logger.info("After handling na: {} samples, {} features".format(*df.shape))
        return df

    def to_numerical(self, df, target):
        """
        Transforms non-numerical columns to numerical columns which are
            machine learning-friendly.

        Args:
            df (pandas.DataFrame): The dataframe containing features
            encode_categories (bool): If True, retains features which are
                categorical (data type is string or object) and then
                one-hot encodes them. If False, drops them.
            encoding_method (str): choose a method for encoding the categorical
                variables. Current options: 'one-hot' and 'label'

        Returns:

        """

        number_cols = []
        object_cols = []
        for c in df.columns.values:
            try:
                df[c] = pd.to_numeric(df[c])
                number_cols.append(c)
            except (TypeError, ValueError):
                # The target is most likely strings which are not numeric.
                # Prevent target being encoded
                if c != target:
                    object_cols.append(c)

        number_df = df[number_cols]
        object_df = df[object_cols]
        if self.encode_categories:
            if self.encoder == 'one-hot':
                object_df = pd.get_dummies(object_df).apply(pd.to_numeric)
            elif self.encoder == 'label':
                for c in object_df.columns:
                    object_df[c] = LabelEncoder().fit_transform(object_df[c])
                self.logger.warning('LabelEncoder used for categorical colums '
                    'For access to the original labels via inverse_transform, '
                    'encode manually and set retain_categorical to False')

            return pd.concat([number_df, object_df], axis=1)
        else:
            return number_df



class FeatureReducer(DataFrameTransformer):
    def __init__(self):


        if n_rebate_features:
            self.logger.info(
                "ReBATE running: retaining {} numerical features.".format(
                    n_rebate_features))
            rf = ReliefF(n_features_to_select=n_rebate_features, n_jobs=-1)
            matrix = rf.fit_transform(X.values, y.values)
            # Todo: Find how to get the original labels back?  - AD
            rfcols = ["ReliefF {}".format(i) for i in range(matrix.shape[1])]
            X = pd.DataFrame(columns=rfcols, data=matrix, index=X.index)

        if n_pca_features:
            if self.scaler is None:
                if X.max().max() > 5.0:  # 5 allowing for StandardScaler
                    raise MatbenchError(
                        'attempted PCA before data normalization!')
            self.logger.info(
                "PCA running: retaining {} numerical features.".format(
                    n_pca_features))
            n_pca_features = PCA(n_components=n_pca_features)
            matrix = n_pca_features.fit_transform(X)
            # Todo: I don't know if there is a way to get labels for these - AD
            pcacols = ["PCA {}".format(i) for i in range(matrix.shape[1])]
            X = pd.DataFrame(columns=pcacols, data=matrix, index=X.index)