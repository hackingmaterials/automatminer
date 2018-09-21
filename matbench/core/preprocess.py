import logging
import numpy as np
import pandas as pd
from matbench.utils.utils import setup_custom_logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import is_numeric_dtype
from skrebate import ReliefF


class Preprocess(object):
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

    def __init__(self, loglevel=None, logpath='.'):
        self.logger = setup_custom_logger(filepath=logpath, level=loglevel)

    def preprocess(self, df, target_key, scale=False, n_pca_features=None,
                   n_rebate_features=None, max_na_frac=0.05, na_method='drop',
                   retain_categorical=True):
        """
        A sequence of data pre-processing steps either through this class or
        sklearn.

        Args:
            df (pandas.DataFrame): Contains features and the target_key
            target_key (str): The name of the target in the dataframe
            scale (bool): whether to scale/normalize the data
            n_pca_features (int or None): Number of features to select with
                principal component analysis (PCA). None or 0 avoids running
                PCA. Only works on numerical features.
            n_rebate_features (int or None): Use the EpitasisLab ReBATE feature
                selection algorithm to reduce the dimensions of the data. None
                or 0 avoids running ReBATE. Only works on numerical features.
            max_na_frac (float): The maximum fraction of na entries in a column
                (feature) allowed before the column is handled by handle_na
            na_method (str): The method by which handle_na handles nulls. Valid
                arguments are 'drop' or pandas.fillna args (e.g., "mode")
            retain_categorical (bool): If True, retains features which are
                categorical and then One-Hot encodes them. If False, drops them.

        Returns (pandas.DataFrame)
        """

        # Remove na rows including those where target=na
        df = self.handle_na(df, max_na_frac=max_na_frac, na_method=na_method)

        for c in df.columns.values:
            try:
                df[c] = pd.to_numeric(df[c])
            except (TypeError, ValueError):
                # The target is most likely strings which are not numeric.
                pass

        if is_numeric_dtype(df[target_key]):
            # Pruning correlated features automatically takes accounts for dtype
            df = self.prune_correlated_features(df, target_key)
        else:
            df[target_key] = df[target_key].astype(str, copy=False)

        # Todo: At the moment, scaling and feature reduction converts ints to floats
        number_cols = [k for k in df.columns.values if
                       is_numeric_dtype(df[k]) and k != target_key]

        if retain_categorical:
            object_cols = [k for k in df.columns.values if
                           k not in number_cols and k != target_key]
        else:
            object_cols = []

        number_df = df[number_cols]
        object_df = df[object_cols]
        targets = df[target_key].copy(deep=True)

        # Todo: StandardScaler might be better
        # Todo: Data *must* be standardized for PCA...
        if scale:
            number_df[number_cols] = MinMaxScaler().fit_transform(number_df)

        object_df = pd.get_dummies(object_df).apply(pd.to_numeric)

        if n_rebate_features:
            self.logger.info(
                "ReBATE running: retaining {} numerical features.".format(
                    n_rebate_features))
            rf = ReliefF(n_features_to_select=n_rebate_features, n_jobs=-1)
            x = rf.fit_transform(number_df.values, targets.values)
            # Todo: Find how to get the original labels back?  - AD
            rfcols = ["ReliefF feature {}".format(i) for i in range(x.shape[1])]
            number_df = pd.DataFrame(columns=rfcols, data=x,
                                     index=number_df.index)
        if n_pca_features:
            self.logger.info(
                "PCA running: retaining {} numerical features.".format(
                    n_pca_features))
            n_pca_features = PCA(n_components=n_pca_features)
            x = n_pca_features.fit_transform(number_df)
            # Todo: I don't know if there is a way to get labels for these - AD
            pcacols = ["PCA feature {}".format(i) for i in range(x.shape[1])]
            number_df = pd.DataFrame(columns=pcacols, data=x,
                                     index=number_df.index)

        return pd.concat([targets, number_df, object_df], axis=1)

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
                        if corr.loc[idx, target_key] > corr.loc[
                            feature, target_key]:
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

    def handle_na(self, df, max_na_frac=0.05, na_method='drop'):
        """
        First pass for handling cells wtihout values (null or nan). Additional
        preprocessing may be necessary as one column may be filled with
        median while the other with mean or mode, etc.

        Args:
            max_na_frac ([str]): after generating features, drop the columns
                that have null/na rows with more than this ratio.
            na_method (str): method of handling null rows.
                Options: "drop", "mode", ... (see pandas fillna method options)
        Returns:

        """
        self.logger.info(
            "Before handling na: {} samples, {} features".format(*df.shape))
        feats0 = set(df.columns)
        df = df.dropna(axis=1, thresh=int((1 - max_na_frac) * len(df)))
        if len(df.columns) < len(feats0):
            feats = set(df.columns)
            self.logger.info('These {} features were removed as they '
                             'had more than {}% missing values:\n{}'.format(
                len(feats0) - len(feats), max_na_frac * 100, feats0 - feats))
        if na_method == "drop":  # drop all rows that contain any null
            df = df.dropna(axis=0)
        else:
            df = df.fillna(method=na_method)
        self.logger.info(
            "After handling na: {} samples, {} features".format(*df.shape))
        return df
