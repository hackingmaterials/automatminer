import logging
import numpy as np
import pandas as pd

from matbench.utils.utils import MatbenchError, setup_custom_logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import is_numeric_dtype
from skrebate import ReliefF


class Preprocess(object):
    """
    PreProcess has several methods to clean and prepare the data
    for visualization and training.

    Args:
        df (pandas.DataFrame): input data
        target (str): if set, the target column may be examined (e.g. to be
            numeric)
        max_colnull (float): after generating features, drop the columns that
            have null/na rows with more than this ratio. Note that there is an
            important trade-off here. this ratio is high, one may lose more
            features and if it is low one may lose more samples.
        loglevel (int): the level of output; e.g. logging.DEBUG
        logpath (str): the path to the logfile dir, current folder by default.
    """

    def __init__(self, loglevel=logging.INFO, logpath='.'):
        self.logger = setup_custom_logger(filepath=logpath, level=loglevel)

    def preprocess(self, df, target_key, scale=False, n_pca_features=None,
                   n_rebate_features=None, na_method='drop'):
        """
        A sequence of data pre-processing steps either through this class or
        sklearn.

        Args:
            df (pandas.DataFrame): Contains features and the target_key
            target_key (str): The name of the target in the dataframe
            scale (bool): whether to scale/normalize the data
            n_pca_features (int or None): Number of features to select with
                principal component analysis (PCA). None or 0 avoids running
                PCA.
            n_rebate_features (int or None): Use the EpitasisLab ReBATE feature
                selection algorithm to reduce the dimensions of the data. None
                or 0 avoids running ReBATE.

        Returns (pandas.DataFrame
        """

        # Remove na rows including those where target=na
        df = self.handle_na(df, na_method=na_method)
        df = self.prune_correlated_features(df, target_key)

        targets = df[target_key].copy(deep=True)
        features = df.drop(columns=target_key)

        if scale:
            features.values = MinMaxScaler().fit_transform(features)

        if n_rebate_features:
            rf = ReliefF(n_features_to_select=n_rebate_features, n_jobs=-1)
            x = rf.fit_transform(features.values, targets.values)
            # Todo: Find how to get the original labels back?  - AD
            rfcols = ["ReliefF feature {}".format(i) for i in x.shape[1]]
            features = pd.DataFrame(columns=rfcols, data=x,
                                    index=features.index)
        if n_pca_features:
            n_pca_features = PCA(n_components=n_pca_features)
            x = n_pca_features.fit_transform(features)
            # Todo: I don't know if there is a way to get labels for these - AD
            pcacols = ["PCA feature {}".format(i) for i in x.shape[1]]
            features = pd.DataFrame(columns=pcacols, data=x,
                                    index=features.index)

        if target_key is not None:
            if not is_numeric_dtype(targets.values):
                targets = targets.astype(str, copy=False)

        # Boolean casting to ints
        # TODO: This might not work with numpy types, haven't checked - AD
        for col in list(features.columns[features.dtypes == bool]):
            features[col] = features[col].apply(int)
        features = pd.get_dummies(features).apply(pd.to_numeric)
        features[target_key] = targets
        return features

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

    def handle_na(self, df, max_colnull=None, na_method='drop'):
        """
        First pass for handling cells wtihout values (null or nan). Additional
        preprocessing may be necessary as one column may be filled with
        median while the other with mean or mode, etc.

        Args:
            max_colnull ([str]): after generating features, drop the columns
                that have null/na rows with more than this ratio.
            na_method (str): method of handling null rows.
                Options: "drop", "mode", ... (see pandas fillna method options)
        Returns:

        """
        self.logger.info(
            "pre handle_na: {} samples, {} features".format(*df.shape))
        feats0 = set(df.columns)
        df = df.dropna(axis=1, thresh=int((1 - max_colnull) * len(df)))
        if len(df.columns) < len(feats0):
            feats = set(df.columns)
            self.logger.info('These {} features were removed as they '
                             'had more than {}% missing values:\n{}'.format(
                len(feats0) - len(feats), max_colnull * 100, feats0 - feats))
        if na_method == "drop":  # drop all rows that contain any null
            df = df.dropna(axis=0)
        else:
            df = df.fillna(method=na_method)
        self.logger.info(
            "post handle_na: {} samples, {} features".format(*df.shape))
        return df
