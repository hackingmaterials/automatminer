import numpy as np
import pandas as pd
from matbench.utils.utils import setup_custom_logger, MatbenchError
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from pandas.api.types import is_numeric_dtype
from skrebate import ReliefF


class Preprocessing(object):
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
        self.scaler = None # make scaler available for inverse_transform

    def preprocess(self, df, target_key, scale=False, n_pca_features=None,
                   n_rebate_features=None, max_na_frac=0.01, na_method='drop',
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
        df = self.handle_na(df, max_na_frac=max_na_frac, na_method=na_method)
        df_numerical = self.to_numerical(df, retain_categorical=retain_categorical)
        y = df_numerical[target_key]
        X = df_numerical.drop(target_key, axis=1)
        if scale:
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)

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
                    raise MatbenchError('attempted PCA before data normalization!')
            self.logger.info(
                "PCA running: retaining {} numerical features.".format(
                    n_pca_features))
            n_pca_features = PCA(n_components=n_pca_features)
            matrix = n_pca_features.fit_transform(X)
            # Todo: I don't know if there is a way to get labels for these - AD
            pcacols = ["PCA {}".format(i) for i in range(matrix.shape[1])]
            X = pd.DataFrame(columns=pcacols, data=matrix, index=X.index)
        return pd.concat([X, y], axis=1)


    def to_numerical(self, df, retain_categorical=True,
                     encoding_method='one-hot'):
        """
        Transforms non-numerical columns to numerical columns which are
            machine learning-friendly.

        Args:
            df (pandas.DataFrame): The dataframe containing features
            retain_categorical (bool): If True, retains features which are
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
                object_cols.append(c)

        number_df = df[number_cols]
        object_df = df[object_cols]
        if retain_categorical:
            if encoding_method=='one-hot':
                object_df = pd.get_dummies(object_df).apply(pd.to_numeric)
            elif encoding_method=='label':
                for c in object_df.columns:
                    object_df[c] = LabelEncoder().fit_transform(object_df[c])
                self.logger.warning('LabelEncoder used for categorical colums '
                    'For access to the original labels via inverse_transform, '
                    'encode manually and set retain_categorical to False')

            return pd.concat([number_df, object_df], axis=1)
        else:
            return number_df


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