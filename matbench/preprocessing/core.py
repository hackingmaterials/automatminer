"""
Preprocessing classes.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from matbench.base import LoggableMixin, DataframeTransformer
from matbench.preprocessing.feature_selection import TreeBasedFeatureReduction, \
    rebate

__authors__ = ["Alex Dunn <ardunn@lbl.gov>",
               "Alireza Faghaninia <alireza@lbl.gov>"]


class DataCleaner(DataframeTransformer, LoggableMixin):
    """
    Transform a featurized dataframe into an ML-ready dataframe.

    Args:
        scale (bool): If True, scales the numerical feature data. Data is scaled
            before one-hot encoding, if encoding is enabled.
        max_na_frac (float): The maximum fraction (0.0 - 1.0) of samples for a
            given feature allowed. Columns containing a higher nan fraction are
            dropped.
        na_method (str): How to deal with samples still containing nans after
            troublesome columns are already dropped. Default is 'drop'. Other
            options are from pandas.DataFrame.fillna: {‘backfill’, ‘bfill’,
            ‘pad’, ‘ffill’, None}
        encode_categories (bool): If True, retains features which are
            categorical (data type is string or object) and then
            one-hot encodes them. If False, drops them.
        encoding_method (str): choose a method for encoding the categorical
            variables. Current options: 'one-hot' and 'label'.
        drop_na_targets (bool): Drop samples containing target values which are
            na.
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default matbench logger will be
            used. If set to False, then no logging will occur.

    Attributes:
        The following attrs are set during fitting.

        dropped_features (list): The features which were dropped.
        retained_features (list): The features which were retained
        object_cols (list): The features identified as objects/categories
        number_cols (list): The features identified as numerical
        dropped_samples (pandas.DataFrame): A dataframe of samples to be dropped.
        df_numerical (pandas.DataFrame): A cleaned dataframe


    """

    def __init__(self, scale=False, max_na_frac=0.01, na_method='drop',
                 encode_categories=True, encoder='one-hot',
                 drop_na_targets=True,
                 logger=True):
        self._logger = self.get_logger(logger)
        self.scale = scale
        self.max_na_frac = max_na_frac
        self.na_method = na_method
        self.encoder = encoder
        self.encode_categories = encode_categories
        self.drop_na_targets = drop_na_targets

        # Attributes which will be set during transformation
        self.dropped_features = None
        self.retained_features = None
        self.object_cols = None
        self.number_cols = None
        self.dropped_samples = None
        self.df_numerical = None

    def fit(self, df, target):
        """
        Assign attributes before actually transforming. Useful if you want
        to see what the transformation will do before actually transforming.

        Args:
            df (pandas.DataFrame): Contains features and the target_key
            target (str): The name of the target in the dataframe

        Returns:

        """
        self.handle_na(df, target)
        self.to_numerical(df, target)
        return self

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
        X = StandardScaler().fit_transform(X) if self.scale else X
        return pd.concat([y, X], axis=1)

    def fit_transform(self, df, target):
        return self.transform(df, target)

    def handle_na(self, df, target):
        """
        First pass for handling cells wtihout values (null or nan). Additional
        preprocessing may be necessary as one column may be filled with
        median while the other with mean or mode, etc.

        Args:
            df (pandas.DataFrame): The dataframe containing features
            target (str): The key defining the ML target.

        Returns:
            (pandas.DataFrame) The cleaned df
        """
        self.logger.info("Before handling na: {} samples, {} features".format(
            *df.shape))

        # Drop targets containing na before further processing
        if self.drop_na_targets:
            clean_df = df.dropna(axis=0, how='any', subset=[target])
            self.dropped_samples = df[~df.index.isin(clean_df.index)]
            df = clean_df

        # Remove features failing the max_na_frac limit
        feats0 = set(df.columns)
        clean_df = df.dropna(axis=1,
                             thresh=int((1 - self.max_na_frac) * len(df)))
        self.dropped_features = [c for c in df.columns.values if
                                 c not in feats0]
        df = clean_df

        if len(df.columns) < len(feats0):
            feats = set(df.columns)
            n_feats = len(feats0) - len(feats)
            napercent = self.max_na_frac * 100
            feat_names = feats0 - feats
            self.logger.info('These {} features were removed as they had more '
                             'than {}% missing values:\n{}'.format(
                n_feats, napercent, feat_names))

        # Handle all rows that still contain any nans
        if self.na_method == "drop":
            clean_df = df.dropna(axis=0, how='any')
            self.dropped_samples = pd.concat(
                (df[~df.index.isin(clean_df.index)], self.dropped_samples),
                axis=0)
            df = clean_df
        else:
            df = df.fillna(method=self.na_method)
        self.logger.info("After handling na: {} samples, {} features".format(
            *df.shape))
        self.retained_features = df.columns.values.tolist()
        return df

    def to_numerical(self, df, target):
        """
        Transforms non-numerical columns to numerical columns which are
        machine learning-friendly.

        Args:
            df (pandas.DataFrame): The dataframe containing features
            target (str): The key defining the ML target.

        Returns:
            (pandas.DataFrame) The numerical df
        """

        self.number_cols = []
        self.object_cols = []
        for c in df.columns.values:
            try:
                df[c] = pd.to_numeric(df[c])
                self.number_cols.append(c)
            except (TypeError, ValueError):
                # The target is most likely strings which are not numeric.
                # Prevent target being encoded
                if c != target:
                    self.object_cols.append(c)

        number_df = df[self.number_cols]
        object_df = df[self.object_cols]
        if self.encode_categories and self.object_cols:
            if self.encoder == 'one-hot':
                object_df = pd.get_dummies(object_df).apply(pd.to_numeric)
            elif self.encoder == 'label':
                for c in object_df.columns:
                    object_df[c] = LabelEncoder().fit_transform(object_df[c])
                self.logger.warning(
                    'LabelEncoder used for categorical colums. For access to '
                    'the original labels via inverse_transform, encode '
                    'manually and set retain_categorical to False')
            return pd.concat([number_df, object_df], axis=1)
        else:
            return number_df


class FeatureReducer(DataframeTransformer, LoggableMixin):
    """
    Perform feature reduction on a clean dataframe.

    Args:
        reducers ((str)): The set of feature reduction operations to be
            performed on the data. The order of strings determines the order
            in which the reducers will be applied. Valid reducer strings are
            the following:
                'corr': Removes any cross-correlated features having corr.
                    coefficients larger than a threshold value. Retains
                    feature names.

                'tree': Perform iterative feature reduction via a tree-based
                    feature reduction, using ._feature_importances implemented
                    in sklearn. Retains feature names.

                'relief': Perform ReliefF feature reduction using the skrebate
                    package. Retains feature names.

                'pca': Perform Principal Component Analysis via
                    eigendecomposition. Note the feature labels will be renamed
                    to "PCA Feature X" if pca is present anywhere in the feature
                    reduction scheme!

            Example: Apply tree-based feature reduction, then pca:
                reducers = ('tree', 'pca')
        n_pca_features (int): The number of features to be retained by PCA, if
            PCA is present in the reducers
        n_rebate_features (int): The number of ReBATE relief features to be
            retained, if ReBATE is present in the reducers.
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default matbench logger will be
            used. If set to False, then no logging will occur.

    Attributes:
        The following attrs are set during fitting.

        removed_features (dict): The keys are the feature reduction methods
            applied. The values are the feature labels removed by that feature
            reduction method.
        retained_features (list): The features retained.
        reducer_params (dict): The keys are the feature reduction methods
            applied. The values are the parameters used by each feature reducer.
    """

    def __init__(self, reducers=('corr', 'tree'), n_pca_features=15,
                 n_rebate_features=15, logger=True):
        for reducer in reducers:
            if reducer not in ["corr", "tree", "rebate", "pca"]:
                raise ValueError(
                    "Reducer {} not found in known reducers!".format(reducer))

        self.reducers = reducers
        self.n_pca_features = n_pca_features
        self.n_rebate_features = n_rebate_features
        self._logger = self.get_logger(logger)
        self.removed_features = {}
        self.retained_features = []
        self.reducer_params = {}

    def fit(self, df, target):
        for r in self.reducers:
            if r == "corr":
                reduced_df = self.rm_correlated(df, target)

            # More advanced feature reduction methods
            else:
                X = df.drop(columns=[target])
                y = df[target]
                if r == "tree":
                    mode = "regression"
                    try:
                        pd.to_numeric(df[target])
                    except BaseException:
                        mode = "classification"
                    tbfr = TreeBasedFeatureReduction(mode=mode,
                                                     logger=self.logger)
                    reduced_df = tbfr.fit_transform(X, y)
                    self.reducer_params[r] = {
                        "importance_percentile": tbfr.importance_percentile,
                        "mode": tbfr.mode,
                        "random_state": tbfr.rs}
                elif r == "rebate":
                    self.logger.info(
                        "ReBATE MultiSURF running: retaining {} numerical "
                        "features.".format(self.n_rebate_features))
                    reduced_df = rebate(df, target,
                                        n_features=self.n_rebate_features)
                    self.logger.info(
                        "ReBATE MultiSURF completed: retained {} numerical "
                        "features.".format(len(reduced_df.columns)))
                    self.logger.debug(
                        "ReBATE MultiSURF gave the following "
                        "features".format(reduced_df.columns.tolist()))
                    self.reducer_params[r] = {"algo": "MultiSURF Algorithm"}

                # todo: PCA will not work with string columns!!!!!
                elif r == "pca":
                    self.logger.info("PCA running: retaining {} numerical "
                                     "features.".format(self.n_rebate_features))
                    matrix = PCA(
                        n_components=self.n_pca_features).fit_transform(
                        X.values, y.values)
                    pcacols = ["PCA {}".format(i) for i in
                               range(matrix.shape[1])]
                    reduced_df = pd.DataFrame(columns=pcacols, data=matrix,
                                              index=X.index)

                    self.logger.info(
                        "PCA completed: retained {} numerical "
                        "features.".format(len(reduced_df.columns)))

            retained = reduced_df.columns.values.tolist()
            removed = [c for c in df.columns.values if c not in retained]
            self.removed_features[r] = removed
            df = reduced_df

        self.retained_features = [c for c in df.columns.tolist() if c != target]
        return self

    def transform(self, df, target):
        return df[self.retained_features + [target]]

    def rm_correlated(self, df, target_key, R_max=0.95):
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
                            self.logger.debug(
                                '"{}" correlates strongly with '
                                '"{}"'.format(feature, idx))
                            self.logger.debug(
                                'removing "{}"...'.format(removed_feat))
                        if removed_feat == feature:
                            break
        if len(rm_feats) > 0:
            df = df.drop(rm_feats, axis=1)
            self.logger.info(
                'These {} features were removed due to cross '
                'correlation with the current features more than '
                '{}:\n{}'.format(len(rm_feats), R_max, rm_feats))
        return df


if __name__ == "__main__":
    pass
