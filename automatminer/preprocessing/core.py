"""
Top level preprocessing classes.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from automatminer.utils.package_tools import AutomatminerError, compare_columns, \
    check_fitted, set_fitted
from automatminer.utils.ml_tools import regression_or_classification
from automatminer.base import LoggableMixin, DataframeTransformer
from automatminer.preprocessing.feature_selection import TreeFeatureReducer, \
    rebate, lower_corr_clf

__authors__ = ["Alex Dunn <ardunn@lbl.gov>",
               "Alireza Faghaninia <alireza@lbl.gov>",
               "Alex Ganose <aganose@lbl.gov>"]


class DataCleaner(DataframeTransformer, LoggableMixin):
    """
    Transform a featurized dataframe into an ML-ready dataframe.

    Args:
        max_na_frac (float): The maximum fraction (0.0 - 1.0) of samples for a
            given feature allowed. Columns containing a higher nan fraction are
            dropped.
        encode_categories (bool): If True, retains features which are
            categorical (data type is string or object) and then
            one-hot encodes them. If False, drops them.
        encoder (str): choose a method for encoding the categorical
            variables. Current options: 'one-hot' and 'label'.
        drop_na_targets (bool): Drop samples containing target values which are
            na.
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default automatminer logger will
            be used. If set to False, then no logging will occur.

    Attributes:
            The following attrs are set during fitting.

        retained_features (list): The features which were retained
        object_cols (list): The features identified as objects/categories
        number_cols (list): The features identified as numerical
        fitted_df (pd.DataFrame): The fitted dataframe
        is_fit (bool): If true, this object has been fit to a dataframe

            The following attrs are set during fitting and/or transformation. Ie
            they are only relevant to the most recent transform.

        dropped_features (list): The features which were dropped.
        dropped_samples (pandas.DataFrame): A dataframe of samples to be dropped
    """

    def __init__(self, max_na_frac=0.01, encode_categories=True,
                 encoder='one-hot', drop_na_targets=True, logger=True):
        self._logger = self.get_logger(logger)
        self.max_na_frac = max_na_frac
        self.encoder = encoder
        self.encode_categories = encode_categories
        self.drop_na_targets = drop_na_targets
        self._reset_attrs()
        self.dropped_features = None
        self.object_cols = None
        self.number_cols = None
        self.fitted_df = None
        self.fitted_target = None
        self.dropped_samples = None
        self.is_fit = False

    @property
    def retained_features(self):
        """
        The features retained during fitting, which may be used to craft the
        dataframe during transform.

        Returns:
            (list): The list of features retained.
        """
        return self.fitted_df.columns.tolist()

    @set_fitted
    def fit(self, df, target, na_method="drop"):
        """
        Determine a sequence of preprocessing steps to clean a dataframe.

        Args:
            df (pandas.DataFrame): Contains features and the target_key
            target (str): The name of the target in the dataframe
            na_method (str): How to deal with samples still containing nans
                after troublesome columns are already dropped. Default is
                'drop'. Other options are from pandas.DataFrame.fillna:
                {‘bfill’, ‘pad’, ‘ffill’}, or 'ignore' to ignore nans.
                Alternatively, specify a value to replace the nans, e.g. 0.

        Returns: self
        """
        self.logger.info("Cleaning (fitting) with respect to samples with "
                         "na_method '{}'".format(na_method))
        if target not in df.columns:
            raise AutomatminerError(
                "Target {} must be contained in df.".format(target))

        self._reset_attrs()
        df = self.to_numerical(df, target)
        df = self.handle_na(df, target, na_method)
        self.fitted_df = df
        self.fitted_target = target
        return self

    @check_fitted
    def transform(self, df, target, na_method=0):
        """
        Apply the sequence of preprocessing steps determined by fit, with the
        option to change the na_method for samples.

        Args:
            df (pandas.DataFrame): Contains features and the target_key
            target (str): The name of the target in the dataframe
            na_method (str): How to deal with samples still containing nans
                after troublesome columns are already dropped. Default is
                'drop'. Other options are from pandas.DataFrame.fillna:
                {‘bfill’, ‘pad’, ‘ffill’}, or 'ignore' to ignore nans.
                Alternatively, specify a value to replace the nans, e.g. 0.

        Returns (pandas.DataFrame)
        """
        self.logger.info("Cleaning (transforming) with respect to samples with "
                         "na_method '{}'".format(na_method))

        if target != self.fitted_target:
            raise AutomatminerError(
                "The transformation target {} is not the same as the fitted "
                "target {}".format(
                    target, self.fitted_target))

        # We assume the two targets are the same from here on out
        df = self.to_numerical(df, target)
        df = self.handle_na(df, target, na_method, coerce_mismatch=True)

        # Ensure the order of columns is identical
        if target in df.columns:
            self.logger.info("Reordering columns...")
            df = df[self.fitted_df.columns]
        else:
            self.logger.info("Target not found in df columns. Ignoring...")
            reordered_cols = self.fitted_df.drop(columns=[target]).columns
            df = df[reordered_cols]
        return df

    def fit_transform(self, df, target):
        self.fit(df, target)
        return self.fitted_df

    def handle_na(self, df, target, na_method, coerce_mismatch=True):
        """
        First pass for handling cells without values (null or nan). Additional
        preprocessing may be necessary as one column may be filled with
        median while the other with mean or mode, etc.

        Args:
            df (pandas.DataFrame): The dataframe containing features
            target (str): The key defining the ML target.
            coerce_mismatch (bool): If there is a mismatch between the fitted
                dataframe columns and the argument dataframe columns, create
                and drop mismatch columns so the dataframes are matching. If
                False, raises an error. New columns are instantiated as all
                zeros, as most of the time this is a onehot encoding issue.
            na_method (str): How to deal with samples still containing nans
                after troublesome columns are already dropped. Default is
                'drop'. Other options are from pandas.DataFrame.fillna:
                {‘bfill’, ‘pad’, ‘ffill’}, or 'ignore' to ignore nans.
                Alternatively, specify a value to replace the nans, e.g. 0.

        Returns:
            (pandas.DataFrame) The cleaned df
        """
        self.logger.info("Before handling na: {} samples, {} features".format(
            *df.shape))

        # Drop targets containing na before further processing
        if self.drop_na_targets and target in df.columns:
            clean_df = df.dropna(axis=0, how='any', subset=[target])
            self.dropped_samples = df[~df.index.isin(clean_df.index)]
            self.logger.info("{} samples did not have target values. They were "
                             "dropped.".format(len(self.dropped_samples)))
            df = clean_df

        # Remove features failing the max_na_frac limit
        feats0 = set(df.columns)
        if not self.is_fit:
            self.logger.info("Handling na by max na threshold of {}."
                             "".format(self.max_na_frac))
            df = df.dropna(axis=1,
                           thresh=int((1 - self.max_na_frac) * len(df)))
            if len(df.columns) < len(feats0):
                feats = set(df.columns)
                n_feats = len(feats0) - len(feats)
                napercent = self.max_na_frac * 100
                feat_names = feats0 - feats
                self.logger.info(
                    'These {} features were removed as they had more '
                    'than {}% missing values:\n{}'.format(
                        n_feats, napercent, feat_names))
        else:
            mismatch = compare_columns(self.fitted_df, df, ignore=target)
            if mismatch["mismatch"]:
                self.logger.warning("Mismatched columns found in dataframe "
                                    "used for fitting and argument dataframe.")
                if not coerce_mismatch:
                    raise AutomatminerError("Mismatch between columns found in "
                                            "arg dataframe and dataframe used "
                                            "for fitting!")
                else:
                    self.logger.warning("Coercing mismatched columns...")
                    if mismatch["df1_not_in_df2"]:  # in fitted, not in arg
                        self.logger.warning(
                            "Assuming missing columns in argument df are "
                            "one-hot encoding issues. Setting to zero the "
                            "following new columns:\n{}".format(
                                mismatch["df1_not_in_df2"]))
                        for c in self.fitted_df.columns:
                            if c not in df.columns and c != target:
                                # Interpret as one-hot problems...
                                df[c] = np.zeros((df.shape[0]))
                    elif mismatch["df2_not_in_df1"]:  # arg cols not in fitted
                        self.logger.warning(
                            "Following columns are being dropped:\n{}".format(
                                mismatch["df2_not_in_df1"]))
                        df = df.drop(columns=mismatch["df2_not_in_df1"])

        self.dropped_features = [c for c in feats0 if
                                 c not in df.columns.values]

        # Handle all rows that still contain any nans
        if na_method == "drop":
            clean_df = df.dropna(axis=0, how='any')
            self.dropped_samples = pd.concat(
                (df[~df.index.isin(clean_df.index)], self.dropped_samples),
                axis=0)
            df = clean_df
        elif na_method == "ignore":
            pass
        elif na_method in ["ffill", "bfill", "pad", None]:
            df = df.fillna(method=na_method)
        else:
            df = df.fillna(na_method)
        self.logger.info("After handling na: {} samples, {} features".format(
            *df.shape))
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
        self.logger.info("Replacing infinite values with nan for easier "
                         "screening.")
        df = df.replace([np.inf, -np.inf], np.nan)
        self.number_cols = []
        self.object_cols = []
        for c in df.columns.values:
            try:
                if df[c].dtype == bool:
                    df[c] = df[c].astype(int)
                else:
                    df[c] = pd.to_numeric(df[c])
                if c != target:
                    self.number_cols.append(c)
            except (TypeError, ValueError):
                # The target is most likely strings which are not numeric.
                # Prevent target being encoded
                if c != target:
                    self.object_cols.append(c)

        if target in df.columns:
            target_df = df[[target]]
        else:
            target_df = pd.DataFrame()
        number_df = df[self.number_cols]
        object_df = df[self.object_cols]
        if self.encode_categories and self.object_cols:
            if self.encoder == 'one-hot':
                self.logger.info("One-hot encoding used for columns {}".format(
                    object_df.columns.tolist()))
                object_df = pd.get_dummies(object_df).apply(pd.to_numeric)
            elif self.encoder == 'label':
                self.logger.info("Label encoding used for columns {}".format(
                    object_df.columns.tolist()))
                for c in object_df.columns:
                    object_df[c] = LabelEncoder().fit_transform(object_df[c])
                self.logger.warning(
                    'LabelEncoder used for categorical colums. For access to '
                    'the original labels via inverse_transform, encode '
                    'manually and set retain_categorical to False')
            return pd.concat([target_df, number_df, object_df], axis=1)
        else:
            return pd.concat([target_df, number_df], axis=1)

    def _reset_attrs(self):
        """
        Reset all fit-dependent attrs.

        Returns:
            None
        """
        self.dropped_features = None
        self.object_cols = None
        self.number_cols = None
        self.fitted_df = None
        self.fitted_target = None
        self.dropped_samples = None
        self.is_fit = False


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

                'rebate': Perform ReliefF feature reduction using the skrebate
                    package. Retains feature names.

                'pca': Perform Principal Component Analysis via
                    eigendecomposition. Note the feature labels will be renamed
                    to "PCA Feature X" if pca is present anywhere in the feature
                    reduction scheme!

            Example: Apply tree-based feature reduction, then pca:
                reducers = ('tree', 'pca')
        n_pca_features (int, float): If int, the number of features to be
            retained by PCA. If float, the fraction of features to be retained
            by PCA once the dataframe is passed to it (i.e., 0.5 means PCA
            retains half of the features it is passed). PCA must be present in
            the reducers.
        n_rebate_features (int, float): If int, the number of ReBATE relief
            features to be retained. If float, the fraction of features to be
            retained by ReBATE once it is passed the dataframe (i.e., 0.5 means
            ReBATE retains half of the features it is passed). ReBATE must be
            present in the reducers.
        keep_features (list, None): A list of features that will not be removed.
            This option does nothing if PCA feature removal is present.
        remove_features (list, None): A list of features that will be removed.
            This option does nothing if PCA feature removal is present.
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default automatminer logger will
            be used. If set to False, then no logging will occur.

    Attributes:
        The following attrs are set during fitting.

        removed_features (dict): The keys are the feature reduction methods
            applied. The values are the feature labels removed by that feature
            reduction method.
        retained_features (list): The features retained.
        reducer_params (dict): The keys are the feature reduction methods
            applied. The values are the parameters used by each feature reducer.
    """

    def __init__(self, reducers=('corr', 'tree'), n_pca_features=0.3,
                 n_rebate_features=0.3, logger=True,
                 keep_features=None, remove_features=None):
        for reducer in reducers:
            if reducer not in ["corr", "tree", "rebate", "pca"]:
                raise ValueError(
                    "Reducer {} not found in known reducers!".format(reducer))

        self.reducers = reducers
        self.n_pca_features = n_pca_features
        self.n_rebate_features = n_rebate_features
        self._logger = self.get_logger(logger)
        self._keep_features = keep_features or []
        self._remove_features = remove_features or []
        self.removed_features = {}
        self.retained_features = []
        self.reducer_params = {}
        self._pca = None
        self._pca_feats = None

    @set_fitted
    def fit(self, df, target):
        self.logger.info("Feature reducer fitting to dataframe...")
        missing_remove_features = [c for c in self._remove_features
                                   if c not in df.columns]
        missing_keep_features = [c for c in self._keep_features
                                 if c not in df.columns]
        for features, name in [(missing_remove_features, 'remove'),
                               (missing_keep_features, 'keep')]:
            if features:
                self.logger.warning(
                    "Asked to {} some features that do not exist in the "
                    "dataframe. Skipping the following features:\n{}".format(
                        name, features))

        reduced_df = df
        for r in self.reducers:
            X = df.drop(columns=[target])
            y = df[target]
            if r == "corr":
                reduced_df = self.rm_correlated(df, target)
                reduced_df = reduced_df.drop(columns=[target])
            if r == "tree":
                tbfr = TreeFeatureReducer(
                    mode=regression_or_classification(y),
                    logger=self.logger)
                reduced_df = tbfr.fit_transform(X, y).copy(deep=True)
                self.reducer_params[r] = {
                    "importance_percentile": tbfr.importance_percentile,
                    "mode": tbfr.mode,
                    "random_state": tbfr.rs}
            elif r == "rebate":
                if isinstance(self.n_rebate_features, float):
                    self.logger.info(
                        "Retaining fraction {} of current {} features.".format(
                            self.n_rebate_features, df.shape[1] - 1))
                    self.n_rebate_features = int(df.shape[1] *
                                                 self.n_rebate_features)
                self.logger.info(
                    "ReBATE MultiSURF running: retaining {} numerical "
                    "features.".format(self.n_rebate_features))
                reduced_df = rebate(df, target,
                                    n_features=self.n_rebate_features)
                reduced_df = reduced_df.copy(deep=True)
                self.logger.info(
                    "ReBATE MultiSURF completed: retained {} numerical "
                    "features.".format(len(reduced_df.columns)))
                self.logger.debug(
                    "ReBATE MultiSURF gave the following "
                    "features: {}".format(reduced_df.columns.tolist()))
                self.reducer_params[r] = {"algo": "MultiSURF Algorithm"}
            elif r == "pca":
                if isinstance(self.n_pca_features, float):
                    self.logger.info("Retaining fraction {} of current "
                                     "{} features."
                                     "".format(self.n_pca_features,
                                               df.shape[1]))
                    self.n_pca_features = int(df.shape[1] *
                                              self.n_pca_features)
                self.logger.info("PCA running: retaining {} numerical "
                                 "features.".format(self.n_pca_features))
                self._pca = PCA(n_components=self.n_pca_features)
                self._pca.fit(X.values, y.values)
                matrix = self._pca.transform(X.values)
                pca_feats = ["PCA {}".format(i) for i in
                             range(matrix.shape[1])]
                self._pca_feats = pca_feats
                reduced_df = pd.DataFrame(columns=pca_feats, data=matrix,
                                          index=X.index)
                self.logger.info(
                    "PCA completed: retained {} numerical "
                    "features.".format(len(reduced_df.columns)))

            retained = reduced_df.columns.values.tolist()
            removed = [c for c in df.columns.values if c not in retained
                       and c != target]

            self.removed_features[r] = removed
            if target not in reduced_df:
                reduced_df.loc[:, target] = y.tolist()
            df = reduced_df

        all_removed = [c for r, rf in self.removed_features.items() for c in rf]
        all_kept = [c for c in df.columns.tolist() if c != target]
        save_from_removal = [c for c in self._keep_features if c in all_removed]
        for_force_removal = [c for c in self._remove_features if c in all_kept]

        if save_from_removal:
            self.logger.info("Saving features from removal. "
                             "Saved features:\n{}".format(save_from_removal))

        if for_force_removal:
            self.logger.info("Forcing removal of features. "
                             "Removed features: \n{}".format(for_force_removal))
            self.removed_features['manual'] = for_force_removal

        self.retained_features = [c for c in all_kept if c not in
                                  self._remove_features or c != target]
        return self

    @check_fitted
    def transform(self, df, target):
        self.logger.info("Feature reducer transforming dataframe...")
        X = df.drop(columns=target)
        for r, f in self.removed_features.items():
            if r == "pca":
                matrix = self._pca.transform(X)
                X = pd.DataFrame(columns=self._pca_feats, data=matrix,
                                 index=X.index)
            else:
                X = X.drop(columns=[c for c in f
                                    if c not in self._keep_features])
        X.loc[:, target] = df[target].values
        return X

    def rm_correlated(self, df, target, r_max=0.95):
        """
        A feature selection method that remove those that are cross correlated
        by more than threshold.

        Args:
            df (pandas.DataFrame): The dataframe containing features, target_key
            target (str): the name of the target column/feature
            r_max (0<float<=1): if R is greater than this value, the
                feature that has lower correlation with the target is removed.

        Returns (pandas.DataFrame):
            the dataframe with the highly cross-correlated features removed.
        """
        mode = regression_or_classification(df[target])
        corr = abs(df.corr())
        if mode == "regression":
            corr = corr.sort_values(by=target)
        rm_feats = []
        for feat in corr.columns:
            if feat == target:
                continue
            for idx, corval in zip(corr.index, corr[feat]):
                if np.isnan(corval):
                    break
                if idx == feat or idx in rm_feats:
                    continue
                else:
                    if corval >= r_max:
                        if mode == "regression":
                            if corr.loc[idx, target] > corr.loc[feat, target]:
                                removed_feat = feat
                            else:
                                removed_feat = idx
                        else:  # mode == "classification"
                            removed_feat = lower_corr_clf(df, target, feat, idx)
                        if removed_feat not in rm_feats:
                            rm_feats.append(removed_feat)
                            self.logger.debug(
                                '"{}" correlates strongly with '
                                '"{}"'.format(feat, idx))
                            self.logger.debug(
                                'removing "{}"...'.format(removed_feat))
                        if removed_feat == feat:
                            break
        if len(rm_feats) > 0:
            df = df.drop(rm_feats, axis=1)
            self.logger.info("{} features removed due to cross correlation more"
                             " than {}".format(len(rm_feats), r_max))
            self.logger.debug("Features removed by cross-correlation were: {}"
                              "".format(rm_feats))
        return df
