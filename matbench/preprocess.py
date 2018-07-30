import logging

from matbench.utils.utils import MatbenchError, setup_custom_logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import is_numeric_dtype
import pandas as pd

class PreProcess(object):
    """
    PreProcess has several methods to clean and prepare the data
    for visualization and training.

    Args:
        df (pandas.DataFrame): input data
        target_col (str): if set, the target column may be examined (e.g. to be
            numeric)
        max_colnull (float): after generating features, drop the columns that
            have null/na rows with more than this ratio. Note that there is an
            important trade-off here. this ratio is high, one may lose more
            features and if it is low one may lose more samples.
        loglevel (int): the level of output; e.g. logging.DEBUG
        logpath (str): the path to the logfile dir, current folder by default.
    """
    def __init__(self, df=None, target_col=None, max_colnull=0.05,
                 loglevel=logging.INFO, logpath='.'):
        self.df = df
        self.target_col = target_col
        self.max_colnull = max_colnull
        self.logger = setup_custom_logger(filepath=logpath, level=loglevel)
        self.scaler = None
        self.pca = None


    def preprocess(self, df=None, scale=False, pca=False, **kwargs):
        """
        A sequence of data pre-processing steps either through this class or
        sklearn.

        Args:
            scale (bool): whether to scale/normalize the data
            pca (bool): whether to use principal component analysis (PCA) to
                reduce the dimensions of the data.
            kwargs (dict): the keyword arguments that are specific to some
                of the preprocessing methods such as PCA

        Returns (pandas.DataFrame):
        """
        df = self._prescreen_df(df)
        df = self.handle_nulls(df, na_method=kwargs.pop('na_method', 'drop'))
        if scale:
            self.scaler = MinMaxScaler()
            df = self.scaler.fit_transform(df)
        if pca:
            self.pca = PCA(n_components=kwargs.pop('n_components', None))
            df = self.pca.fit_transform(df)
        if self.target_col:
            if not is_numeric_dtype(df[self.target_col]):
                raise MatbenchError('Target column "{}" must be numeric'.format(
                    self.target_col))

        # TODO: remove/modify the following once preprocessing methods for str/objects are implemented:
        # df = df.drop(list(df.columns[df.dtypes == object]), axis=1)
        for col in list(df.columns[df.dtypes == bool]):
            df[col] = df[col].apply(int)
        df = pd.get_dummies(df)
        df = df.apply(pd.to_numeric)
        return df


    def prune_highly_correlated_features(self, df=None, target_col=None,
                                         threshold=0.9):
        """
        Goes over the features and remove those that are cross correlated by
        more than threshold. Target_col must be specified!

        Args:
            target_col (str): the name of the target column/feature
            threshold (0<float<=1): if R is greater than this value, the
                feature that has lower correlation with the target is removed.

        Returns (pandas.DataFrame):
            the dataframe with the highly cross-correlated features removed.
        """


    def _prescreen_df(self, df):
        if df is None:
            df = self.df.copy(deep=True)
        return df


    def handle_nulls(self, df=None, max_colnull=None, na_method='drop'):
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
        df = self._prescreen_df(df)
        max_colnull = max_colnull or self.max_colnull
        feats0 = set(df.columns)
        df = df.dropna(axis=1, thresh=int((1-max_colnull)*len(df)))
        if len(df.columns) < len(feats0):
            feats = set(df.columns)
            self.logger.info('The following {} features were removed as they '
                             'had more than {}% missing values:\n{}'.format(
                len(feats0)-len(feats), max_colnull*100, feats0-feats))
        if na_method == "drop": # drop all rows that contain any null
            df = df.dropna(axis=0)
        else:
            df = df.fillna(method=na_method)
        return df
