from matbench.utils.utils import MatbenchError
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import is_numeric_dtype


class PreProcess(object):
    """
    PreProcess has several methods to clean and prepare the data
    for visualization and training.

    Args:
        df (pandas.DataFrame): input data
        target_col (str): if set, the target column may be examined (e.g. to be
            numeric)
        max_colnull (float): after generating features, drop the columns that
            have null/na rows with more than this ratio.
    """
    def __init__(self, df=None, target_col=None, max_colnull=0.1):
        self.df = df
        self.target_col = target_col
        self.max_colnull = max_colnull


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
            pca = PCA(n_components=kwargs.pop('n_components', None))
            df = pca.fit_transform(df)
        if self.target_col:
            if not is_numeric_dtype(df[self.target_col]):
                raise MatbenchError('Target column "{}" must be numeric'.format(
                    self.target_col))
        return df


    def _prescreen_df(self, df):
        if df is None:
            df = self.df.copy(deep=True)
        return df


    def handle_nulls(self, df=None, max_colnull=None, na_method='drop'):
        """

        Args:
            max_colnull ([str]): after generating features, drop the columns
                that have null/na rows with more than this ratio.
            na_method (str): method of handling null rows.
                Options: "drop", "mode", ... (see pandas fillna method options)
        Returns:

        """
        df = self._prescreen_df(df)
        max_colnull = max_colnull or self.max_colnull
        df = df.dropna(axis=1, thresh=int((1-max_colnull)*len(df)))
        if na_method == "drop": # drop all rows that contain any null
            df = df.dropna(axis=0)
        else:
            df = df.fillna(method=na_method)
        return df
