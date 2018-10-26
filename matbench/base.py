class MatbenchError(BaseException):
    """
    Exception specific to matbench methods.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "MatbenchError : " + self.msg


class DataFrameTransformer:
    """
    Abstract base class for transforming dataframes in the same way
    BaseEstimator and TransformerMixin are abc's for sklearn matrix
    transformation.
    """

    def fit(self, df, target):
        raise NotImplementedError(
            "All dataframe transformers must implement fit.")

    def transform(self, df, target):
        raise NotImplementedError(
            "All dataframe transformers must implement transform.")

    def fit_transform(self, df, target):
        self.fit(df, target)
        return self.transform(df, target, )
