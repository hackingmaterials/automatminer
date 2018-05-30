

class PreProcess(object):
    """
    PreProcess has several methods to clean and prepare the data
    for visualization and training.
    """
    def __init__(self):
        pass


    def handle_nulls(self, max_null_ratio=0.05, method='drop'):
        """

        Args:
            max_null_ratio ([str]): after generating features, drop the columns
                that have null/na rows with more than this ratio. Default 0.05
            method (str): method of handling null rows.
                Options: "drop", "mode", ... (see pandas fillna method options)
        Returns:

        """
        self.df = self.df.dropna(
                        axis=1, thresh=int((1-max_null_ratio)*len(self.df)))
        if method == "drop": # drop all rows that contain any null
            self.df = self.df.dropna(axis=0)
        else:
            self.df = self.df.fillna(method=method)