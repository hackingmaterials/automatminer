from matminer.datasets.convenience_loaders import load_double_perovskites_gap


class VisualizeData(object):
    """
    Takes in a featurized dataframe (e.g. output of PrepareData) and provide
    convenient tools to visualize the most relevant features and their
    relationship.
    """
    def __init__(self, df, target_df=None):
        if target_df:
            df = df.concat(target_df)
        self.df = df

    def targetted_visualize(self, target, ncols=10):
        """
        Visualize the top few features that correlate with the target feature

        Args:
            target (str): target column to be predicted/studied

        Returns:
        """
        print(self.df.corr)


if __name__ == "__main__":
    df_init, lumos = load_double_perovskites_gap(return_lumo=True)
