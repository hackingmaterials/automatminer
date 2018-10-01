from matbench.data.load import load_double_perovskites_gap
from matbench.prepare import PrepareData


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
    prep = PrepareData(df_init,
                       targets=['gap gllbsc'],
                       ignore_cols=['A1', 'A2', 'B1', 'B2'])
    prep.auto_featurize()
    # prep.handle_na()

    vis = VisualizeData(prep.get_train_target())
    vis.targetted_visualize(target='gap gllbsc')