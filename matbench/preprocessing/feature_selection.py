from matbench.utils.utils import setup_custom_logger, MatbenchError
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, \
    GradientBoostingClassifier, GradientBoostingRegressor


class TreeBasedFeatureReduction(object):
    """
    Provides different methods for feature selection/reduction mainly before
    the production model is fit hence being under preprocessing.

    Args:
        importance_percentile (float): the selected percentile of the features
            sorted (descending) based on their importance.
    """
    def __init__(self, mode, importance_percentile=0.95, loglevel=None, logpath='.'):
        self.mode = mode
        self.logger = setup_custom_logger(filepath=logpath, level=loglevel)
        self.importance_percentile = importance_percentile
        self.selected_features = None

    def get_top_features(self, feat_importance):
        selected_feats = []
        frac = 0.0
        for feat in feat_importance:
            selected_feats.append(feat[0])
            frac += feat[1]
            if frac >= self.importance_percentile:
                break
        return selected_feats

    def fit(self, X, y, tree='rf', recursive=True):
        """

        Args:
            X (pandas.DataFrame):
            y:
            tree:
            recursive:

        Returns:

        """
        m0 = len(X.columns)
        if isinstance(tree, str):
            if tree.lower() in ['rf', 'random forest', 'randomforest']:
                if self.mode.lower() in ['classification', 'classifier']:
                    tree = RandomForestClassifier()
                else:
                    tree = RandomForestRegressor()
            elif tree.lower() in ['gb', 'gbt', 'gradiet boosting']:
                if self.mode.lower() in ['classification', 'classifier']:
                    tree = GradientBoostingClassifier()
                else:
                    tree = GradientBoostingRegressor()
            else:
                raise MatbenchError('Unsupported tree_type {}!'.format(tree))
        m_curr = 0 # current number of top/important features
        m_prev = len(X.columns)
        while m_curr < m_prev:
            tree.fit(X, y)
            fimportance = sorted(zip(X.columns, tree.feature_importances_),
                                 key=lambda x: x[1], reverse=True)
            tfeats = self.get_top_features(fimportance)
            m_curr = len(tfeats)
            m_prev = len(X.columns)
            self.logger.debug('nfeatures: {}->{}'.format(len(X.columns), m_curr))
            X = X[tfeats]
            if not recursive:
                break
        self.logger.info('Finished one_tree_reduction reducing the number of '
                         'features from {} to {}'.format(m0, len(tfeats)))
        self.selected_features = tfeats

    def transform(self, X, y=None):
        if self.selected_features is None:
            raise MatbenchError('Please call fit first')
        return X[self.selected_features]