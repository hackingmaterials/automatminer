from matbench.utils.utils import setup_custom_logger, MatbenchError
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import check_cv


class TreeBasedFeatureReduction:
    """
    Tree-based feature reduction tools based on sklearn models that have
        the .feature_importances_ attribute.

    Args:
        mode (str): "regression" or "classification"
        importance_percentile (float): the selected percentile of the features
            sorted (descending) based on their importance.
        random_state (int): relevant if non-deterministic algorithms such as
            random forest are used.
    """
    def __init__(self, mode, importance_percentile=0.95, loglevel=None,
                 logpath='.', random_state=0):
        self.mode = mode
        self.logger = setup_custom_logger(filepath=logpath, level=loglevel)
        self.importance_percentile = importance_percentile
        self.selected_features = None
        self.rs = random_state

    def get_top_features(self, feat_importance):
        """
        Simple function to through a sorted list of features and select top
            percentiles.

        Args:
            feat_importance ([(str, float)]): a sorted list of
                (feature, importance) tuples

        Returns ([str]): list of the top * percentile of features. * determined
            by importance_percentile argument.

        """
        selected_feats = []
        frac = 0.0
        for feat in feat_importance:
            selected_feats.append(feat[0])
            frac += feat[1]
            if frac >= self.importance_percentile:
                break
        return selected_feats

    def get_reduced_features(self, tree_model, X, y, recursive=True):
        """
        Gives a reduced list of feature names given a tree-based model that
            has the .feature_importances_ attribute.

        Args:
            tree_model (instantiated sklearn tree-based model):
            X (pandas.dataframe):
            y (pandas.Series or numpy.ndarray): the target column
            recursive (bool):

        Returns ([str]): list of the top * percentile of features. * determined
            by importance_percentile argument.
        """
        m_curr = 0  # current number of top/important features
        m_prev = len(X.columns)
        while m_curr < m_prev:
            tree_model.fit(X, y)
            fimportance = sorted(zip(X.columns, tree_model.feature_importances_),
                                 key=lambda x: x[1], reverse=True)
            tfeats = self.get_top_features(fimportance)
            m_curr = len(tfeats)
            m_prev = len(X.columns)
            self.logger.debug(
                'nfeatures: {}->{}'.format(len(X.columns), m_curr))
            X = X[tfeats]
            if not recursive:
                break
        return tfeats

    def fit(self, X, y, tree='rf', recursive=True, cv=5):
        """
        Fits to the data (X) and target (y) to determine the selected_features.

        Args:
            X (pandas.DataFrame): input data, note that numpy matrix is NOT
                accepted since the X.columns is used for feature names
            y (pandas.Series or np.ndarray): list of outputs used for fitting
                the tree model
            tree (str or instantiated sklearn tree-based model): if a model is
                directly fed, it must have the .feature_importances_ attribute
            recursive (bool): whether to recursively reduce the features (True)
                or just do it once (False)
            cv (int or CrossValidation): sklearn's cross-validation with the
                same options (int or actual instantiated CrossValidation)

        Returns (None):
            sets the class attribute .selected_features
        """
        m0 = len(X.columns)
        if isinstance(tree, str):
            if tree.lower() in ['rf', 'random forest', 'randomforest']:
                if self.mode.lower() in ['classification', 'classifier']:
                    tree = RandomForestClassifier(random_state=self.rs)
                else:
                    tree = RandomForestRegressor(random_state=self.rs)
            elif tree.lower() in ['gb', 'gbt', 'gradiet boosting']:
                if self.mode.lower() in ['classification', 'classifier']:
                    tree = GradientBoostingClassifier(random_state=self.rs)
                else:
                    tree = GradientBoostingRegressor(random_state=self.rs)
            else:
                raise MatbenchError('Unsupported tree_type {}!'.format(tree))

        cv = check_cv(cv=cv, y=y, classifier=is_classifier(tree))
        all_feats = []
        for train, test in cv.split(X, y, groups=None):
            Xtrn = X.iloc[train]
            ytrn = y[train]
            all_feats += self.get_reduced_features(tree, Xtrn, ytrn, recursive)
        # take the union of selected features of each fold
        self.selected_features = list(set(all_feats))
        self.logger.info('Finished tree-based feature reduction of {} intial '
                         'features to {}'.format(m0, len(self.selected_features)))

    def transform(self, X, y=None):
        """
        Transforms the data with the subset of features determined after
            calling the fit method on the data.

        Args:
            X (pandas.DataFrame): input data, note that numpy matrix is NOT
                accepted since the X.columns is used for feature names
            y (placeholder): ignored input (for consistency in notation)

        Returns (pandas.DataFrame): the data with reduced number of features.
        """
        if self.selected_features is None:
            raise MatbenchError('The fit method should be called first!')
        return X[self.selected_features]

    def citations(self):
        return []

    def implementors(self):
        return ['Alireza Faghaninia']