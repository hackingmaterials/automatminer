import numpy as np
import pandas as pd
from collections import OrderedDict
from mslearn.utils.utils import MatbenchError
from sklearn.model_selection import train_test_split


class Analytics(object):
    """
    Evaluates the importance of features, errors and uncertainty of a given
    machine learning model, bias, variance and tools assisting in manual
    inspection of errors and areas of improvements with maximized impact.
    We also take advantage of methods already available in lime ml evaluation
    package: https://github.com/marcotcr/lime

    Args:
        model (TPOTAutoML or sklearn Estimator): must have fit and predict
            methods.
        ...
        mode (str): options are "regression" or "classification"
        ...
    """
    def __init__(self, model, X_train, y_train, X_test, y_test, mode, target=None,
                 features=None, test_samples_index=None, random_state=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.mode = mode
        self.target = target
        if features is None:
            if isinstance(X_train, pd.DataFrame):
                features = X_train.columns.values
            else:
                features = ['feature {}'.format(i+1) for i in range(X_train.shape[1])]
        self.features = features
        self.test_samples_index = test_samples_index
        self.random_state = random_state
        self.false_positives = None
        self.false_negatives = None


    def from_dataframe_iid(self, df, target, mode,
                           train_size=0.75, test_size=0.25, random_state=None):
        """
        A helper function to be called only if 1) the model is not fit yet and
        train/test split hasn't been done AND 2) the user does not want to
        assign so many arguments necessary in class instantiation.

        * Note that if this method is called, the model is again fit to X_train
        to avoid data leakage. If the model is already trained, do not call this
        method.

        Args:
            df:
            target:
            mode:
            train_size:
            test_size:
            random_state:

        Returns:

        """
        y = np.array(df[target])
        X = df.drop(target, axis=1).values

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                             train_size=train_size,
                             test_size=test_size,
                             random_state=random_state)
        # re-train the model as it has been fit to part of the\ current X_test!
        self.model.fit(X_train, y_train)
        super (Analytics, self).__init__(X_train, y_train, X_test, y_test, mode,
                                         target=target, eatures=df.columns, test_samples_index=y_test.index)

    def _pre_screen_test_data(self, X_test, y_test):
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        return X_test, y_test

    def get_data_for_error_analysis(self, X_test=None, y_test=None, nmax=100):
        """
        Returns points with the wrong labels in case of a classification
        problem or high error in case of a regression problem. This can be
        used for further manual error analysis.

        Notes:
        * this method must be called after the fit.
        * upon calling this method false_positives and false_negatives
            attributes are also populated.

        Args:
            X_test (nxm numpy matrix where n is numer of samples and m is
                the number of features)
            y_test (nx1 numpy array): target labels/values
            nmax (int): maximum number of bad predictions returned

        Returns (pandas.DataFrame):
        """
        X_test, y_test = self._pre_screen_test_data(X_test, y_test)
        if X_test.shape[1] != len(list(self.features)):
            raise MatbenchError('The number of columns/features of X_test '
                                'do NOT match with the original features')
        predictions = self.model.predict(X_test)
        if self.mode == 'regression':
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        false_positives = []
        false_negatives = []
        for idx, pred in enumerate(predictions):
            if pred != y_test[idx]:
                if self.mode == 'classification':
                    if y_test[idx] > 0:
                        false_negatives.append(idx)
                    else:
                        false_positives.append(idx)
                elif self.mode == 'regression':
                    if abs(pred - y_test[idx]) >= rmse:
                        if pred < y_test[idx]:
                            false_negatives.append(idx)
                        else:
                            false_positives.append(idx)
        wrong_pred_idx = false_positives + false_negatives
        if len(wrong_pred_idx) > nmax:
            wrong_pred_idx = np.random.choice(wrong_pred_idx, nmax,
                                              replace=False)
        df = pd.DataFrame(X_test, columns=self.features,
                          index=self.test_samples_index)
        if isinstance(y_test, pd.Series):
            y_name = y_test.name
            y_test = np.array(y_test)
        else:
            y_name = 'target'
        df['{}_true'.format(y_name)] = y_test
        df['{}_predicted'.format(y_name)] = predictions
        self.false_negatives = df.iloc[false_negatives]
        self.false_positives = df.iloc[false_positives]
        df = df.iloc[wrong_pred_idx]
        return df

    def get_feature_importance(self, X_test=None, y_test=None, sort=False):
        X_test, y_test = self._pre_screen_test_data(X_test, y_test)
        pred0 = (self.model.predict(X_test)).astype(float)
        dfX = pd.DataFrame(X_test, columns=self.features)
        featance = OrderedDict({feat: 0.0 for feat in dfX})
        for feat in dfX:
            for sgn in [-1.0, +1.0]:
                dfX_perturbed = dfX.copy()
                dfX_perturbed[feat] = dfX_perturbed[feat] + sgn*dfX[feat].std()
                pred = (self.model.predict(dfX_perturbed.values)).astype(float)
                rmse = np.sqrt(np.mean((pred - pred0) ** 2))
                featance[feat] += rmse
            featance[feat] /= 2.0
        normalization_sum = sum(featance.values())
        for feat in featance:
            featance[feat] /= normalization_sum
        if sort:
            featance = OrderedDict(sorted(featance.items(),
                                          key=lambda x:x[1],
                                          reverse=True))
        self.feature_importance = featance
        return self.feature_importance


if __name__ == '__main__':
    from matminer.datasets import load_dataset

    from mslearn.pipeline import MatPipe

    df = load_dataset('elastic_tensor_2015')
    df = df[["formula", "K_VRH"]]
    df = df.rename({"formula": "composition"}, axis=1)

    fitted_pipeline = MatPipe(time_limit_mins=5).fit(df, "K_VRH")

    print("Done fitting")
