"""
This file is modified from the default config files of the TPOT library.
It contains a customed dict of operators that we want to optimize using
genetic algorithm.

We can add/remove Regressors/Preprocessors/Selectors-related operators
to custom the operators to be optimized by tpot in the future. For instance,
the Preprocessors/Selectors-related procedures are currently taken care of
by the Preprocess class in automatminer, so we may consider to comment out the
related operators in the config_dicts (or use tpot instead of Preprocess
to optimize some procedures).

Check the TPOT documentation for information on the structure of config_dicts
"""

import numpy as np

tree_estimators = [20, 100, 200, 500, 1000]
tree_max_features = np.arange(0.05, 1.01, 0.1)
tree_learning_rates = [1e-2, 1e-1, 0.5, 1.0]
tree_max_depths = range(1, 11, 2)
tree_min_samples_split = range(2, 21, 3)
tree_min_samples_leaf = range(1, 21, 3)
tree_ensemble_subsample = np.arange(0.05, 1.01, 0.1)


TPOT_REGRESSOR_CONFIG = {
    # Regressors
    "sklearn.linear_model.ElasticNetCV": {
        "l1_ratio": np.arange(0.0, 1.01, 0.05),
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    "sklearn.ensemble.ExtraTreesRegressor": {
        "n_estimators": tree_estimators,
        "max_features": tree_max_features,
        "min_samples_split": tree_min_samples_split,
        "min_samples_leaf": tree_min_samples_leaf,
        "bootstrap": [True, False],
    },
    "sklearn.ensemble.GradientBoostingRegressor": {
        "n_estimators": tree_estimators,
        "loss": ["ls", "lad", "huber", "quantile"],
        "learning_rate": tree_learning_rates,
        "max_depth": tree_max_depths,
        "min_samples_split": tree_min_samples_split,
        "min_samples_leaf": tree_min_samples_leaf,
        "subsample": np.arange(0.05, 1.01, 0.05),
        "max_features": np.arange(0.05, 1.01, 0.05),
        "alpha": [0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
    },
    # 'sklearn.ensemble.AdaBoostRegressor': {
    #     'n_estimators': tree_estimators,
    #     'learning_rate': tree_learning_rates,
    #     'loss': ["linear", "square", "exponential"],
    #     'max_depth': tree_max_depths
    # },
    "sklearn.tree.DecisionTreeRegressor": {
        "max_depth": tree_max_depths,
        "min_samples_split": tree_min_samples_split,
        "min_samples_leaf": tree_min_samples_leaf,
    },
    "sklearn.neighbors.KNeighborsRegressor": {
        "n_neighbors": range(1, 101),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "sklearn.linear_model.LassoLarsCV": {"normalize": [True, False]},
    "sklearn.svm.LinearSVR": {
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "dual": [True, False],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "epsilon": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    },
    "sklearn.ensemble.RandomForestRegressor": {
        "n_estimators": tree_estimators,
        "max_features": np.arange(0.05, 1.01, 0.1),
        "min_samples_split": tree_min_samples_split,
        "min_samples_leaf": tree_min_samples_leaf,
        "bootstrap": [True, False],
    },
    "sklearn.linear_model.RidgeCV": {},
    # "xgboost.XGBRegressor": {
    #     "n_estimators": tree_estimators,
    #     "max_depth": tree_max_depths,
    #     "learning_rate": tree_learning_rates,
    #     "subsample": tree_ensemble_subsample,
    #     "min_child_weight": range(1, 21, 4),
    #     "nthread": [1],
    # },
    # Preprocesssors
    "sklearn.preprocessing.Binarizer": {"threshold": np.arange(0.0, 1.01, 0.05)},
    "sklearn.decomposition.FastICA": {"tol": np.arange(0.0, 1.01, 0.05)},
    "sklearn.cluster.FeatureAgglomeration": {
        "linkage": ["ward", "complete", "average"],
        "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"],
    },
    "sklearn.preprocessing.MaxAbsScaler": {},
    "sklearn.preprocessing.MinMaxScaler": {},
    "sklearn.preprocessing.Normalizer": {"norm": ["l1", "l2", "max"]},
    "sklearn.kernel_approximation.Nystroem": {
        "kernel": [
            "rbf",
            "cosine",
            "chi2",
            "laplacian",
            "polynomial",
            "poly",
            "linear",
            "additive_chi2",
            "sigmoid",
        ],
        "gamma": np.arange(0.0, 1.01, 0.05),
        "n_components": range(1, 11),
    },
    "sklearn.decomposition.PCA": {
        "svd_solver": ["randomized"],
        "iterated_power": range(1, 11),
    },
    "sklearn.preprocessing.PolynomialFeatures": {
        "degree": [2],
        "include_bias": [False],
        "interaction_only": [False],
    },
    "sklearn.kernel_approximation.RBFSampler": {"gamma": np.arange(0.0, 1.01, 0.05)},
    "sklearn.preprocessing.RobustScaler": {},
    "sklearn.preprocessing.StandardScaler": {},
    "tpot.builtins.ZeroCount": {},
    "tpot.builtins.OneHotEncoder": {
        "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
        "sparse": [False],
        "threshold": [10],
    },
    # Selectors
    "sklearn.feature_selection.SelectFwe": {
        "alpha": np.arange(0, 0.05, 0.001),
        "score_func": {"sklearn.feature_selection.f_regression": None},
    },
    "sklearn.feature_selection.SelectPercentile": {
        "percentile": range(1, 100),
        "score_func": {"sklearn.feature_selection.f_regression": None},
    },
    "sklearn.feature_selection.VarianceThreshold": {
        "threshold": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
    "sklearn.feature_selection.SelectFromModel": {
        "threshold": np.arange(0, 1.01, 0.05),
        "estimator": {
            "sklearn.ensemble.ExtraTreesRegressor": {
                "n_estimators": [100],
                "max_features": tree_max_features,
            }
        },
    },
}


TPOT_CLASSIFIER_CONFIG = {
    # Classifiers
    "sklearn.naive_bayes.GaussianNB": {},
    "sklearn.naive_bayes.BernoulliNB": {
        "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        "fit_prior": [True, False],
    },
    "sklearn.naive_bayes.MultinomialNB": {
        "alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        "fit_prior": [True, False],
    },
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": tree_max_depths,
        "min_samples_split": tree_min_samples_split,
        "min_samples_leaf": tree_min_samples_leaf,
    },
    "sklearn.ensemble.ExtraTreesClassifier": {
        "n_estimators": tree_estimators,
        "criterion": ["gini", "entropy"],
        "max_features": tree_max_features,
        "min_samples_split": tree_min_samples_split,
        "min_samples_leaf": tree_min_samples_leaf,
        "bootstrap": [True, False],
    },
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": tree_estimators,
        "criterion": ["gini", "entropy"],
        "max_features": tree_max_features,
        "min_samples_split": tree_min_samples_split,
        "min_samples_leaf": tree_min_samples_leaf,
        "bootstrap": [True, False],
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": tree_estimators,
        "learning_rate": tree_learning_rates,
        "max_depth": tree_max_depths,
        "min_samples_split": tree_min_samples_split,
        "min_samples_leaf": tree_min_samples_leaf,
        "subsample": tree_ensemble_subsample,
        "max_features": tree_max_features,
    },
    "sklearn.neighbors.KNeighborsClassifier": {
        "n_neighbors": range(1, 101),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "sklearn.svm.LinearSVC": {
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "dual": [True, False],
        "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
    },
    "sklearn.linear_model.LogisticRegression": {
        "penalty": ["l1", "l2"],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "dual": [True, False],
    },
    # "xgboost.XGBClassifier": {
    #     "n_estimators": tree_estimators,
    #     "max_depth": tree_max_depths,
    #     "learning_rate": tree_learning_rates,
    #     "subsample": tree_ensemble_subsample,
    #     "min_child_weight": range(1, 21),
    #     "nthread": [1],
    # },
    # Preprocesssors
    "sklearn.preprocessing.Binarizer": {"threshold": np.arange(0.0, 1.01, 0.05)},
    "sklearn.decomposition.FastICA": {"tol": np.arange(0.0, 1.01, 0.05)},
    "sklearn.cluster.FeatureAgglomeration": {
        "linkage": ["ward", "complete", "average"],
        "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"],
    },
    "sklearn.preprocessing.MaxAbsScaler": {},
    "sklearn.preprocessing.MinMaxScaler": {},
    "sklearn.preprocessing.Normalizer": {"norm": ["l1", "l2", "max"]},
    "sklearn.kernel_approximation.Nystroem": {
        "kernel": [
            "rbf",
            "cosine",
            "chi2",
            "laplacian",
            "polynomial",
            "poly",
            "linear",
            "additive_chi2",
            "sigmoid",
        ],
        "gamma": np.arange(0.0, 1.01, 0.05),
        "n_components": range(1, 11),
    },
    "sklearn.decomposition.PCA": {
        "svd_solver": ["randomized"],
        "iterated_power": range(1, 11),
    },
    "sklearn.preprocessing.PolynomialFeatures": {
        "degree": [2],
        "include_bias": [False],
        "interaction_only": [False],
    },
    "sklearn.kernel_approximation.RBFSampler": {"gamma": np.arange(0.0, 1.01, 0.05)},
    "sklearn.preprocessing.RobustScaler": {},
    "sklearn.preprocessing.StandardScaler": {},
    "tpot.builtins.ZeroCount": {},
    "tpot.builtins.OneHotEncoder": {
        "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
        "sparse": [False],
        "threshold": [10],
    },
    # Selectors
    "sklearn.feature_selection.SelectFwe": {
        "alpha": np.arange(0, 0.05, 0.001),
        "score_func": {"sklearn.feature_selection.f_classif": None},
    },
    "sklearn.feature_selection.SelectPercentile": {
        "percentile": range(1, 100),
        "score_func": {"sklearn.feature_selection.f_classif": None},
    },
    "sklearn.feature_selection.VarianceThreshold": {
        "threshold": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },
    "sklearn.feature_selection.RFE": {
        "step": np.arange(0.05, 1.01, 0.05),
        "estimator": {
            "sklearn.ensemble.ExtraTreesClassifier": {
                "n_estimators": [100],
                "criterion": ["gini", "entropy"],
                "max_features": tree_max_features,
            }
        },
    },
    "sklearn.feature_selection.SelectFromModel": {
        "threshold": np.arange(0, 1.01, 0.05),
        "estimator": {
            "sklearn.ensemble.ExtraTreesClassifier": {
                "n_estimators": [100],
                "criterion": ["gini", "entropy"],
                "max_features": tree_max_features,
            }
        },
    },
}
