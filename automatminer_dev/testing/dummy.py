import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from automatminer.utils.ml import regression_or_classification
from automatminer.utils.ml import AMM_CLF_NAME, AMM_REG_NAME
from automatminer_dev.config import BENCHMARK_FULL_SET


benchmark_dir = os.environ["AMM_DATASET_DIR"]

for p in BENCHMARK_FULL_SET:
    pname = p["name"]
    print("Loading {}".format(pname))
    df = pd.read_pickle(os.path.join(benchmark_dir, p["data_pickle"]))
    target = p["target"]
    ltype = p["problem_type"]
    if ltype == AMM_REG_NAME:
        kf = KFold(n_splits=5, random_state=18012019, shuffle=True)
        estimator = DummyRegressor(strategy="mean")
        scoring = "neg_mean_absolute_error"
        multiplier = -1
    elif ltype == AMM_CLF_NAME:
        kf = StratifiedKFold(n_splits=5, random_state=18012019, shuffle=True)
        estimator = DummyClassifier(strategy="stratified")
        multiplier = 1
        scoring = "roc_auc"
    else:
        raise ValueError("problem type {} is not known.".format(ltype))

    cvs = cross_val_score(
        estimator, df.drop(columns=[target]), y=df[target], scoring=scoring, cv=kf
    )

    cvs = multiplier * cvs
    mean_cvs = np.mean(cvs)
    print(pname, mean_cvs)
