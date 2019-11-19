import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from automatminer.utils.ml import regression_or_classification
from automatminer.utils.ml import AMM_CLF_NAME, AMM_REG_NAME
from automatminer_dev.config import BENCHMARK_FULL_SET, GLASS, EXPT_IS_METAL, EXPT_GAP
from matminer.utils.io import load_dataframe_from_json


benchmark_dir = os.environ["AMM_DATASET_DIR"]

bmarks = BENCHMARK_FULL_SET
bmarks = [GLASS, EXPT_GAP, EXPT_IS_METAL]

for p in bmarks:
    pname = p["name"]
    print("Loading {}".format(pname))
    df = load_dataframe_from_json(os.path.join(benchmark_dir, p["data_file"]))
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


# for p in bmarks:
#     pname = p["name"]
#     print("Loading {}".format(pname))
#     df = load_dataframe_from_json(os.path.join(benchmark_dir, p["data_file"]))
#     target = p["target"]
#     ltype = p["problem_type"]
#
#     data = df[target]
#     mad = data.mad()
#     print(f"Mean average deviation for {p} is {mad}")