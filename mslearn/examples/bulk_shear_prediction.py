from math import sqrt, floor
from time import sleep

from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from mslearn.data.load import load_tehrani_superhard_mat
from mslearn.featurization import AutoFeaturizer
from mslearn.preprocessing import DataCleaner, FeatureReducer
from mslearn.automl.adaptors import TPOTAdaptor

"""
This script reproduces the results found by Tehrani et al in the following:
pubs.acs.org/doi/suppl/10.1021/jacs.8b02717/suppl_file/ja8b02717_si_001.pdf

We first train an SVM model as described in the above paper using identical
training data and model parameters. We then compare their results to the 
performance of our own model auto generated using the matsci-learn pipeline 
with features generated using the matminer software package.
"""

# Suppress warnings for readability
import warnings
warnings.filterwarnings("ignore")

# TEHRANI REPRODUCTION
# Load in training data and separate out features from targets
df = load_tehrani_superhard_mat(data="engineered_features")

target_labels = ["bulk_modulus", "shear_modulus"]
features = df.drop(target_labels, axis=1).values

# Normalize and scale data to work better with SVM model
features = normalize(features)
features = scale(features)

# Define a cross validation scheme and model to test
kfr = KFold(n_splits=10, shuffle=True, random_state=42)
model = SVR(gamma=.01, C=10)

# For each target train a model and evaluate with cross validation
for target_label in target_labels:
    target = df[target_label].values

    r2_scores = []
    rmse_scores = []
    prediction_list = []
    target_list = []
    for train_index, test_index in tqdm(
            kfr.split(features, target),
            desc="Cross validation progress for {}".format(target_label),
            total=10):
        train_features = features[train_index]
        test_features = features[test_index]

        train_target = target[train_index]
        test_target = target[test_index]

        model.fit(train_features, train_target)
        prediction = model.predict(test_features)
        r2_scores.append(r2_score(test_target, prediction))
        rmse_scores.append(sqrt(mean_squared_error(test_target, prediction)))

        prediction_list += [item for item in prediction]
        target_list += [item for item in test_target]

    bounds = [0, max(target_list) + 20]
    plt.scatter(target_list, prediction_list)
    plt.title("{} actual vs. predicted".format(target_label))
    plt.xlim(bounds)
    plt.ylim(bounds)
    x = np.linspace(0, bounds[1] + 20, 10000)
    plt.plot(x, x, color="black")
    plt.ylabel("Cross validated predictions for {}".format(target_label))
    plt.xlabel("Actual values for {}".format(target_label))
    plt.show()

    tqdm.write("RMSE for {}: {}".format(
        target_label, sum(rmse_scores) / len(rmse_scores)
    ))
    tqdm.write("R^2 for {}: {}".format(
        target_label, sum(r2_scores) / len(r2_scores)
    ))

    sleep(1)

# COMPARE TO MATBENCH
df = load_tehrani_superhard_mat(data="basic_descriptors")

df = df.drop(["formula", "material_id", "shear_modulus",
              "initial_structure"], axis=1)
traindf = df.iloc[:floor(.8 * len(df))]
testdf = df.iloc[floor(.8 * len(df)):]
target = "bulk_modulus"

# Get top-level transformers
autofeater = AutoFeaturizer()
cleaner = DataCleaner()
reducer = FeatureReducer()
learner = TPOTAdaptor("regression", max_time_mins=5)

# Fit transformers on training data
traindf = autofeater.fit_transform(traindf, target)
traindf = cleaner.fit_transform(traindf, target)
traindf = reducer.fit_transform(traindf, target)
learner.fit(traindf, target)

# Apply the same transformations to the testing data
testdf = autofeater.transform(testdf, target)
testdf = cleaner.transform(testdf, target)
testdf = reducer.transform(testdf, target)
testdf = learner.predict(testdf, target)    #predict validation data
print(testdf)
