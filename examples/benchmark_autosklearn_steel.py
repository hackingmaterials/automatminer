import os
import pstats
from time import time
from cProfile import Profile

import pandas as pd
from automatminer.automl.autosklearn_utils import AutoSklearnML
from matminer.datasets.convenience_loaders import load_steel_strength
from automatminer.featurize import Featurize
from automatminer.preprocess import PreProcess

data_name = "steel"
target = "yield strength"
timelimit_secs = 7200
rs = 29

feature_output_path = "example_data/matbench_data/featurized_data/"
model_tmp_path = r'example_data/matbench_data/autosklearn_output/tmp/'
model_output_path = r'example_data/matbench_data/autosklearn_output/output/'

feature_output_file = \
    os.path.join(feature_output_path,
                 "{}_all_featurized_data.csv".format(data_name))

if os.path.exists(feature_output_file):
    df = pd.read_csv(feature_output_file, index_col=0)
else:
    df_init = load_steel_strength()[['formula', target]]

    prof = Profile()
    prof.enable()

    featzer = Featurize()
    df_feats = featzer.featurize_formula(df_init, featurizers="all")
    prep = PreProcess(max_colnull=0.1)
    df = prep.preprocess(df_feats)

    prof.create_stats()
    print("featurize time:\n")
    pstats.Stats(prof).strip_dirs().sort_stats("time").print_stats(5)

    if os.path.exists(feature_output_path):
        print("output path: {} exists!".format(feature_output_path))
    else:
        os.makedirs(feature_output_path)
        print("create output path: {} successful!".format(feature_output_path))

    prof.dump_stats(
        os.path.join(feature_output_path,
                     "cProfile_for_featurize_{}.log".format(data_name)))

    df.to_csv(feature_output_file)

X = df.drop(target, axis=1).values
y = df[target]

output_folder = os.path.join(model_output_path, data_name, target)
tmp_folder = os.path.join(model_tmp_path, data_name, target)
autosklearnml = AutoSklearnML(X, y,
                              dataset_name="{}_{}".format(data_name,
                                                          target),
                              time_left_for_this_task=timelimit_secs,
                              per_run_time_limit=int(timelimit_secs/10),
                              ml_memory_limit=2048,
                              exclude_estimators=["random_forest"],
                              include_preprocessors=["no_preprocessing", ],
                              resampling_strategy='cv',
                              resampling_strategy_arguments={'folds': 5},
                              output_folder=output_folder,
                              tmp_folder=tmp_folder,
                              random_state=rs)

prof = Profile()
prof.enable()
start_time = time()

auto_regressor = autosklearnml.regression()

prof.create_stats()
print("featurize time:\n")
pstats.Stats(prof).strip_dirs().sort_stats("time").print_stats(5)

prof.dump_stats(
    os.path.join(output_folder,
                 "cProfile_for_autosklearn_{}.log".format(data_name)))

print(auto_regressor.get_models_with_weights())
print(auto_regressor.sprint_statistics())
print('total fitting time: {} s'.format(time() - start_time))

