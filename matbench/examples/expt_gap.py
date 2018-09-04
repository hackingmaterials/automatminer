import json

import matbench.data.load as loader
import numpy as np
import os
import pandas as pd
import pickle
from time import time
from matbench.analysis import Analysis
from matbench.automl.tpot_utils import TpotAutoml
from matbench.featurize import Featurize
from matbench.preprocess import PreProcess
from sklearn.model_selection import train_test_split

# inputs
loader_func = loader.load_expt_gap
LIMIT = 30
IGNORE_THESE_COLUMNS = []
TARGET = 'gap expt'
MODE = 'regression'
CALC_DIR = 'run_data'
TIMEOUT_MINS = None
GENERATIONS = 2
POPULATION_SIZE = 10
SCORING = 'r2'
SEED = 13
FEAT_FROM_FILE = False
TPOT_FROM_FILE = False
EXCLUDED_FEATURIZERS = ['CohesiveEnergy', 'AtomicPackingEfficiency',
                        'PartialRadialDistributionFunction',
                        'RadialDistributionFunction',
                        'CoulombMatrix',
                        'SineCoulombMatrix',
                        'OrbitalFieldMatrix',
                        'MinimumRelativeDistances',
                        'ElectronicRadialDistributionFunction']
FEATUREIZE_THESE_COLUMNS = ["formula"]
N_JOBS = 4
MULTIINDEX = True
if MULTIINDEX:
    TARGET = ('Input Data', TARGET)


# actual pipeline:
fname_base = loader_func.__name__[5:] # loader_func names LIKE load_%
np.random.seed(SEED)
df_init = loader_func()
if LIMIT and LIMIT<len(df_init):
    df_init = df_init.iloc[np.random.choice(len(df_init), LIMIT, replace=False)]

if not FEAT_FROM_FILE:
    featzer = Featurize(ignore_cols=IGNORE_THESE_COLUMNS,
                        exclude=EXCLUDED_FEATURIZERS,
                        multiindex=MULTIINDEX,
                        drop_featurized_col=True,
                        n_jobs=N_JOBS)

    df = featzer.featurize_columns(df_init,
                                   input_cols=FEATUREIZE_THESE_COLUMNS,
                                   guess_oxidstates=False)
    df.to_pickle(os.path.join(CALC_DIR, '{}_data.pickle'.format(fname_base)))
else:
    df = pd.read_pickle(os.path.join(CALC_DIR, '{}_data.pickle'.format(fname_base)))


prep = PreProcess(target=TARGET)
df = prep.preprocess(df)


X_train, X_test, y_train, y_test = train_test_split(
    df.drop(TARGET, axis=1), df[TARGET], random_state=SEED)

if not TPOT_FROM_FILE:
    print('start timing...')
    start_time = time()
    tpot = TpotAutoml(mode=MODE,
                      max_time_mins=TIMEOUT_MINS,
                      generations=GENERATIONS,
                      population_size=POPULATION_SIZE,
                      scoring=SCORING,
                      random_state=SEED,
                      feature_names=df.drop(TARGET, axis=1).columns,
                      n_jobs=4,
                      verbosity=2)
    tpot.fit(X_train, y_train)
    print('total fitting time: {} s'.format(time() - start_time))

    with open(os.path.join(CALC_DIR, '{}.pickle'.format(fname_base)), 'wb') as fm:
        pickle.dump(tpot.fitted_pipeline_, fm)
    with open(os.path.join(CALC_DIR, '{}_models.json'.format(fname_base)), 'w') as fj:
        json.dump(tpot.evaluated_individuals_, fj)
else:
    with open(os.path.join(CALC_DIR, '{}.pickle'.format(fname_base)), 'rb') as fm:
        tpot = TpotAutoml(mode=MODE,
                          feature_names=df.drop(TARGET, axis=1).columns)
        tpot.fitted_pipeline_ = pickle.load(fm)
    with open(os.path.join(CALC_DIR, '{}_models.json'.format(fname_base)), 'r') as fj:
        tpot.evaluated_individuals_ = json.load(fj)

print(tpot.predict(X_test))
top_scores = tpot.get_top_models(return_scores=True)
print('top cv scores:')
print(top_scores)
print('top models')
print(tpot.top_models)
test_score = tpot.score(X_test, y_test)
print('the best test score:')
print(test_score)


analysis = Analysis(tpot, X_train, y_train, X_test, y_test, MODE,
                   target=TARGET,
                   features=df.drop(TARGET, axis=1).columns,
                   test_samples_index=X_test.index,
                   random_state=SEED)

feature_importance = analysis.get_feature_importance(sort=True)
print('Top 15 feature importance')
print(list(feature_importance.items())[:15])