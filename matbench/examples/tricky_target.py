import matbench.data.load as loader
from time import time
from matbench.analysis import Analysis
from matbench.automl.tpot_utils import TpotAutoml
from matbench.featurize import Featurize
from matbench.preprocess import PreProcess
from sklearn.model_selection import train_test_split

# inputs
loader_func = loader.load_castelli_perovskites
LIMIT = 1000
IGNORE_THESE_COLUMNS = ['cbm', 'vbm']
TARGET = 'gap gllbsc'
MODE = 'regression'
TIMEOUT_MINS = None
GENERATIONS = 3
POPULATION_SIZE = 40
SCORING = 'r2'
RS = 13
EXCLUDED_FEATURIZERS = ['CohesiveEnergy', 'AtomicPackingEfficiency',
                        'PartialRadialDistributionFunction',
                        'RadialDistributionFunction',
                        'CoulombMatrix',
                        'SineCoulombMatrix',
                        'OrbitalFieldMatrix',
                        'MinimumRelativeDistances',
                        'ElectronicRadialDistributionFunction']
FEATUREIZE_THESE_COLUMNS = ["formula", "structure"]
MULTIINDEX = True
if MULTIINDEX:
    TARGET = ('Input Data', TARGET)


# actual pipeline:
df_init = loader_func()[:LIMIT]
featzer = Featurize(ignore_cols=IGNORE_THESE_COLUMNS,
                    exclude=EXCLUDED_FEATURIZERS,
                    multiindex=MULTIINDEX,
                    drop_featurized_col=True)

print(df_init.columns)
df = featzer.featurize_columns(df_init,
                               input_cols=FEATUREIZE_THESE_COLUMNS,
                               guess_oxidstates=True)

prep = PreProcess(target=TARGET)
df = prep.preprocess(df)


X_train, X_test, y_train, y_test = train_test_split(
    df.drop(TARGET, axis=1), df[TARGET])

print('start timing...')
start_time = time()
tpot = TpotAutoml(mode=MODE,
                  max_time_mins=TIMEOUT_MINS,
                  generations=GENERATIONS,
                  population_size=POPULATION_SIZE,
                  scoring=SCORING,
                  random_state=RS,
                  feature_names=df.drop(TARGET, axis=1).columns,
                  n_jobs=1,
                  verbosity=2)
tpot.fit(X_train, y_train)
print('total fitting time: {} s'.format(time() - start_time))

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
                   random_state=RS)

feature_importance = analysis.get_feature_importance(sort=True)
print('feature importance')
print(feature_importance)