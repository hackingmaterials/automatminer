import numpy as np
from matbench.analysis import Analysis
import matbench.data.load as loader
from matbench.featurize import Featurize
from matbench.preprocess import PreProcess
from matminer import PlotlyFig
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# inputs
loader_func = loader.load_boltztrap_mp
LIMIT = 5
IGNORE_THESE_COLUMNS = []
TARGET = 'gap gllbsc'
RS = 24
MODE = 'regression'
EXCLUDED_FEATURIZERS = ['CohesiveEnergy', 'AtomicPackingEfficiency',
                        'PartialRadialDistributionFunction']
FEATUREIZE_THESE_COLUMNS = ["formula", "structure"]
MULTIINDEX = True
if MULTIINDEX:
    TARGET = ('Input Data', TARGET)


# actual pipeline:
df_init = loader_func()[:LIMIT]
featzer = Featurize(df_init,
                    ignore_cols=IGNORE_THESE_COLUMNS,
                    exclude=EXCLUDED_FEATURIZERS,
                    multiindex=MULTIINDEX)

df_init.to_csv('test.csv')
df = featzer.featurize_columns(df_init,
                               input_cols=FEATUREIZE_THESE_COLUMNS,
                               guess_oxidstates=True)


##****** It works up to this point I get this error at preprocess at pruning: KeyError: ('Input Data', 'gap gllbsc')

prep = PreProcess(target=TARGET)
df = prep.preprocess(df)

print(df.head())
df.to_csv('test.csv')

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(TARGET, axis=1), df[TARGET])

model = RandomForestRegressor(n_estimators=100,
                              bootstrap=False,
                              max_features=0.8,
                              min_samples_leaf=1,
                              min_samples_split=4,
                              random_state=RS)


model.fit(X_train.values, y_train.values)

analysis = Analysis(model, X_train, y_train, X_test, y_test, MODE,
                   target=TARGET,
                   features=df.drop(TARGET, axis=1).columns,
                   test_samples_index=X_test.index,
                   random_state=RS)

x = list(analysis.get_feature_importance(sort=False).values())
y = model.feature_importances_
lr = linregress(x, y)
xreg = np.linspace(0.0, round(max(x),2), num=2)
yreg = lr.intercept + xreg * lr.slope

print('correlation, r={}'.format(lr.rvalue))
print('p-value, p={}'.format(lr.pvalue))

pf = PlotlyFig(
    title='Comparison of feature importances in predicting expt. gap',
    x_title='Analysis.feature_importance (Variance Sensitivity Analysis)',
    y_title='RandomForestRegressor.feature_importances_')
pf.xy([(x, y), (xreg, yreg)],
      labels=analysis.features,
      modes=['markers', 'line'],
      showlegends=False)