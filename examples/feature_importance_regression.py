import numpy as np
from automatminer.analytics import Analytics
from matminer.datasets.convenience_loaders import load_expt_gap
from automatminer.featurize import Featurize
from automatminer.preprocess import PreProcess
from matminer import PlotlyFig
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# inputs
target = 'gap expt'
RS = 24
mode = 'regression'
MULTIINDEX = True
if MULTIINDEX:
    target = ('Input Data', target)

df_init = load_expt_gap()
featzer = Featurize(exclude=['CohesiveEnergy', 'AtomicPackingEfficiency'],
                    multiindex=MULTIINDEX)

df = featzer.featurize_formula(df_init,
                               featurizers='all',
                               guess_oxidstates=False)

prep = PreProcess(target=target)
df = prep.preprocess(df)

print(df.head())
df.to_csv('test.csv')

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(target, axis=1), df[target])

model = RandomForestRegressor(n_estimators=100,
                              bootstrap=False,
                              max_features=0.8,
                              min_samples_leaf=1,
                              min_samples_split=4,
                              random_state=RS)


model.fit(X_train.values, y_train.values)
print('test score:')
print(model.score(X_test, y_test))

analysis = Analytics(model, X_train, y_train, X_test, y_test, mode,
                     target=target,
                     features=df.drop(target, axis=1).columns,
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
    x_title='Analytics.feature_importance (Variance Sensitivity Analysis)',
    y_title='RandomForestRegressor.feature_importances_')
pf.xy([(x, y), (xreg, yreg)],
      labels=analysis.features,
      modes=['markers', 'line'],
      showlegends=False)