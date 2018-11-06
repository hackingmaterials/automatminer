import numpy as np
from mslearn.analysis import Analysis
from mslearn.data.load import load_glass_formation
from mslearn.featurize import Featurize
from mslearn.preprocess import PreProcess
from matminer import PlotlyFig
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# inputs
target = 'gfa'
RS = 24
mode = 'classification'
MULTIINDEX = True
if MULTIINDEX:
    target = ('Input Data', target)

df_init = load_glass_formation(phase="ternary").drop('phase', axis=1)
featzer = Featurize(preset_name='deml',
                    exclude=['CohesiveEnergy', 'AtomicPackingEfficiency'],
                    multiindex=MULTIINDEX)
df = featzer.featurize_formula(df_init,
                               featurizers='all',
                               guess_oxidstates=False)

prep = PreProcess(target=target)
df = prep.preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(target, axis=1), df[target])

model = RandomForestClassifier(n_estimators=100,
                              random_state=RS)


model.fit(X_train.values, y_train.values)
print('test score:')
print(model.score(X_test, y_test))

analysis = Analysis(model, X_train, y_train, X_test, y_test, mode,
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
    title='Comparison of feature importances in predicting glass formation',
    x_title='Analysis.feature_importance (Variance Sensitivity Analysis)',
    y_title='RandomForestClassifier.feature_importances_')
pf.xy([(x, y), (xreg, yreg)],
      labels=analysis.features,
      modes=['markers', 'line'],
      showlegends=False)