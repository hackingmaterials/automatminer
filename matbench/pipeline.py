import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from matbench.data.load import load_mp
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import BagofBonds, BondFractions, StructuralHeterogeneity, StructureComposition, ChemicalOrdering, MaximumPackingEfficiency, DensityFeatures, GlobalSymmetryFeatures
from matminer.figrecipes.plot import PlotlyFig


# Try predict ehull from initial structure

print("Reading csv...")
df = load_mp('mp_all.csv').iloc[-100:]

rf = RandomForestRegressor()
cv = KFold(n_splits=10, shuffle=True)

ep = ElementProperty.from_preset("matminer")
bb = BagofBonds()
bf = BondFractions(approx_bonds=True)
sh = StructuralHeterogeneity()
# sc = StructureComposition(featurizer=ep)
co = ChemicalOrdering()
# mp = MaximumPackingEfficiency()
de = DensityFeatures()
gs = GlobalSymmetryFeatures()

structure_featurizers = [bb, bf, sh, co, de]

fls = []
for sf in structure_featurizers:
    print("\n Featurizing {}".format(sf.__class__.__name__))
    sf.fit_featurize_dataframe(df, 'initial structure', ignore_errors=True)
    fls += sf.feature_labels()

df = df[fls + ['e_hull', 'structure']]
df['formula'] = [f.composition.reduced_formula for f in df['structure']]
df = df.dropna(axis=1, thresh=0.8).dropna(axis=0)
dfx = df[fls]
X = dfx.values
y = df['e_hull']

print("Produced {} samples of {} features: {}".format(X.shape[0], X.shape[1], dfx.columns.values))
print("Cross validating...")
scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=4)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


pf_rf = PlotlyFig(x_title='DFT Energy Above Hull (eV)',
                  y_title='Random forest Energy Above Hull (eV)',
                  title='Random forest regression',
                  filename="rf_regression.html")

pf_rf.xy([(y, cross_val_predict(rf, X, y, cv=cv)), ([min(y), max(y)], [min(y), max(y)])],
      labels=df['formula'], modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], showlegends=False)
