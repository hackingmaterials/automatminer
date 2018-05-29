from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf
from pymatgen import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

from matbench.data.load_data import load_double_perovskites_gap

# user inputs:
nonna_ratio = 0.95
ignore_columns = ['A1', 'A2', 'B1', 'B2']
target = 'gap gllbsc'

# load data
df_init, lumos = load_double_perovskites_gap(return_lumo=True)
if target not in df_init:
    raise ValueError('target feature {} not in the data!'.format(target))

df_init['composition'] = df_init['formula'].apply(Composition)

# featurize
featurizer = MultipleFeaturizer([
    cf.ElementProperty.from_preset(preset_name='matminer'),
    cf.IonProperty()
])

df = featurizer.featurize_dataframe(df_init, col_id='composition')
df = df.drop(['composition'], axis=1)

# clean
df = df.drop(ignore_columns, axis=1)
df = df.dropna(axis=1, thresh=int(nonna_ratio*len(df))).dropna(axis=0)

possible_indexes = ['formula']
for possible_index in possible_indexes:
    if possible_index in df:
        df = df.set_index(possible_index)
        break
# print(df_trimmed.head())

# encode categorical vars
print(df.select_dtypes(include=['object']))
print(df.head())

# train
y = df[target]
X = df.drop(target, axis=1)

rfr = RandomForestRegressor(n_jobs=-1)

rfr.fit(X, y)
kf = KFold(n_splits=10, random_state=0)
score = cross_val_score(rfr, X, y, cv=kf, scoring="neg_mean_squared_error")
score_r2 = cross_val_score(rfr, X, y, cv=kf, scoring="r2")

print('rmse score', score)
print('r2 score', score_r2)