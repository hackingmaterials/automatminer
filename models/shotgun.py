from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf
from pymatgen import Composition

from matbench.data.load_data import load_double_perovskites_gap

# user inputs:
nonna_ratio = 0.95

# load data
df_init, lumos = load_double_perovskites_gap(return_lumo=True)
df_init['composition'] = df_init['formula'].apply(Composition)

# featurize
featurizer = MultipleFeaturizer([
    cf.ElementProperty.from_preset(preset_name='matminer'),
    cf.IonProperty()
])

df = featurizer.featurize_dataframe(df_init, col_id='composition')
df = df.drop(['composition'], axis=1)

# clean
df_trimmed = df.dropna(axis=1, thresh=int(nonna_ratio*len(df))).dropna(axis=0)
for possible_index in ['formula']:
    if possible_index in df_trimmed:
        df_trimmed.set_index(possible_index)
        break
print(df_trimmed.head())

# train