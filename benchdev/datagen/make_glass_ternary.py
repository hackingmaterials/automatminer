from matminer.datasets.dataset_retrieval import load_dataset
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval


import pandas as pd
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mpdr = MPDataRetrieval()


df = load_dataset("glass_ternary_landolt")

df = df.rename(columns={"formula": "composition"})
df = df[["composition", "gfa"]]

df["gfa"] = df["gfa"] == 0
df = df.reset_index(drop=True)

print(df)
print(df["gfa"].value_counts())
df.to_pickle("glass.pickle.gz")