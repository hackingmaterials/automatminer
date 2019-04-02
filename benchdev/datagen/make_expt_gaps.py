from matminer.datasets.dataset_retrieval import load_dataset
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval


import pandas as pd
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mpdr = MPDataRetrieval()


df = load_dataset("expt_gap")

df = df.rename(columns={"formula": "composition"})

df["is_metal"] = df["gap expt"] == 0

df = df.drop(columns=["gap expt"])

print(df["is_metal"].value_counts())

df = df.reset_index(drop=True)

print(df)
df.to_pickle("expt_gaps.pickle.gz")
