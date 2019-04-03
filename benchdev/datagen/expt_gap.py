"""
This file makes the following benchmarking datasets:
    - expt_gap
    - expt_is_metal

From matminer's dataset library.
"""
from matminer.datasets.dataset_retrieval import load_dataset


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = load_dataset("expt_gap")
df = df.rename(columns={"formula": "composition"})
df.to_pickle("expt_gap.pickle.gz")
print(df)
df["is_metal"] = df["gap expt"] == 0
df = df.drop(columns=["gap expt"])
print(df["is_metal"].value_counts())
df = df.reset_index(drop=True)
print(df)
df.to_pickle("expt_is_metal.pickle.gz")
