"""
This file makes the following benchmarking datasets:
    - steels

From matminer's dataset library.
"""

from matminer.datasets.dataset_retrieval import load_dataset



# Note the units are in MPa, NOT GPa
if __name__ == "__main__":
    df = load_dataset("steel_strength")
    df = df[["formula", "yield strength"]]
    df = df.rename(columns={"formula": "composition"})
    print(df)
    df.to_pickle("steels.pickle.gz")
