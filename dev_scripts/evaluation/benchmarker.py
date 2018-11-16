"""
This file will eventually hold a function that tests a mslearn
pipeline on a set of datasets for predictive power.
"""

from matminer.datasets.dataset_retrieval import load_dataset, get_available_datasets
# from matminer.datasets.convenience_loaders import

if __name__ == "__main__":
    df_piezo = load_dataset("piezoelectric_tensor")
    df_exgap = load_dataset("expt_gap")
    df_elastic = load_dataset("elastic_tensor_2015")
    df_glass = load_dataset("glass_binary")




