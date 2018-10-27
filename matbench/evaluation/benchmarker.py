"""
This file will eventually hold a function that tests a matbench pipeline on a set of datasets for predictive power.
"""

from matminer.datasets.dataset_retrieval import load_dataset, available_datasets



if __name__ == "__main__":
    print(available_datasets())