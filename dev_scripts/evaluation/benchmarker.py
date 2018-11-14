"""
This file will eventually hold a function that tests a mslearn
pipeline on a set of datasets for predictive power.
"""

from matminer.datasets import load_dataset, get_available_datasets


if __name__ == "__main__":
    print(get_available_datasets())
