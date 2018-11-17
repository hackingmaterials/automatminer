"""
This file will eventually hold a function that tests a mslearn
pipeline on a set of datasets for predictive power.


3 computational / 3 experimental
2 classification / 4 regression
3 small datasets / 3 large datasets

"""

from matminer.datasets.dataset_retrieval import load_dataset, get_available_datasets, get_dataset_column_description
# from matminer.datasets.convenience_loaders import

if __name__ == "__main__":
    # print(get_available_datasets())
    # df_piezo = load_dataset("piezoelectric_tensor")      # 941    regression      computational  hard      (predict max of piezoelectric tensor)
    # df_exgap = load_dataset("expt_gap")                  # 6,354  classification  experimental   moderate  (predict metal vs nonmetal)
    df_elastic = load_dataset("elastic_tensor_2015")     # 1,181  regression      computational  easy      (predict bulk modulus)
    # df_glass = load_dataset("glass_binary")              # 5,959  classification  experimental   moderate  (predict if metallic glass forms)
    # df_steel = load_dataset("steel_strength")            # 371    regression      experimental   moderate  (predict tensile strength of steels)
    # df_boltz = load_dataset("boltztrap_mp")              # 8,924  regression      computational  hard      (predict effective masses)
    # df_mp = load_dataset("mp_all")                       # 70,000 regression      computational  hard
    # df_exform = load_dataset("expt_formation_enthalpy")  # 1,276  regression      experimental   moderate  (predict formation enthalpy)

    print(df_elastic)
    for column in df_steel.columns:
        print(column, get_dataset_column_description('steel_strength', column))



