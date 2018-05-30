import pandas as pd
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

"""
Functions for generating data to csv form from various resources, if it is not
already in csv form.
"""

def generate_mp(max_nsite=0, initial_structures=True):
    """
    Grabs all mp materials. This will return two csv files:
        * mp_nostruct.csv: All MP materials, not including structures (.005GB)
        * mp_all.csv: All MP materials, including structures (0.6 - 1.2 GB)

    Args:
        max_nsite (int): The maximum number of sites to include in the query.
        initial_structures (bool): If true, include the structures before
            relaxation.

    Returns:
        None
    """
    props = ['mpid', 'pretty_formula', 'e_above_hull', 'band_gap',
             'total_magnetization', 'elasticity.elastic_anisotropy',
             'elasticity.K_VRH', 'elasticity.G_VRH', 'structure']
    mpdr = MPDataRetrieval()
    mpdf = None
    for nsites in list(range(1, 101)) + [{'$gt': 100}]:
        if nsites==max_nsite:
            break
        print("Processing nsites= {}".format(nsites))
        df = mpdr.get_dataframe(criteria={'nsites': nsites},
                                properties=props,
                                index_mpid=True)
        if initial_structures:
            # prevent data limit API error using this conditional
            isdf = mpdr.get_dataframe(criteria={'nsites': nsites},
                                      properties=['initial_structure'],
                                      index_mpid=True),
            df = df.join(isdf, how='inner')
        if mpdf is None:
            mpdf = df
        else:
            mpdf = pd.concat([mpdf, df])
    mpdf.to_csv("sources/mp_all.csv")
    mpdf = mpdf.drop(['structure', 'initial_structure'], axis=1)
    mpdf.to_csv("sources/mp_nostruct.csv")


if __name__ == "__main__":
    generate_mp(initial_structures=False)