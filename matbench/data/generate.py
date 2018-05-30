import pandas as pd
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

"""
Functions for generating data to csv form from various resources, if it is not
already in csv form.
"""


def generate_mp(max_nsite=0):
    """
    Grabs all mp materials. Size of spreadsheet with all structures is
    about 0.6GB. Size of spreadsheet without structures is 0.005GB.
    Of course, if you include a prop like bandstructure, it will be much
    larger!
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
        if mpdf is None:
            mpdf = df
        else:
            mpdf = pd.concat([mpdf, df])
    mpdf.to_csv("mp_all.csv")
    mpdf = mpdf.drop('structure', axis=1)
    mpdf.to_csv("mp_nostruct.csv")


if __name__ == "__main__":
    generate_mp()