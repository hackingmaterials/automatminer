import numpy as np

"""
Derive a list of meta-features of a given dataset to get recommendation of 
featurizers. 

The meta-features serve as indicators of the dataset characteristics that may
affect the choice of featurizers.

Based on the meta-features and then benchmarking (i) featurizer availability, 
(ii) featurizer importance and (iii) featurizer computational budget of 
existing featurizers on a variety of datasets, we can get a sense of how these
featurizers perform for datasets with different meta-features, and then make 
some strategies of featurizer selection.

When given a new dataset, we can compute its meta-features, and then get the 
recommended featurizers based on the pre-defined strategies (e.g. one way is 
to get the L1 distances of meta-features of all pre-defined strategies and 
meta-features of the new dataset, then find some "nearest" strategies and make
an estimation of computational budget, and finally taking all these factors 
together to make a final recommendation of featurizers)

Current meta-features to be considered (many can be further added):
(for featurizer availability and featurizer importance)
    - is metallic or not:
    - is organic/molecule or not: 
        may need to call other packages e.g. deepchem or just fail this task as 
        we cannot directly support it in matminer.
    - contain all metallic elements or not:
    - contain transition metals or not: 
    - percent of transition metals: 
        e.g. can be used to determine whether to use featurizers such as Miedema 
        that is more applicable to transitional alloys.
    - number of elements present in the entire dataset: 
        e.g. can help to decided whether to use ChemicalSRO or Bob featurizers
        that can return O(N^2) features (increase rapidly with the number of 
        elements present)
    - is structures ordered or not: 
    - percent of ordered structures:
        e.g. can help to decide whether to use some featurizers that only apply
        to ordered structure such as GlobalSymmetryFeatures
    - is structure 1D, 2D or 3D?
    - regression or classification: 
        maybe some featurizers work well only for classification or only for 
        regression

(for featurizer computational budget)
    - number of compositions:
    - number of structures:
    - avg. number of elements in compositions/structures:
    - min. number of elements in compositions/structures:
    - max. number of elements in compositions/structures:
    - mean. number of elements in compositions/structures:
    - avg. number of atoms in structures:
    - min. number of atoms in structures:
    - max. number of atoms in structures: 
    - mean. number of atoms in structures:
        e.g. can be important for deciding on some extremely computational 
        expensive featurizers such as Voronoi-based ones or site statistics 
        ones such as SiteStatsFingerprint. They are assumed to be quite slow 
        if there are too many atoms in the structures.

Todo: split meta-features for compositions and structures, i.e. create two
    sets of meta-features? (there are pros and cons)

"""


class Metafeatures:
    pass

