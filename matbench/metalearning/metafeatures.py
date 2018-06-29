import numpy as np

"""
Derive a list of meta-features of a given dataset to get recommendation of 
featurizers. 

The relationship of featurizer availability (e.g. some featurizers only apply 
to a certain subset of compositions or structures), featurizer importance and 
computation budget will be considered in making the decision.

Current meta-features to be considered:
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
    - is structure ordered or not: 
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
        e.g. can be used to determine whether to use featurizers that are based
        on first calculating the 

    
Based on deriving these pre-defined meta-features and benchmarking the 
featurizer availability, featurizer importance and featurizer computational 
budget on existing datasets or based on our understanding, we can get a good 
sense of how those featurizers in matminer perform, for datasets with different 
characteristics, and then make strategies of featurizer selection.

When given a new dataset, we can compute its meta-featurizers, then get the 
recommended featurizers based on the pre-defined strategies (e.g. one way is 
to get the L1 distances of meta-features of all pre-defined strategies and 
meta-features of the new dataset, then find some "nearest" strategies, make
an estimation of computational budget, and finally taking all these factors 
together to make a final recommendation of featurizers)

Todo: split meta-features for compositions and structures, i.e. create two
    sets of meta-features? (there are pros and cons)

"""


class Metafeatures:
    pass

