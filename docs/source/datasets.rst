MatBench benchmark
===================

Overview
------------

MatBench is an `ImageNet <http://www.image-net.org>`_ for materials science; a
set of 13 supervised ML tasks for benchmarking and fair comparison spanning a wide domain of
inorganic materials science applications.

.. image:: _static/matbench_pie_charts.png
   :alt: matbench
   :align: center
   :width: 600px

Details on the benchmark are coming soon in a publication we have submitted.
Stay tuned for more details on the evaluation procedure, best scores, and more!

For now, you can still access the benchmark datasets. See the "Accessing MatBench"
section for more info.



Accessing MatBench
------------------

We have made the MatBench benchmark publicly available via the `matminer
datasets repository <https://hackingmaterials.lbl.gov/matminer/dataset_summary.html>`_
and through the `Materials Project MPContribs-ML Deployment <https://ml.materialsproject.org>`_.
All the Matbench datasets begin with :code:`matbench_`.


Here's a full list of the 13 datasets in Matbench v0.1:

.. list-table::
   :align: left
   :header-rows: 1

   * - dataset name
     - target column
     - number of samples
     - task type
     - download link
   * - :code:`matbench_dielectric`
     - :code:`n`
     - 4764
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_expt_gap`
     - :code:`gap expt`
     - 4604
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_expt_is_metal`
     - :code:`is_metal`
     - 4921
     - classification
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_glass`
     - :code:`gfa`
     - 5680
     - classification
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_jdft2d`
     - :code:`exfoliation_en`
     - 636
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_log_gvrh`
     - :code:`log10(G_VRH)`
     - 10987
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_log_kvrh`
     - :code:`log10(K_VRH)`
     - 10987
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_mp_e_form`
     - :code:`e_form`
     - 132752
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_mp_gap`
     - :code:`gap pbe`
     - 106113
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_mp_is_metal`
     - :code:`is_metal`
     - 106113
     - classification
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_perovskites`
     - :code:`e_form`
     - 18928
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_phonons`
     - :code:`last phdos peak`
     - 1265
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_
   * - :code:`matbench_steels`
     - :code:`yield strength`
     - 312
     - regression
     - `link <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_


Getting dataset info
--------------------

You can get more info (such as the meaning of column names, brief cleaning
procedures, etc.) on a dataset with :code:`matminer.datasets.get_all_dataset_info`:

.. code-block:: python

    from matminer.datasets import get_all_dataset_info

    # Get dataset info from matminer
    info = get_all_dataset_info("matbench_steels")

    # Check out the info about the dataset.
    print(info)


.. code-block:: text

    Dataset: matbench_steels
    Description: Matbench v0.1 dataset for predicting steel yield strengths from chemical composition alone. Retrieved from Citrine informatics. Deduplicated.
    Columns:
        composition: Chemical formula.
        yield strength: Target variable. Experimentally measured steel yield strengths, in GPa.
    Num Entries: 312
    Reference: https://citrination.com/datasets/153092/
    Bibtex citations: ['@misc{Citrine Informatics,\ntitle = {Mechanical properties of some steels},\nhowpublished = {\\url{https://citrination.com/datasets/153092/},\n}']
    File type: json.gz
    Figshare URL: https://ml.materialsproject.org/matbench_steels.json.gz


You can also view all the Matbench datasets on the matminer
`Dataset Summary page <https://hackingmaterials.lbl.gov/matminer/dataset_summary.html>`_ (search
for "matbench").


(Down)loading datasets
-----------------------

While you can download the zipped json datasets via the download links above, we
recommend using matminer's tools to load datasets. Matminer intelligently manages the
dataset downloads in its central folder and provides methods for robustly loading dataframes containing
pymatgen primitives such as structures.

You can load the datasets with the :code:`matminer.datasets.load_dataset`
function; the function accepts the dataset name as an argument.
Here's an example of loading the Matbench task for predicting refractive index (calculated with
DFPT) from crystal structure.

.. code-block:: python

    from matminer.datasets import load_dataset

    # Download and load the dataset
    # The dataset is stored locally after being downloaded the first time
    df = load_dataset("matbench_dielectric")

    # Check out the downloaded dataframe
    print(df)


.. code-block:: text

                                                  structure         n
    0     [[4.29304147 2.4785886  1.07248561] S, [4.2930...  1.752064
    1     [[3.95051434 4.51121437 0.28035002] K, [4.3099...  1.652859
    2     [[-1.78688104  4.79604117  1.53044621] Rb, [-1...  1.867858
    3     [[4.51438064 4.51438064 0.        ] Mn, [0.133...  2.676887
    4     [[-4.36731958  6.8886097   0.50929706] Li, [-2...  1.793232
                                                     ...       ...
    4759  [[ 2.79280881  0.12499663 -1.84045389] Ca, [-2...  2.136837
    4760  [[0.         5.50363806 3.84192106] O, [4.7662...  2.690619
    4761  [[0. 0. 0.] Ba, [ 0.23821924  4.32393487 -0.35...  2.811494
    4762  [[0.         0.18884638 0.        ] K, [0.    ...  1.832887
    4763  [[0. 0. 0.] Cs, [2.80639641 2.80639641 2.80639...  2.559279
    [4764 rows x 2 columns]


This loads the dataframe in this format:

:code:`df` (:code:`matbench_dielectric`)

.. list-table::
   :align: left
   :header-rows: 1

   * - :code:`structure`
     - :code:`n`
   * - :code:`<structure object>`
     - 1.752064
   * - :code:`<structure object>`
     - 1.652859
   * - :code:`<structure object>`
     - 1.867858
   * - ...
     - ...


*Note: Larger datasets will take several minutes to load.*