MatBench v0.1 benchmark
===================

Overview
------------

MatBench is an `ImageNet <http://www.image-net.org>`_ for materials science; a
set of 13 supervised, pre-cleaned, ready-to-use ML tasks for benchmarking and fair comparison. The tasks span a wide domain of
inorganic materials science applications.

.. image:: _static/matbench_pie_charts.png
   :alt: matbench
   :align: center
   :width: 600px

You can find details and results on the benchmark in our paper
`Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm <https://doi.org/10.1038/s41524-020-00406-3>`_. Please consider citing this paper if you use Matbench v0.1 for benchmarking, comparison, or prototyping.


Accessing the ML tasks
-----------------------

There are three ways to access the Matbench problems:

1. Programmatically, via the `matminer datasets repository <https://hackingmaterials.lbl.gov/matminer/dataset_summary.html>`_. Recommended for benchmarking and test usage. See the code examples in the following sections for details on this process.
2. Interactively, through the `Materials Project MPContribs-ML Deployment <https://ml.materialsproject.org>`_; links to each dataset are in the table below.
3. Via static download links (given in table).


Here's a full list of the 13 datasets in Matbench v0.1:

.. list-table::
   :align: left
   :header-rows: 1

   * - task name
     - target column (unit)
     - number of samples
     - task type
     - links
   * - :code:`matbench_dielectric`
     - :code:`n` (unitless)
     - 4764
     - regression
     - `download <https://ml.materialsproject.org/matbench_dielectric.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_dielectric/>`_
   * - :code:`matbench_expt_gap`
     - :code:`gap expt` (eV)
     - 4604
     - regression
     - `download <https://ml.materialsproject.org/matbench_expt_gap.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_expt_gap/>`_
   * - :code:`matbench_expt_is_metal`
     - :code:`is_metal` (unitless)
     - 4921
     - classification
     - `download <https://ml.materialsproject.org/matbench_expt_is_metal.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_expt_is_metal/>`_
   * - :code:`matbench_glass`
     - :code:`gfa` (unitless)
     - 5680
     - classification
     - `download <https://ml.materialsproject.org/matbench_glass.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_glass/>`_
   * - :code:`matbench_jdft2d`
     - :code:`exfoliation_en` (meV/atom)
     - 636
     - regression
     - `download <https://ml.materialsproject.org/matbench_jdft2d.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_jdft2d/>`_
   * - :code:`matbench_log_gvrh`
     - :code:`log10(G_VRH)` (log(GPa))
     - 10987
     - regression
     - `download <https://ml.materialsproject.org/matbench_log_gvrh.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_log_gvrh/>`_
   * - :code:`matbench_log_kvrh`
     - :code:`log10(K_VRH)` (log(GPa))
     - 10987
     - regression
     - `download <https://ml.materialsproject.org/matbench_log_kvrh.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_log_kvrh/>`_
   * - :code:`matbench_mp_e_form`
     - :code:`e_form` (eV/atom)
     - 132752
     - regression
     - `download <https://ml.materialsproject.org/matbench_mp_e_form.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_mp_e_form/>`_
   * - :code:`matbench_mp_gap`
     - :code:`gap pbe` (eV)
     - 106113
     - regression
     - `download <https://ml.materialsproject.org/matbench_mp_gap.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_mp_gap/>`_
   * - :code:`matbench_mp_is_metal`
     - :code:`is_metal` (unitless)
     - 106113
     - classification
     - `download <https://ml.materialsproject.org/matbench_mp_is_metal.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_mp_is_metal/>`_
   * - :code:`matbench_perovskites`
     - :code:`e_form` (eV, per unit cell)
     - 18928
     - regression
     - `download <https://ml.materialsproject.org/matbench_perovskites.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_perovskites/>`_
   * - :code:`matbench_phonons`
     - :code:`last phdos peak` (cm^-1)
     - 1265
     - regression
     - `download <https://ml.materialsproject.org/matbench_phonons.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_phonons/>`_
   * - :code:`matbench_steels`
     - :code:`yield strength` (MPa)
     - 312
     - regression
     - `download <https://ml.materialsproject.org/matbench_steels.json.gz>`_, `interactive <https://ml.materialsproject.cloud/matbench_steels/>`_



Leaderboard
------------

.. list-table::
   :align: left
   :header-rows: 1

   * - task name
     - verified top score (MAE or ROCAUC)
     - algorithm
     - is algorithm general purpose?
   * - :code:`matbench_dielectric`
     - 0.299
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_expt_gap`
     - 0.416 eV
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_expt_is_metal`
     - 0.92
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_glass`
     - 0.861
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_jdft2d`
     - 38.6 meV/atom
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_log_gvrh`
     - 0.0849 log(GPa)
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_log_kvrh`
     - 0.0679 log(GPa)
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_mp_e_form`
     - 95.2 MPa
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_mp_gap`
     - 95.2 MPa
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_mp_is_metal`
     - 95.2 MPa
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_perovskites`
     - 95.2 MPa
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_phonons`
     - 95.2 MPa
     - Automatminer v1.0.3.2019111
     - yes
   * - :code:`matbench_steels`
     - 95.2 MPa
     - Automatminer v1.0.3.2019111
     - yes


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


Benchmarking and reporting your algorithm
-----------------------------------------

Benchmarking on Matbench v0.1 is done exclusively with nested cross validation
(NCV). See more details on NCV on the :ref:`advanced_usage` page and the `original publication <https://doi.org/10.1038/s41524-020-00406-3>`_.

If you want to evaluate your own (algorithm outside of the Automatminer framework) and compare to the scores on this page, please use the following steps:

0. **Download the dataset programmatically through matminer (instructions above).** Note the dataset must be used in the exact order in which it was downloaded.
1. **Generating test folds:** Use the scikit-learn :code:`KFold` (5 splits, shuffled, random seed 18012019) for regression problems and :code:`StratifiedKFold` (5 splits, shuffled, random seed 18012019) for classification problems.
2. **For each fold**:
    a. Train, validate, and select your best model using this fold's set of training data **only**. After training and validating, **no modiications may be made to the model based on the test set of this fold**.
    b. Remove the target variable column from the test set. Use this model to predict the test set. **Note: this test data is for reporting only, and cannot be used for validation or training within this fold.**
    c. Record the mean MAE or ROC-AUC for each fold's test set. Save the test fold data.
    d. Save your model.
3. **Post your results for verification.** Make a post on `the discussion forum <https://matsci.org/c/matminer/>`_ with the tag [Matbench] in the title. Once your results are verified, your algorithm will appear on the leaderboard!


If you are benchmarking a general-purpose algorithm, please include results for all Matbench v0.1 datasets.