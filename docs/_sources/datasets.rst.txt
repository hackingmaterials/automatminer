MatBench benchmark
===================

Overview
------------

MatBench is an `ImageNet <http://www.image-net.org>`_ for materials science; a
set of 13 benchmarking ML problems for fair comparison, across a wide domain of
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
(and also via `Figshare <https://figshare.com/account/home#/projects/67337>`_).

You can download the datasets with the :code:`matminer.datasets.load_dataset`
function; the names of the datasets are named :code:`matbench-*` where :code:`*`
is the name of the benchmark problem.

Here's the MatBench benchmark for predicting refractive index (calculated with
DFPT) from crystal structure.

.. code-block:: python

    from matminer.datasets import load_dataset

    # Download and load the dataset
    # The dataset is stored locally after being downloaded the first time
    df = load_dataset("matbench_dielectric")

    # Check out the downloaded dataframe
    print(df)


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


Find all the MatBench problem names and info
`here <https://hackingmaterials.lbl.gov/matminer/dataset_summary.html>`_ (search
for "matbench").

*Note: Larger datasets will take several minutes to load.*