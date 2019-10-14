Advanced Usage
==================

Running a benchmark
--------------------

**Introduction to benchmarking**

Automatminer can be used for benchmarking ML performance on materials
problems in a standardized fashion. A common example use case is comparing one
published method to another; another use is getting a rough idea how an
Automatminer model will generalize to making "real" predictions. To mitigate
unfair model advantages from biased splits or hyperparameter tuning,
Automatminer uses nested cross validation with identical
outer splits for benchmarking:

.. image:: _static/cv_nested.png
   :alt: server
   :align: center
   :width: 600px

Nested CV is analagous to using multiple hold-out test sets.

*Note: Nested CV is a computationally expensive benchmarking procedure!*

**Usage**

:code:`MatPipe` has a :code:`benchmark` method which can be used for
automatically benchmarking a pipeline on a dataset. Once you have your
data loaded in a dataframe, the procedure is:

1. Define a k-fold cross validation scheme (to use as outer test folds).

2. Use the :code:`benchmark` method of :code:`MatPipe` to get predictions for
each outer fold

3. Use your scoring function of choice to evaluate each fold.

.. code-block:: python

    from sklearn.model_evaluation import KFold

    # We recommend KFold for regression problems and StratifiedKFold
    # for classification
    kf = KFold(n_splits=5, shuffle=True)

    from automatminer.pipeline import MatPipe

    pipe = MatPipe.from_preset("express")
    predicted_folds = pipe.benchmark(my_df, "my_property", kf)

:code:`benchmark` returns a list of the predicted test folds (i.e., your
entire dataset as if it were test folds). These test folds can then be used
to get estimates of error, compare to other pipelines, etc.


**Matbench**

`Matminer <https://github.com/hackingmaterials/matminer>`_
provides access to the MatBench benchmark suite, a curated set of 13 diverse
materials ML problems which work in Automatminer benchmarks. Learn more here:
:doc:`MatBench </datasets>`


Time Savers and Practical Tools
-------------------------------
ignoring a column


Customizing pipelines
---------------------


Using DFTransformers individually
---------------------------------






