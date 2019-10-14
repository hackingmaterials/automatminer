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

**Usage**

:code:`MatPipe` has a :code:`benchmark` method which can be used for
automatically benchmarking a pipeline on a dataset.

.. code-block:: python

    from sklearn.model_evaluation import KFold

    from automatminer.pipeline import MatPipe

    # Fit a pipeline to training data to predict band gap
    pipe = MatPipe()
    pipe.fit(train_df, "band gap")

    # Predict bandgap of some unknown materials
    predicted_df = pipe.predict(unknown_df)




` Matminer <https://github.com/hackingmaterials/matminer>`_
provides access to the MatBench benchmark suite, a curated set of 13 diverse
materials ML problems which work in Automatminer benchmarks. Learn more here:


Customizing pipelines
---------------------


Using DFTransformers individually
---------------------------------





