.. raw:: html

    <style> .red {color:#aa0060; font-weight:bold; font-size:16px} </style>

.. role:: red

:red:`WARNING! These docs and examples are a work in progress. Read and use at your own risk!``

.. image:: _static/logo.svg
   :alt: server
   :align: center
   :width: 600px


Automatminer is a tool for *automatically* creating **complete** machine learning pipelines for materials science, including automatic featurization with `matminer <https://github.com/hackingmaterials/matminer>`_, feature reduction, and an AutoML backend. Put in a materials dataset, get out a machine that predicts materials properties.

How it works
=============

Automatminer automatically decorates a dataset using hundreds of descriptor techniques from matminer's descriptor library, picks the most useful features for learning, and runs a separate AutoML pipeline using TPOT. Once a pipeline has been fit, it can be examined with skater's interpretability tools, summarized in a text file, saved to disk, or used to make new predictions.

.. image:: _static/automatminer_big.jpg
   :alt: server
   :align: center


Here's an example of training on known data, and extending the model to out of sample data.

.. code-block:: python

    from automatminer.pipeline import MatPipe

    # Fit a pipeline to training data to predict band gap
    pipe = MatPipe()
    pipe.fit(train_df, "band gap")

    # Predict bandgap of some unknown materials
    predicted_df = pipe.predict(unknown_df, "band gap")


Or, run a (relatively) rigorous nested cross validation benchmark on a known dataset, and then compare the results against your own ML models:

.. code-block:: python

    from automatminer.pipeline import MatPipe
    from sklearn.model_selection import KFold

    pipe = MatPipe()
    predictions_per_fold = pipe.benchmark(df, "bulk modulus", KFold(n_splits=5))


Scope
=====

Automatminer can work with many kinds of data:
----------------------------------------------
-   both computational and experimental data
-   small (~100 samples) to moderate (~100k samples) sized datasets
-   crystalline datasets
-   composition-only (i.e., unknown phases) datasets
-   datasets containing electronic bandstructures or density of states

Many kinds of target properties:
--------------------------------
-   electronic
-   mechanical
-   thermodynamic
-   any other kind of property

And many featurization (descriptor) techniques:
-----------------------------------------------
See `matminer's Table of Featurizers <https://hackingmaterials.github.io/matminer/featurizer_summary.html>`_ for a full (and growing) list.


Installation
============

Install from Pypi:

.. code-block:: bash

    pip install automatminer


Clone latest code from github

.. code-block:: bash

    git clone https://github.com/hackingmaterials/automatminer
    cd automatminer
    pip install -e .

Full Code Examples
==================

We are now going to walk through how to create a MatPipe using the default
configurations and the elastic_tensor_2015 dataset. We will then use this
MatPipe to benchmark the target property K_VRH and we will use our results
to determine the mean squared error. Buckle up!

Setting up the Dataframe
------------------------

We will use the matminer function load_dataset to give us access to the
elastic_tensor_2015 dataset. The result is a Pandas dataframe.

.. code-block:: python

    from matminer.datasets.dataset_retrieval import load_dataset

    df = load_dataset("elastic_tensor_2015") #Loads in Pandas dataset


Next, we will use get_preset_config to use different pre-built configurations
for a MatPipe. The options include production, default, robust, and debug.
Specific details about each config can be seen in `presets.py
<api/automatminer.get_preset_config.html>`_. In this example, we will be using
the debug config for a short program runtime. Of course, you do not need to use
a preset configuration. Simply use the `MatPipe <api/automatminer.MatPipe.html>`_
functions to choose your own adaptor. After this step, we will pass in the parameter
as an argument of `MatPipe <api/automatminer.MatPipe.html>`_ to get a MatPipe
object.

.. code-block:: python

    from automatminer.presets import get_preset_config
    from automatminer.pipeline import MatPipe

    # Get preset configurations for debug. The debug configuration allows
    # for rapid testing while the other configurations are more useful for
    # real-world applications.
    debug_config = get_preset_config("debug")
    # Create a MatPipe using our configuration.
    pipe = MatPipe(**debug_config)


The preset automatminer uses pre-defined column names 'composition' and 'structure'
to find the composition and structure columns. You can easily fix this by renaming
your respective columns to the correct names.

.. code-block:: python

    # Rename the appropriate dataframe columns to create a dataframe that
    # can be passed into our automatminer functions.
    df = df.rename(columns={"formula": "composition"})[["composition", "structure", "K_VRH"]]


Benchmarking automatminer's performance
---------------------------------------

In this example, we are performing a machine learning benchmark using MatPipe
in order to see how well our MatPipe can predict a certain target property.
The target property we will be benchmarking in this example is K_VRH. Keep in
mind that benchmarking requires a KFold object since benchmarks are run with
nested cross validation. But why nested cross validation?


Nested CV for benchmarking
----------------------------
Reporting a regular cross validation score is fine, if you are not tuning the
hyperparameters of your model.


.. image:: _static/cv_single.png
   :alt: cv_overfit
   :align: center
   :width: 70%

But if the model's hyperparameters are adjusted
to improve its CV score, reporting the CV score as the generalization error introduces model selection bias into the error estimate. In other words, you are implicitly introducing test set knowledge into the model's training.

.. image:: _static/cv_overfit.png
   :alt: cv_overfit
   :align: center
   :width: 70%


Using a hold out test set is a better - in this procedure, all training and
validation is done without knowledge of the final test set, then the
generalization error is estimated from the prediction error on the test set.
However, the choice of test set may result in over- or under-representing
your models generalization error.

.. image:: _static/cv_holdout.png
   :alt: cv_overfit
   :align: center

Furthermore, if your model's hyperparameters
are adjusted based on the test score (and not only the validation score), the
model will also be overfit to the test set.

Nested CV mitigates these issues by repeating the following hold-out test procedure k times:

    1. Split data into train/validation and hold out test.
    2. Give only the train/validation data to the model and allow it to optimize hyperparameters using any method it chooses
    3. Once the model's hyperparameters are set, predict the hold out test set and report that as the generalization error.

1-3 are repeated for each of k folds in the Nested CV, ensuring every sample in the dataset is tested once.
This mitigates the performance of the benchmark based on the choice of test set and also better estimates the generalization error
than a single validation/test split would.


.. image:: _static/cv_nested.png
   :alt: cv_overfit
   :align: center


tl;dr: **“A nested CV procedure provides an almost unbiased estimate of the true error.”** – Varma and Simon, 2006 (`10.1186/1471-2105-7-91 <https://www.ncbi.nlm.nih.gov/pubmed/16504092>`_)

Setting up the benchmark
----------------------------

.. code-block:: python

    """
    MatPipe benchmarks with a nested cross validation, meaning it makes
    k validation/test splits, where all model selection is done on the train
    /validation set (a typical CV). When the model is done validating, it is
    used to predict the previously unseen test set data.
    """
    kfold = KFold(n_splits=5) #We will use a 5-Fold object.
    predicted_folds = pipe.benchmark(df, "K_VRH", kfold)


The result of pipe.benchmark() will be a list of dataframes (one for each
nested CV fold). Each new dataframe has the predicted results
stored in a column called the property name combined with " predicted". In this
example, it will be stored in "K_VRH predicted."


Calculating MSE
---------------

For each test fold in our nested CV, we can calculate the prediction error.

Next, we can use the sklearn package to calculate a wide variety of metrics on
our predictions. In this case, we want the mean squared error so we will use that
function.

Finally, we calculate the mean mse across our test sets.

.. code-block:: python

    from sklearn.metrics.regression import mean_squared_error

    # A list to hold our mse scores for each test fold
    mses = []

    # Calculating mse for each test fold
    for predicted in predicted_folds:
        # Save the actual K_VRH Series to y_true.
        y_true = predicted["K_VRH"]
        # Save the predicted K_VRH Series to y_test.
        y_test = predicted["K_VRH predicted"]
        mse = mean_squared_error(y_true, y_test)
        mses.append(mse)

    print(mses)



And voilà, we are done! We have successfully loaded in a dataset, benchmarked a test property
using a MatPipe with 'debug' configs, and then ran an analysis on our results by calculating
MSE for each fold of a nested CV.


Citing automatminer
===================
We are in the process of writing a paper for automatminer. In the meantime, please use the citation given in the `matminer repo <https://github.com/hackingmaterials/matminer>`_.

Contributing
============
Interested in contributing? See our `contribution guidelines <https://github.com/hackingmaterials/automatminer/blob/master/CONTRIBUTING.md>`_ and make a pull request! Please submit questions, issues/bug reports, and all other communication through the  `matminer Google Group <https://groups.google.com/forum/#!forum/matminer>`_.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`Python API<directory>`
