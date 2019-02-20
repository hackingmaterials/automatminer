.. raw:: html

    <style> .red {color:#aa0060; font-weight:bold; font-size:16px} </style>

.. role:: red

:red:`WARNING! These docs are incomplete. Read and use at your own risk!``

.. image:: _static/logo_med.png
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
for a MatPipe. The options include production, default, fast, and debug.
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
nested cross validation. This mitigates the performance of the benchmark based
on the choice of test set and also better estimates the generalization error
than a single validation/test split would.

.. code-block:: python

    """
    MatPipe benchmarks with a nested cross validation, meaning it makes
    k validation/test splits, where all model selection is done on the train
    /validation set (a typical CV). When the model is done validating, it is
    used to predict the previously unseen test set data.
    """
    kfold = KFold(n_splits=5) #We will use a 5-Fold object.
    predicted = pipe.benchmark(df, "K_VRH", kfold)


The result of pipe.benchmark() will be a new dataframe with the predicted results
stored in a column called the property name combined with " predicted". In this
example, it will be stored in "K_VRH predicted."


Calculating MSE
---------------

As mentioned above, the "predicted" variable is a dataframe that contains several columns,
including actual property values and predicted property values. In this example, we will
use the actual K_VRH data and the predicted K_VRH data in order to see how well the
benchmarking went.

.. code-block:: python

    # Save the actual K_VRH Series to y_true.
    y_true = predicted["K_VRH"]
    # Save the predicted K_VRH Series to y_test.
    y_test = predicted["K_VRH predicted"]


Finally, we can use the sklearn package to calculate a wide variety of metrics on
these two Series. In this case, we want the mean squared error so we will use that
function.

.. code-block:: python
    from sklearn.metrics.regression import mean_squared_error

    # Calculate the mean squared error between our two Series and save it to mse.
    mse = mean_squared_error(y_true, y_test)


And voilà, we are done! We have successfully loaded in a dataset, benchmarked a test property
using a MatPipe with 'debug' configs, and then ran an analysis on our results by calculating
MSE. And all that in less than 15 lines of code!


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
