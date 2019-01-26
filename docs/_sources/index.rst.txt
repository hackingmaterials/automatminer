.. raw:: html

    <style> .red {color:#aa0060; font-weight:bold; font-size:16px} </style>

.. role:: red

:red:`WARNING! These docs are incomplete. Read and use at your own risk!``

.. image:: _static/logo_med.png
   :alt: server
   :align: center
   :width: 400px


Automatminer is a tool for automatically creating complete machine learning pipelines for materials science, which includes automatic featurization with `matminer <https://github.com/hackingmaterials/matminer>`_, feature reduction, and an AutoML backend. Put in a materials dataset, get out a machine that predicts materials properties.


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


Alternatively, run a nested cross validation benchmark on a known dataset, and then compare the results against your own ML models:

.. code-block:: python

    from automatminer.pipeline import MatPipe
    from sklearn.model_selection import KFold

    pipe = MatPipe()
    predictions_per_fold = pipe.benchmark(df, "bulk modulus", KFold(n_splits=5))



automatminer is applicable to many problems
-------------------------------------------

Automatminer can work with many kinds of data:
*   both computational and experimental data
*   small (~100 samples) to moderate (~100k samples) sized datasets
*   crystalline datasets
*   composition-only (i.e., unknown phases) datasets
*   datasets containing electronic bandstructures or density of states

...Many kinds of target properties
*   electronic
mechanical
thermodynamic
any other kind of property

...And many featurization (descriptor) techniques:
*list them*

Automatminer automatically decorates a dataset using hundreds of descriptor techniques from matminer's descriptor library, picks the most useful features for learning, and runs a separate AutoML pipeline using TPOT. Once a pipeline has been fit, it can be examined with skater's interpretability tools, summarized in a text file, saved to disk, or used to make new predictions.


Code Examples
=============

The easiest (and most automatic) way to use automatminer is through the MatPipe object. First, fit the MatPipe to a dataframe containing materials objects such as chemical compositions (or pymatgen Structures) and some material target property.
```python

```

Now use your pipeline to predict the properties of some other data, such as a new composition or structure.
```python

```

You can also use it to benchmark against other machine learning models with the `benchmark` method of MatPipe, which runs a Nested Cross Validation. The Nested CV scheme
is typically a more robust way of estimating an ML pipeline's generalizaiton error than a simple train/validation/test split.
```python
from automatminer.pipeline import MatPipe
from sklearn.model_selection import KFold

pipe = MatPipe()
predictions_per_fold = pipe.benchmark(df, "bulk modulus", KFold(n_splits=5))
```

Once a MatPipe has been fit, you can examine it internally to see how it works using `pipe.digest()`; or pickle it for later with `pipe.save()`.

### Citing automatminer
We are in the process of writing a paper for automatminer. In the meantime, please use the citation given in the matminer repo.

## Contributing
Interested in contributing? See our [contribution guidelines](https://github.com/hackingmaterials/automatminer/blob/master/CONTRIBUTING.md) and make a pull request! Please submit questions, issues / bug reports, and all other communication through the [matminer Google Group](https://groups.google.com/forum/#!forum/matminer).



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
