Basic Usage
==================

Basic usage of Automatminer requires using only one class - :code:`MatPipe`.


:code:`MatPipe` works with dataframes as input and output. It is able to train
on training data using it's :code:`fit` method, predict on new data using
:code:`predict`, and run benchmarks using :code:`benchmark` - all in an
automatic and end-to-end fashion.

Materials primitives (e.g., crystal structures) go in one end, and property
predictions come out the other. :code:`MatPipe` handles the intermediate
operations such as assigning descriptors, cleaning problematic data, data
conversions, imputation, and machine learning.


This is just a quick overview of the basic functionality. For a detailed and
comprehensive tutorial, see the jupyter notebooks in the  automatminer directory
of the
`matminer_examples <https://github.com/hackingmaterials/matminer_examples>`_
repository.


Initializing a pipeline
-----------------------

The easiest way to initialize a matpipe is using a preset.

.. code-block:: python

    from automatminer import MatPipe

    pipe = MatPipe.from_preset("express")

This preset is a set of options specifying exactly how each of
:code:`MatPipe`'s constituent classes are set up. Typically, the "express"
preset will give you results with a moderate degree of accuracy and relatively
quick training, so we'll use that here.

*Note: The default :code:`MatPipe()` is equivalent to
:code:`MatPipe.from_preset("express")`; other presets have different
configuration options!*



Training a pipeline
---------------------

MatPipe has similar fit/transform syntax to scikit-learn. Your dataframe
might be of the form:

:code:`train_df`

.. list-table::
   :align: left
   :header-rows: 1

   * - :code:`structure`
     - :code:`my_property`
   * - :code:`<structure object>`
     - 0.3819
   * - :code:`<structure object>`
     - -0.1123
   * - :code:`<structure object>`
     - -0.091
   * - ...
     - ...

Where the structure column contains :code:`pymatgen` :code:`Structure`s and
the property column is the property you are interested in (the target). Use
:code:`fit` to train, and specify the target column. For the dataframe we
used above, you'd do:


.. code-block:: python

    from automatminer import MatPipe

    pipe = MatPipe.from_preset("express")

    # Fitting pipe on train_df using "my_property" as target
    pipe.fit(train_df, "my_property")

The MatPipe is now fit.


Making predictions
-------------------

Once the pipeline is fit, we can make predictions on out-of-sample data, provided
that data has the same input types that our pipeline was trained on. For example:


:code:`prediction_df`

.. list-table::
   :align: left
   :header-rows: 1

   * - :code:`structure`
   * - :code:`<structure object>`
   * - :code:`<structure object>`
   * - :code:`<structure object>`
   * - ...

Use :code:`predict` to predict new data.

.. code-block:: python

    from automatminer import MatPipe

    pipe = MatPipe.from_preset("express")
    pipe.fit(train_df, "my_property")

    # Predicting my_property values of some unknown prediction_df structures
    prediction_df = pipe.predict(prediction_df)

The output will be stored in a column called :code:`"<your property> predicted"`.


:code:`prediction_df`

.. list-table::
   :align: left
   :header-rows: 1

   * - :code:`structure`
     - :code:`my_property predicted`
   * - :code:`<structure object>`
     - 0.449
   * - :code:`<structure object>`
     - -0.573
   * - :code:`<structure object>`
     - -0.005
   * - ...
     - ...


Using different presets
-----------------------

You can try out different configurations - such as more intensive featurization
routines, quicker training, etc. by initializing MatPipe with a different
config.

The "heavy" preset typically includes more CPU-intensive featurization and
longer training times.

.. code-block:: python

    from automatminer import MatPipe

    pipe = MatPipe.from_preset("heavy")


In contrast, use "debug" if you want very quick predictions.

.. code-block:: python

    from automatminer import MatPipe

    pipe = MatPipe.from_preset("debug")



Saving your pipeline for later
------------------------------

Once fit, you can save your pipeline as a pickle file:

.. code-block:: python

    pipe.save("my_pipeline.p")


To load your file, use the :code:`MatPipe.load` static method.

.. code-block:: python

    pipe = MatPipe.load("my_pipeline.p")


Examine your pipeline
---------------------

**Summarize**

For a human-readable executive summary of your pipeline, use
:code:`MatPipe.summarize()`.

.. code-block:: python

    summary = pipe.summarize()

The dict returned by summarize specifies the top-level information as strings.
An analogy: if your pipeline were a plumbing system, :code:`summarize` would
tell you how long each section of pipe is and the pump model.


**Inspect**

To get comprehensive details on a pipeline, use :code:`MatPipe.inspect()`.

.. code-block:: python

    details = pipe.inspect()

Inspection specifies all parameters to all Automatminer objects needed to
construct the pipeline and specify all of its internal operations. It is
typically very long, though human readable. An analogy: if your pipeline were
a plumbing system, :code:`inspect` would tell you everything :code:`summarize`
tells you, plus the model numbers of all the bolts, joints, and valves.


**Save to a file**

Both :code:`summarize` and :code:`inspect` accept a filename argument if you'd
like to save their outputs to JSON, YAML, or text.

.. code-block:: python

    summary = pipe.summarize("my_summary.yaml")
    details = pipe.inspect("my_details.json")


Monitoring the log
------------------

The Automatminer log is a powerful tool for determining what is happening within
the pipeline. We recommend you monitor it closely as the pipeline runs.


Quick reminders
---------------

**A quick note**:
Default MatPipe configs automatically infer the type of pymatgen object from
the dataframe column name: e.g.,

"composition" = :code:`pymatgen.Composition`,

"structure" = :code:`pymatgen.Structure`,

"bandstructure" = :code:`pymatgen.electronic_structure.bandstructure.BandStructure`,

"dos" = :code:`pymatgen.electronic_structure.dos.DOS`.

**Make sure your dataframe has the correct name for its input!** If you want to
use custom names, see the advanced usage page.



