Basic Usage
==================

Basic usage of Automatminer requires using only one class - :code:`MatPipe`.


:code:`MatPipe` works with dataframes as input and output. It is able to train
on training data using it's :code:`fit` method, predict on new data using
:code:`predict`, and run benchmarks using :code:`benchmark` - all in an
automatic and end-to-end fashion.

Materials primitives (e.g., crystal structures go in one end, and property
predictions come out the other). :code:`MatPipe` handles the intermediate
operations such as assigning descriptors, cleaning problematic data, data
conversions, imputation, and machine learning.


This is just a quick overview of the basic functionality. For a detailed and
comprehensive tutorial, see the jupyter notebooks in the  automatminer directory
 of the
`matminer_examples <https://github.com/hackingmaterials/matminer_examples>`_
repository.


Initializing a MatPipe
-----------------------

The easiest way to initialize a matpipe is using a preset.

.. code-block:: python

    from automatminer import get_preset_config, MatPipe

    express_config = get_preset_config("express")
    pipe = MatPipe(**express_config)

This preset is a dictionary of options specifying exactly how each of
:code:`MatPipe`'s constituent classes are set up. Typically, the "express"
preset will give you results with a moderate degree of accuracy and relatively
quick training, so we'll use that here.


Training a pipeline
---------------------

MatPipe has similar fit/transform syntax to scikit-learn. Your dataframe might
be of the form:

| a | b | c | d |   |
|:-:|---|---|:-:|---|
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |



Making predictions
-------------------



Running a benchmark
--------------------



Using different presets
-----------------------





