Using AMSET
===========

AMSET can be used from the command-line as a standalone program or from the
Python API. In both cases, the primary input is a ``vasprun.xml`` file from a
uniform band structure calculation (i.e., on a regular k-point grid and not
along high-symmetry lines).

Temperature and doping ranges, scattering rates, and calculation performance
parameters are controlled through the settings file. More details on the
available settings are provided in the `settings section <settings>`_ of the
documentation. An example settings file is given :ref:`here <example-settings>`.

From the command-line
---------------------

AMSET can be run from the command-line using the ``amset`` command. The help
menu listing a summary of the command-line options can be printed using:

.. code-block:: bash

    amset -h

By default, AMSET will look for a ``vasprun.xml`` file and ``settings.yaml``
file in the current directory. A different directory can be specified using
the ``directory`` option, e.g.:
v
.. code-block:: bash

    amset --directory path/to/files

Any settings specified via the command line will override those in the settings
file. For example, the interpolation factor can be easily controlled using:

.. code-block:: bash

    amset --interpolation-factor 20

From the Python API
-------------------

Greater configurability is available when running AMSET from the Python API.
For example, the following snippet will look for a ``vasprun.xml`` and
``settings.yaml`` file in the current directory, then run AMSET.

.. code-block:: python

    from amset.run import AmsetRunner

    runner = AmsetRunner.from_directory(directory='.')
    runner.run()

The API allows for easily converging performance parameters. For example,
the following snippet will run AMSET using multiple interpolation parameters.

.. code-block:: python

    from amset.run import AmsetRunner

    settings = {'general': {'interpolation_factor': 5}}

    for i_factor in range(10, 100, 10):
        settings["general"]["interpolation_factor"] = i_factor

        runner = AmsetRunner.from_directory(
            directory='.', settings_override=settings)
        runner.run()

When running AMSET from the API, it is not necessary to use a settings file
at all. Instead the settings can be passed as a dictionary. For example:

.. code-block:: python

    from amset.run import AmsetRunner

    settings = {
        "general": {
            "interpolation_factor": 150,
            "doping": [1.99e+14, 2.20e+15, 1.72e+16,
                       1.86e+17, 1.46e+18, 4.39e+18],
            "temperatures": [300]
        },

        "material": {
            "deformation_potential": (6.5, 6.5),
            "elastic_constant": 190,
            "static_dielectric": 13.1,
        },
    }

    runner = AmsetRunner.from_vasprun_and_settings("vasprun.xml.gz", settings)
    runner.run()

Output files
------------

Convergence
-----------



