Installation
============

From source
-----------

To install AMSET from source, first clone the repository from GitHub, then
install using pip:

.. code-block:: bash

    git clone https://github.com/hackingmaterials/amset.git
    cd amset
    pip install .

If not installing from inside a virtual environment or conda environment, you
may need to specify to install as a *user* via:

.. code-block:: bash

    pip install . --user

Installing AMSET on NERSC
~~~~~~~~~~~~~~~~~~~~~~~~~

The BolzTraP2 dependency requires some configuration to be installed properly on
CRAY systems. Accordingly, AMSET can be installed using:

.. code-block:: bash

    CXX=icc CRAYPE_LINK_TYPE=shared pip install amset
