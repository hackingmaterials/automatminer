Installation
============

Automatminer supports Python 3.6.7 and Python 3.7.1+ on MacOS and Linux. Windows
users may be able to install Automatminer (and we will try to help you as much
as possible on the `forum <https://discuss.matsci.org>`_), but it is not officially
supported.


From PyPi (using pip)
---------------------

You can install the latest released version of automatminer through pip

.. code-block:: bash

    pip install automatminer


From source
-----------

To install Automatminer from source, first clone the repository from GitHub,
then use pip to install:

.. code-block:: bash

    git clone https://github.com/hackingmaterials/automatminer.git
    cd automatminer
    pip install .

If not installing from inside a virtual environment or conda environment, you
may need to specify to install as a *user* via:

.. code-block:: bash

    pip install . --user

