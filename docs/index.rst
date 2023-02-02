BAyesian Model-Building Interface (Bambi) in Python
===================================================
|PyPI version|
|Tests|
|Coverage|
|Black|


.. |PyPI version| image:: https://badge.fury.io/py/bambi.svg
    :target: https://badge.fury.io/py/bambi

.. |Tests| image:: https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg
    :target: https://github.com/bambinos/bambi

.. |Coverage| image:: https://codecov.io/gh/bambinos/bambi/branch/main/graph/badge.svg?token=ZqH0KCLKAE
    :target: https://codecov.io/gh/bambinos/bambi

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black



Bambi is a high-level Bayesian model-building interface written in Python. It works with the probabilistic programming frameworks `PyMC <https://www.pymc.io/>`__ and is designed to make it extremely easy to fit Bayesian mixed-effects models common in biology, social sciences and other disciplines.


Dependencies
============
Bambi is tested on Python 3.8+ and depends on ArviZ, formulae, NumPy, pandas and PyMC (see `pyproject.toml <https://github.com/bambinos/bambi/blob/main/pyproject.toml>`_ for version information).

Installation
============

Bambi is available from the Python Package Index at `<https://pypi.org/project/bambi/>`_, alternatively it can be installed using Conda.

PyPI
----

The latest release of Bambi can be installed using pip:

.. code-block:: bash

   pip install bambi

Alternatively, if you want the bleeding edge version of the package, you can install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/bambinos/bambi.git


Conda
-----

If you use Conda, you can also install the latest release of Bambi with the following command:

.. code-block:: bash

   conda install -c conda-forge bambi


Usage
=====
A simple fixed effects model is shown in the example below.

.. code-block:: python

    import arviz as az
    import bambi as bmb
    import pandas as pd

    # Read in a tab-delimited file containing our data
    data = pd.read_table('my_data.txt', sep='\t')

    # Initialize the fixed effects only model
    model = bmb.Model('DV ~ IV1 + IV2', data)

    # Fit the model using 1000 on each of 4 chains
    results = model.fit(draws=1000, chains=4)

    # Use ArviZ to plot the results
    az.plot_trace(results)

    # Key summary and diagnostic info on the model parameters
    az.summary(results)

For a more in-depth introduction to Bambi see our `Quickstart <https://github.com/bambinos/bambi#quickstart>`_ or our set of example notebooks.

Citation
========
If you use Bambi and want to cite it please use

.. code-block::

    @article{Capretto2022,
            title={Bambi: A Simple Interface for Fitting Bayesian Linear Models in Python},
            volume={103},
            url={https://www.jstatsoft.org/index.php/jss/article/view/v103i15},
            doi={10.18637/jss.v103.i15},
            number={15},
            journal={Journal of Statistical Software},
            author={Capretto, Tomás and Piho, Camen and Kumar, Ravin and Westfall, Jacob and Yarkoni, Tal and Martin, Osvaldo A},
            year={2022},
            pages={1–29}
            }


Contributing
============
We welcome contributions from interested individuals or groups! For information about contributing to Bambi, check out our instructions, policies, and guidelines `here <https://github.com/bambinos/bambi/blob/main/docs/CONTRIBUTING.md>`_.

Contributors
============
See the `GitHub contributor page <https://github.com/bambinos/bambi/graphs/contributors>`_.

Contents
========

.. toctree::
   :maxdepth: 4

   notebooks/getting_started
   examples
   api_reference
   faq

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
