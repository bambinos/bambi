BAyesian Model-Building Interface (Bambi) in Python
===================================================
|Build|
|Coverage|


.. |Build| image:: https://travis-ci.org/bambinos/bambi.svg
    :target: https://travis-ci.org/bambinos/bambi

.. |Coverage| image:: https://coveralls.io/repos/github/bambinos/bambi/badge.svg
    :target: https://coveralls.io/github/bambinos/bambi

Bambi is a high-level Bayesian model-building interface written in Python. It works with the probabilistic programming frameworks `PyMC3 <https://docs.pymc.io/>`__ and is designed to make it extremely easy to fit Bayesian mixed-effects models common in biology, social sciences and other disciplines.


Dependencies
============
Bambi is tested on Python 3.6+ and depends on NumPy, Pandas, PyMC3, PyStan, Patsy and ArviZ (see `requirements.txt <https://github.com/bambinos/bambi/blob/master/requirements.txt>`_ for version information).

Installation
============
The latest release of Bambi can be installed using pip:

.. code-block:: bash

   pip install bambi

Alternatively, if you want the bleeding edge version of the package, you can install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/bambinos/bambi.git

Usage
=====
A simple fixed effects model is shown below as example.

.. code-block:: python

    from bambi import Model
    import pandas as pd

    # Read in a tab-delimited file containing our data
    data = pd.read_table('my_data.txt', sep='\t')

    # Initialize the model
    model = Model(data)

    # Fixed effects only model
    results = model.fit('DV ~ IV1 + IV2', draws=1000, chains=4)

    # Use ArviZ to plot the results
    az.plot_trace(results)

    # Key summary and diagnostic info on the model parameters
    az.summary(results)

    # Drop the first 100 draws (burn-in)
    results_bi = results.sel(draw=slice(100, None))

For a more in-depth introduction to Bambi see our `Quickstart <https://github.com/bambinos/bambi#quickstart>`_ or our set of example notebooks.

Contributing
============
We welcome contributions from interested individuals or groups! For information about contributing to Bambi, check out our instructions, policies, and guidelines `here <https://github.com/bambinos/bambi/blob/master/CONTRIBUTING.md>`_.

Contributors
============
See the `GitHub contributor page <https://github.com/bambinos/bambi/graphs/contributors>`_.

Contents
========

.. toctree::
   :maxdepth: 4

   getting_started
   examples
   api_reference

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
