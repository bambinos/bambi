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

.. |Coverage| image:: https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE
    :target: https://codecov.io/gh/bambinos/bambi

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black



Bambi is a high-level Bayesian model-building interface written in Python. It works with the probabilistic programming frameworks `PyMC3 <https://docs.pymc.io/>`__ and is designed to make it extremely easy to fit Bayesian mixed-effects models common in biology, social sciences and other disciplines.


Dependencies
============
Bambi is tested on Python 3.7+ and depends on ArviZ, formulae, NumPy, pandas, PyMC3 and statsmodels (see `requirements.txt <https://github.com/bambinos/bambi/blob/master/requirements.txt>`_ for version information).

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

    # Initialize the fixed effects only model
    model = Model('DV ~ IV1 + IV2', data)

    # Fit the model using 1000 on each of 4 chains
    results = model.fit(draws=1000, chains=4)

    # Use ArviZ to plot the results
    az.plot_trace(results)

    # Key summary and diagnostic info on the model parameters
    az.summary(results)

    # Drop the first 100 draws (burn-in)
    results_bi = results.sel(draw=slice(100, None))

For a more in-depth introduction to Bambi see our `Quickstart <https://github.com/bambinos/bambi#quickstart>`_ or our set of example notebooks.

Citation
========
If you use Bambi and want to cite it please use |arXiv|

.. |arXiv| image:: https://img.shields.io/badge/arXiv-2012.10754-b31b1b.svg
    :target: https://arxiv.org/abs/2012.10754

Here is the citation in BibTeX format

.. code-block::

    @misc{capretto2020,
        title={Bambi: A simple interface for fitting Bayesian linear models in Python},
        author={Tomás Capretto and Camen Piho and Ravin Kumar and Jacob Westfall and Tal Yarkoni and Osvaldo A. Martin},
        year={2020},
        eprint={2012.10754},
        archivePrefix={arXiv},
        primaryClass={stat.CO}
    }


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

   notebooks/getting_started
   examples
   api_reference

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
