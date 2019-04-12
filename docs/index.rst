.. bambi documentation master file, created by
   sphinx-quickstart on Sun Apr  6 15:22:20 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BAyesian Model-Building Interface (BAMBI) in Python
===================================================

.. title:: Bambi Documentation

.. image:: https://travis-ci.org/bambinos/bambi.svg?branch=master
    :target: https://travis-ci.org/bambinos/bambi

.. image:: https://coveralls.io/repos/github/bambinos/bambi/badge.svg?branch=master
    :target: https://coveralls.io/github/bambinos/bambi?branch=master

Bambi is a high-level Bayesian model-building interface written in Python. It works with two probabilistic programming frameworks, `PyMC3 <https://docs.pymc.io/>`__ or `PyStan <https://pystan.readthedocs.io/en/latest/>`__, and is designed to make it extremely easy to fit Bayesian mixed-effects models common in biology, social sciences and other disciplines.

New Features
============

Bambi version ``0.1.1`` will be the final version supporting Python 2, but look forward to the forthcoming Bambi version ``0.1.2``!

Dependencies
============
Bambi is tested on Python 2.7 and 3.6 and depends on NumPy, Pandas, PyMC3, PyStan, and Patsty (see `requirements.txt <https://github.com/bambinos/bambi/blob/master/requirements.txt>`_ for version information).

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
   results = model.fit('DV ~ IV1 + IV2', samples=1000, chains=4)

   # Drop the first 100 burn-in samples from each chain and plot
   results[100:].plot()

   # Key summary and diagnostic info on the model parameters
   results[100:].summary()

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
