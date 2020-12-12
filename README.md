# Bambi

BAyesian Model-Building Interface in Python

[![Build Status](https://travis-ci.org/bambinos/bambi.svg?branch=master)](https://travis-ci.org/bambinos/bambi)
[![Coverage Status](https://coveralls.io/repos/github/bambinos/bambi/badge.svg?branch=master)](https://coveralls.io/github/bambinos/bambi?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

## Overview

Bambi is a high-level Bayesian model-building interface written in Python. It's built on top of the [PyMC3](https://github.com/pymc-devs/pymc3) probabilistic programming framework, and is designed to make it extremely easy to fit mixed-effects models common in social sciences settings using a Bayesian approach.

## Installation

Bambi requires a working Python interpreter (3.7+). We recommend installing Python and key numerical libraries using the [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads), which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine (including pip), Bambi itself can be installed in one line using pip:

    pip install bambi

Alternatively, if you want the bleeding edge version of the package you can install from GitHub:

    pip install git+https://github.com/bambinos/bambi.git

### Dependencies

Bambi requires working versions of numpy, pandas, matplotlib, patsy, pymc3, and theano. Dependencies are listed in `requirements.txt`, and should all be installed by the Bambi installer; no further action should be required.

## Documentation

The Bambi documentation can be found in the [official docs](https://bambinos.github.io/bambi/index.html)

## Contributions

Bambi is a community project and welcomes contributions. Additional information can be found in the [Contributing](https://github.com/bambinos/bambi/blob/master/CONTRIBUTING.md) Readme.

For a list of contributors see the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page

## Code of Conduct

Bambi wishes to maintain a positive community. Additional details can be found in the [Code of Conduct](https://github.com/bambinos/bambi/blob/master/CODE_OF_CONDUCT.md)

## License

[MIT License](https://github.com/bambinos/bambi/blob/master/LICENSE)
