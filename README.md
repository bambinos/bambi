<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200></img>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/ci.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

BAyesian Model-Building Interface in Python

## Overview

Bambi is a high-level Bayesian model-building interface written in Python. It's built on top of the [PyMC](https://github.com/pymc-devs/pymc) probabilistic programming framework, and is designed to make it extremely easy to fit mixed-effects models common in biology, social sciences, and other disciplines using a Bayesian approach.

## Installation

Bambi requires Python 3.12 or later. Install the latest release with pip:

```bash
pip install bambi
```

With uv, add Bambi to your project:

```bash
uv add bambi
```

Or install it with conda:

```bash
conda install -c conda-forge bambi
```

To install the development version from GitHub:

```bash
pip install git+https://github.com/bambinos/bambi.git
```

### Dependencies

Bambi requires working versions of ArviZ, formulae, NumPy, pandas and PyMC. Dependencies are listed in `pyproject.toml` and should all be installed by the Bambi installer; no further action should be required.

## Documentation

Visit the [Bambi documentation](https://bambinos.github.io/bambi/) for guides,
examples, and the API reference. Start with the
[Quickstart](https://bambinos.github.io/bambi/notebooks/getting_started.html),
explore the [examples gallery](https://bambinos.github.io/bambi/notebooks/), or
consult the [API reference](https://bambinos.github.io/bambi/reference/).

## Citation

If you use Bambi and want to cite it please use

```bibtex
@article{Capretto2022,
 title={Bambi: A Simple Interface for Fitting {Bayesian} Linear Models in {Python}},
 volume={103},
 url={https://www.jstatsoft.org/index.php/jss/article/view/v103i15},
 doi={10.18637/jss.v103.i15},
 number={15},
 journal={Journal of Statistical Software},
 author={Capretto, Tom\'{a}s and Piho, Camen and Kumar, Ravin and Westfall, Jacob and Yarkoni, Tal and Martin, Osvaldo A},
 year={2022},
 pages={1–29}
}
```

## Contributions

Bambi is a community project and welcomes contributions. Additional information can be found in the [Contributing](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md) Readme.

For a list of contributors see the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page.

## Donations

If you want to support Bambi financially, you can [make a donation](https://numfocus.org/donate-to-pymc) to our sister project PyMC.

## Code of Conduct

Bambi wishes to maintain a positive community. Additional details can be found in the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/CODE_OF_CONDUCT.md)

If you have experienced or witnessed any behaviour that violates our Code of Conduct, we encourage you to report it through [this reporting form](https://bambinos.github.io/bambi/coc-violation.html).

## License

[MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)
