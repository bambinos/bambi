<img src="https://raw.githubusercontent.com/bambinos/bambi/main/docs/logos/RGB/Bambi_logo.png" width=200></img>

[![PyPi version](https://badge.fury.io/py/bambi.svg)](https://badge.fury.io/py/bambi)
[![Build Status](https://github.com/bambinos/bambi/actions/workflows/test.yml/badge.svg)](https://github.com/bambinos/bambi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/bambinos/bambi/branch/master/graph/badge.svg?token=ZqH0KCLKAE)](https://codecov.io/gh/bambinos/bambi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

BAyesian Model-Building Interface in Python

## Overview

Bambi is a high-level Bayesian model-building interface written in Python. It's built on top of the [PyMC](https://github.com/pymc-devs/pymc) probabilistic programming framework, and is designed to make it extremely easy to fit mixed-effects models common in social sciences settings using a Bayesian approach.

## Installation

Bambi requires a working Python interpreter (3.10+). We recommend installing Python and key numerical libraries using the [Anaconda Distribution](https://www.anaconda.com/products/individual#Downloads), which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine (including pip), Bambi itself can be installed in one line using pip:

    pip install bambi

Alternatively, if you want the bleeding edge version of the package you can install from GitHub:

    pip install git+https://github.com/bambinos/bambi.git

### Dependencies

Bambi requires working versions of ArviZ, formulae, NumPy, pandas and PyMC. Dependencies are listed in `pyproject.toml` and should all be installed by the Bambi installer; no further action should be required.

## Examples

In the following two examples we assume the following basic setup

```python
import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
```

### Linear regression

A simple fixed effects model is shown in the example below.

```python
# Read in a dataset from the package content
data = bmb.load_data("sleepstudy")

# See first rows
data.head()
 
# Initialize the fixed effects only model
model = bmb.Model('Reaction ~ Days', data)

# Get model description
print(model)

# Fit the model using 1000 on each chain
results = model.fit(draws=1000)

# Key summary and diagnostic info on the model parameters
az.summary(results)

# Use ArviZ to plot the results
az.plot_trace(results)
```
``` 
   Reaction  Days  Subject
0  249.5600     0      308
1  258.7047     1      308
2  250.8006     2      308
3  321.4398     3      308
4  356.8519     4      308
```
```
       Formula: Reaction ~ Days
        Family: gaussian
          Link: mu = identity
  Observations: 180
        Priors:
    target = mu
        Common-level effects
            Intercept ~ Normal(mu: 298.5079, sigma: 261.0092)
            Days ~ Normal(mu: 0.0, sigma: 48.8915)

        Auxiliary parameters
            sigma ~ HalfStudentT(nu: 4.0, sigma: 56.1721)
```
```
                   mean     sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
Intercept       251.552  6.658  238.513  263.417      0.083    0.059    6491.0    2933.0    1.0
Days             10.437  1.243    8.179   12.793      0.015    0.011    6674.0    3242.0    1.0
Reaction_sigma   47.949  2.550   43.363   52.704      0.035    0.025    5614.0    2974.0    1.0
```

First, we create and build a Bambi `Model`. Then, the method `model.fit()` tells the sampler to start
running and it returns an `InferenceData` object, which can be passed to several ArviZ functions
such as `az.summary()` to get a summary of the parameters distribution and sample diagnostics or
`az.plot_trace()` to visualize them.

### Logistic regression

In this example we will use a simulated dataset created as shown below.

```python
data = pd.DataFrame({
    "g": np.random.choice(["Yes", "No"], size=50),
    "x1": np.random.normal(size=50),
    "x2": np.random.normal(size=50)
})
```

Here we just add the `family` argument set to `"bernoulli"` to tell Bambi we are modelling a binary
response. By default, it uses a logit link. We can also use some syntax sugar to specify which event
we want to model. We just say `g['Yes']` and Bambi will understand we want to model the probability
of a `"Yes"` response. But this notation is not mandatory. If we use `"g ~ x1 + x2"`, Bambi will
pick one of the events to model and will inform us which one it picked.

```python
model = bmb.Model("g['Yes'] ~ x1 + x2", data, family="bernoulli")
fitted = model.fit()
```

After this, we can evaluate the model as before. 

### More

For a more in-depth introduction to Bambi see our [Quickstart](https://github.com/bambinos/bambi#quickstart) and check the notebooks in the [Examples](https://bambinos.github.io/bambi/notebooks/) webpage.

## Documentation

The Bambi documentation can be found in the [official docs](https://bambinos.github.io/bambi/index.html)

## Citation

If you use Bambi and want to cite it please use

```bibtex
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
```

## Contributions

Bambi is a community project and welcomes contributions. Additional information can be found in the [Contributing](https://github.com/bambinos/bambi/blob/main/docs/CONTRIBUTING.md) Readme.

For a list of contributors see the [GitHub contributor](https://github.com/bambinos/bambi/graphs/contributors) page

## Donations

If you want to support Bambi financially, you can [make a donation](https://numfocus.org/donate-to-pymc) to our sister project PyMC.

## Code of Conduct

Bambi wishes to maintain a positive community. Additional details can be found in the [Code of Conduct](https://github.com/bambinos/bambi/blob/main/docs/CODE_OF_CONDUCT.md)

## License

[MIT License](https://github.com/bambinos/bambi/blob/main/LICENSE)
