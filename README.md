bambi
------
BAyesian Model-Building Interface in Python

## Status
* [![Build Status](https://travis-ci.org/bambinos/bambi.svg?branch=master)](https://travis-ci.org/bambinos/bambi)
* [![Coverage Status](https://coveralls.io/repos/github/bambinos/bambi/badge.svg?branch=master)](https://coveralls.io/github/bambinos/bambi?branch=master)

## Overview

Bambi is a high-level Bayesian model-building interface written in Python. It's built on top of the [PyMC3](https://github.com/pymc-devs/pymc3) probabilistic programming framework, and is designed to make it extremely easy to fit mixed-effects models common in social sciences settings using a Bayesian approach.

## Installation

Bambi requires a working Python interpreter (either 2.7+ or 3+). We recommend installing Python and key numerical libraries using the [Anaconda Distribution](https://www.continuum.io/downloads), which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine (including pip), Bambi can be installed in one line using pip:

    pip install bambi

For advanced users, or those who intend to contribute to Bambi development, we recommend installing PyMC3 manually before installing Bambi with pip. Follow the instructions on the [PyMC3 repo](https://github.com/pymc-devs/pymc3) to install PyMC3 with a development version of Theano, then run the above to intall Bambi.

### Dependencies

Bambi requires working versions of numpy, pandas, matplotlib, patsy, pymc3, and theano. Dependencies are listed in `requirements.txt`, and should all be installed by the Bambi installer; no further action should be required.

## Table of Contents
- [Quickstart](#quickstart)
- [User guide](#user-guide)
    + [Creating a model](#creating-a-model)
    + [Model specification](#model-specification)
        * [Formula-based specification](#formula-based-specification)
        * [Term-based specification](#term-based-specification)
        * [Mixing specification approaches](#mixing-specification-approaches)
        * [Notes on fixed and random effects in Bambi](#notes-on-fixed-and-random-effects-in-bambi)
        * [Coding of categorical variables](#coding-of-categorical-variables)
        * [Model specification notes](#model-specification-notes)
            - [Term naming caveats](#term-naming-caveats)
    + [Fitting the model](#fitting-the-model)
        * [Building the model](#building-the-model)
    + [Specifying priors](#specifying-priors)
        * [Different ways of specifying priors](#different-ways-of-specifying-priors)
        * [Mapping priors onto terms](#mapping-priors-onto-terms)
    + [Generalized linear mixed models](#generalized-linear-mixed-models)
        * [Families](#families)
    * [Results](#results)
    * [Accessing back-end objects](#accessing-back-end-objects)


## Quickstart

Suppose we have data for a typical within-subjects psychology experiment with 2 experimental conditions. Stimuli are nested within condition, and subjects are crossed with condition. We want to fit a model predicting reaction time (RT) from the fixed effect of condition, random intercepts for subjects, random condition slopes for students, and random intercepts for stimuli. We can fit this model and summarize its results as follows in bambi:

```python
from bambi import Model

# Assume we already have our data loaded
model = Model(data)
results = model.fit('rt ~ condition', random=['condition|subject', '1|stimulus'], samples=5000)
results.plot(burn_in=1000)
results.summary(1000)
```

## User guide

### Creating a model
Creating a new model in Bambi is simple:

```
from bambi import Model
import pandas as pd

# Read in a tab-delimited file containing our data
data = pd.read_table('my_data.txt', sep='\t')

# Initialize the model
model = Model(data)
```

Typically, we'll initialize a bambi `Model` by passing it a pandas `DataFrame` as the only argument. We get back a model that we can immediately start adding terms to.

#### Data format
As with most mixed effect modeling packages, Bambi expects data in "long" format--meaning that each row should reflect a single observation at the most fine-grained level of analysis. For example, given a model where students are nested into classrooms and classrooms are nested into schools, we would want data with the following kind of structure:

| student | gender | gpa | class | school |
|---------|--------|-----|-------|--------|
| 1       | F      | 3.4 | 1     | 1      |
| 2       | F      | 3.7 | 1     | 1      |
| 3       | M      | 2.2 | 1     | 1      |
| 4       | F      | 3.9 | 2     | 1      |
| 5       | M      | 3.6 | 2     | 1      |
| 6       | M      | 3.5 | 2     | 1      |
| 7       | F      | 2.8 | 3     | 2      |
| 8       | M      | 3.9 | 3     | 2      |
| 9       | F      | 4.0 | 3     | 2      |

### Model specification
Bambi provides two (mutually compatible) ways to specify a mixed-effects model: a high-level, formula-based API, and a lower-level, term-based API.

#### Formula-based specification
The more compact, high-level approach to model specification is to use a formula-based syntax similar to what one might find in R packages like lme4 or nlme. Some examples that illustrate the breadth of models that can be easily specified in Bambi:

```python
# Fixed effects only
model.fit('rt ~ attention + color')

# Fixed and random effects
model.fit('y ~ 0 + gender + condition*age', random=['1|subject'])
```

Each of the above examples specifies a full model that will immediately be fitted using the NUTS sampler implemented in PyMC3 (more on that below).

Notice how, in contrast to lme4 (but similar to nlme), fixed and random effects are specified separately in Bambi. We describe the syntax and operators supported by each type of effect below; briefly, however, the fixed effects specification relies on [patsy](http://patsy.readthedocs.io/en/latest/overview.html), and [hence formulas are parsed](http://patsy.readthedocs.io/en/latest/formulas.html) almost exactly the same way [as in R](http://patsy.readthedocs.io/en/latest/R-comparison.html). Random effects terms must be specified one at a time, and currently only support simple nesting or crossing relationships (e.g., '1|subject', 'condition|stimulus', etc.).

#### Term-based specification
An alternative approach that is more verbose but potentially clearer and more flexible is to enter each term into the model separately. The `add_term()` method provides a simple but powerful interface for specifying a range of fixed and random effects.

```python
from bambi import Model, Prior

# Initialize model
model = Model(data)
# Add intercept
model.add_intercept()
# Continuous fixed effect (in this case, a binary indicator)
model.add_term('condition')
# Categorical fixed effect, setting a narrow prior
model.add_term('age_group', categorical=True, prior='narrow')
# Random subject intercepts. Note that term label can be
# explicitly set if we don't want to use the dataset column name
model.add_term('subj', random=True, categorical=True, label='subject')
# Random condition slopes distributed over subjects
model.add_term('condition', random=True, over='subj')
# Add outcome
model.add_y('y')
# Fit the model and save results
results = model.fit()
```

As the above example illustrates, the only mandatory argument to `add_term` is a string giving the name of the dataset column to use for the term. If no other arguments are specified, the corresponding variable will be modeled as a fixed effect with a normally-distributed prior (a detailed explanation of how priors are handled in Bambi can be found below). The type of variable (i.e., categorical or continuous) will be determined based on the `dtype` of the column in the pandas `DataFrame`, so it's a good idea to make sure all variables are assigned the correct `dtype` when you first read in the data. You can also force a continuous variable to be treated as a categorical factor by passing `categorical=True` (e.g., `add_term('subject', categorical=True)`).

To specify that a term should be modeled as a random effect, simply set `random=True`. Whether the term is interpreted as coding random intercepts or random slopes then depends on the other arguments. When `random=True` and `over` is specified, Bambi will add random slopes for the named variable distributed over the column named in `over`. For example, `add_term('condition', random=True, over='subject')` will add random condition slopes distributed over subjects.

#### Mixing specification approaches
The two approaches to model specification described above are fully compatible with one another. While the `fit()` interface will typically be used to fit a model in one shot, we can also add only part of a model via the `add_formula()` interface, which allows us to add additional terms later via either additional formulas, or individual term specifications. Consider the following model construction:

```python
model = Model(data)
model.add_formula('condition + gender', random='condition|subject')
model.add_term('age')
model.add_y('rt')
```

Here we add two fixed predictors, as well as random intercepts and slopes for subjects, to the model via an initial formula call; then we add another fixed covariate (age) via `add_term()`; finally, we add the outcome variable.

We could, of course, just as easily have specified the equivalent model in a single line:

```
model = Model(data)
model.fit('rt ~ condition + gender + age', random='condition|subject')
```

In this case, the latter is probably a more sensible approach. However, as we illustrate later, there may be cases where using a mixed approach like the first one is more convenient--e.g., because we want to use a custom prior for only one or two terms.

#### Notes on fixed and random effects in Bambi
As noted above, Bambi handles fixed and random effects separately. The fixed effects specification relies on the [patsy](http://patsy.readthedocs.io/en/latest/overview.html) package, which supports nearly all of the standard formula operators handled in base R--including `:`, `*`, `-`, etc. Unfortunately, patsy doesn't support grouping operators, so random effects are handled separately in Bambi. At present, random effects support is limited to simple specification of slopes and intercepts. All terms must be passed in as elements in a list. For example:

```python
random_terms = [
    '1|student',    # Random student intercepts
    'classroom',     # Random classroom intercepts; equivalent to '1|stimulus'
    'treatment|school' # Random treatment slopes distributed over schools; school intercepts will also be automtically added.
]
model.add_formula(random=random_terms)
```

##### Coding of categorical variables
When a categorical fixed effect with N levels is added to a model, by default, it is coded by N-1 dummy variables (i.e., reduced rank coding). For example, suppose we write `'y ~ condition + age + gender'`, where condition is a categorical variable with 4 levels, and age and gender are continuous variables. Then our model would contain an intercept term (added to the model by default, as in R), three dummy-coded variables (each contrasting the first level of `condition` with one of the subsequent levels), and continuous predictors for age and gender. Suppose, however, that we would rather use full-rank coding of conditions. If we explicitly remove the intercept--as in `'y ~ 0 + condition + age + gender'`--then we get the desired effect. Now, the intercept is no longer included, and condition will be coded using 4 dummy indicators--each one coding for the presence or absence of the respective condition, without reference to the other conditions.

Random effects are handled in a comparable way. When adding random intercepts, coding is always full-rank (e.g., when adding random intercepts for 100 schools, one gets 100 dummy-coded indicators coding each school separately, and not 99 indicators contrasting each school with the very first one). For random slopes, coding proceeds the same way as for fixed effects. The random effects specification `['condition|subject']` would add an intercept for each subject, plus N-1 condition slopes (each coded with respect to the first, omitted, level as the referent). If we instead specify `['0+condition|subject']`, we get N condition slopes and no intercepts.

Note that the above only holds for the formula-based specification approach. When using the `add_term()` interface, we can explicitly control how categorical variables are coded using the `drop_first` argument:

```python
# Add random subject intercept--1 per subject
model.add_term('subject', random=True, categorical=True, drop_first=False)
# Add N - 1 random condition slopes, each distributed over all subjects
model.add_term('condition', random=True, categorical=True, drop_first=False)
```

#### Model specification notes
##### Term naming caveats
You might notice that the `add_term()` method takes both a `variable` and a `label` argument. The former refers to the name of the dataset column containing the data for the term; the latter is the name to use for the term when storing it in the model. By default, `label=None`, and the name of the column passed in `variable` will be re-used as the term label. However, there is at least one situation where this can potentially bite us in the ass: when adding random slopes or crossed random intercepts to the model--i.e., variables that are distributed over other variables--the term label will include *both* variables, to make sure that the generated term is kept distinct from other terms that might use the same underlying data. For example:

```python
# This will add a term named 'condition'
model.add_term('condition')
# This will generate a term named 'condition|subject', and *not* 'condition'!
model.add_term('condition', random=True, over='subject')
```

To avoid any potential conflicts, we generally recommend giving the terms in your model sensible names by explicitly setting the `label` value whenever appropriate. (This is not possible when using the formula interface, as term names will be generated dynamically based on the formula specification.)

### Fitting the model
Once a model is fully specified, we need to run the PyMC3 sampler to generate parameter estimates. If we're using the one-line `fit()` interface, sampling will begin right away:

```python
model = Model(data)
results = model.fit('rt ~ condition + gender + age', random='condition|subject')
```

The above code will obtain 1,000 samples (the default value) and return them as a `ModelResults` instance (for more details, see the [Results](#results) section). The `fit()` method accepts optional keyword arguments to pass onto PyMC3's `sample()` method, so any methods accepted by `sample()` can be specified here. We can also explicitly set the number of samples via the `samples` argument. For example, if we call `fit('y ~ X1', samples=2000, njobs=2)`, the PyMC3 sampler will run 2 parallel jobs, drawing 2,000 samples for each one. We could also specify starting parameter values, the step function to use, and so on (for full details, see the [PyMC3 documentation](https://pymc-devs.github.io/pymc3/api.html#pymc3.sampling.sample)).

Alternatively, we can specify our model in steps, and only call `fit()` once the model is complete:

```python
# Initializing with intercept=True adds an intercept right away
model = Model(data, intercept=True)
model.add_term('food_type', categorical=True)
model.add_term('subject', categorical=True, random=True)
...
model.fit(samples=5000)
```

#### Building the model
When `fit()` is called, Bambi internally performs two separate steps. First, the model is built or compiled, via a `build()` call. During the build, the PyMC3 model is compiled by Theano, in order to optimize the underlying Theano graph and improve sampling efficiency. This process can be fairly time-consuming, depending on the size and complexity of the model. It's possible to build the model explicitly, without beginning the sampling process, by calling `build()` directly on the model:

```python
model = Model(data)
model.add.formula('rt ~ condition + gender + age', random='condition|subject')
model.build()
```

The above code compiles the model, but doesn't begin sampling. This can be useful if we want to inspect the internal PyMC3 model before we start the (potentially long) sampling process. Once we're satisfied, and wish to run the sampler, we can then simply call `model.fit()`, and the sampler will start running.

### Specifying priors
Bayesian inference requires one to specify *prior* probability distributions that represent the analyst's belief (in advance of seeing the data) about the likely values of the model parameters. In practice, analysts often lack sufficient information to formulate well-defined priors, and instead opt to use "weakly informative" priors that mainly serve to keep the model from exploring completely pathological parts of the parameter space (e.g., when defining a prior on the distribution of human heights, a value of 3,000 cms should be assigned a probability of exactly 0).

By default, Bambi will intelligently generate weakly informative priors for all model terms, by loosely scaling them to the observed data. While the default priors will behave well in most typical settings, there are many cases where an analyst will want to specify their own priors--and in general, when informative priors are available, it's a good idea to use them.

#### Different ways of specifying priors

Bambi provides two ways to specify a custom prior. First, one can manually specify only the scale of the prior, while retaining the default distribution. 
By default, Bambi sets a "wide" prior on all fixed and random effects. Priors are specified on a partial correlation scale that quantifies the expected standardized contribution of each individual term to the outcome variable when controlling for other terms. The default wide prior sets the scale of the prior distribution (either a normal or a cauchy distribution, depending on the type of term) to sqrt(1/3). In cases where we want to keep the default prior distributions, but alter their scale, we can specify either a numeric scale value or pass the name of a predefined constant. For example:

```python
model = Model(data)
# Add condition to the model as a fixed effect with a very wide prior
model.add_term('condition', prior='superwide')
# Add random subject slopes to the model, with a narrow prior on their variance
model.add_term('subject', random=True, prior=0.1)
```

Named scales include 'superwide' (scale = 0.8), 'wide' (sqrt(1/3)), 'medium' (0.4), and 'narrow' (0.2). The ability to specify prior scales this way is helpful, but also limited: we will sometimes find ourselves wanting to use something other than a normal or cauchy distribution to model our priors. Fortunately, Bambi is built on top of PyMC3, which means that we can seamlessly use any of the over 40 `Distribution` classes defined in PyMC3. We can specify such priors in Bambi using the `Prior` class, which initializes with a `name` argument (which must map on exactly to the name of a valid PyMC3 `Distribution`) followed by any of the parameters accepted by the corresponding `distribution`. For example:

```python
from bambi import Prior
# A laplace prior with mean of 0 and scale of 10
my_favorite_prior = Prior('Laplace', mu=0., b=10)

# Set the prior when adding a term to the model
model.add_term('subject', random=True, prior=my_favorite_prior)
```

Priors specified using the `Prior` class can be nested to arbitrary depths--meaning, we can set any of a given prior's argument to point to another `Prior` instance. This is particularly useful when specifying hierarchical priors on random effects, where the individual random slopes or intercepts are constrained to share a common source distribution:

```python
subject_sd = Prior('HalfCauchy', mu=0, beta=5)
subject_prior = Prior('Normal', mu=0, sd=subject_sd)
model.add_term('subject', random=True, prior=subject_prior)
```

The above prior specification indicates that each individual subject's intercept should be modeled as a normal distribution with mean of 0, where the the individual intercepts are assumed to be randomly sampled from a higher-order half-cauchy distribution with a beta of 5.

#### Mapping priors onto terms
Bambi provides several different ways to map custom priors onto their corresponding model terms. The most convenient approach is probably to use the `Model` instance's `set_priors()` method, which allows us to easily apply multiple priors to any/all of the terms that have already been added to the model. There are also `fixed` and `random` arguments that make it easy to apply the same priors to all fixed or random effects in the model. For example:

```python
model = Model(data)
model.add_formula('y ~ X1 + X2', random=['1|X3', '1|X4'])

# Example 1: set each prior by name
model.set_priors({
    'X1': 0.3,
    'X2': 'normal',
    'X3': Prior('ZeroInflatedPoisson', theta=10, psi=0.5)
})

# Example 2: specify priors for all fixed effects and all random effects
model.set_priors(fixed=Prior('Normal', sd=100), random='wide')
```

Notice how this interface allows us to specify terms either by name (including passing tuples as keys in cases where we want multiple terms to share the same prior), or by term type (i.e., to set the same prior on all fixed or random effects). If we pass both named priors and fixed or random effects defaults, the former will take precedence over the latter.

If we prefer, we can also pass a full set of priors to the `fit()` call, in the `priors` argument:

```python
priors = {
    'names': {
        ('X1', 'X4'): Prior('Normal', sd=70),
        'X2': Prior('Normal', sd=Prior('Uniform', lower=10, upper=100))
    },
    'fixed': 0.5
}
model.fit('y ~ X1 + X3 + X4', random='1|X2', priors=priors)
```

Here we stipulate that terms X1 and X4 will use the same normal prior, X2 will use a different normal prior with a uniform hyperprior on its standard deviation, and all other fixed effects will use the default prior with a scale of 0.5.

Finally, and as we've already seen in other examples above, each term's prior can be set when adding it to the model with `add_term()`:

```
random_prior = Prior('Normal', sd=Prior('Uniform', lower=10, upper=100))
model.add_term('subject', random=True, prior=random_prior)
```

It's important to note that explicitly setting priors by passing in `Prior` objects will disable Bambi's default behavior of scaling priors to the data in order to ensure that they remain weakly informative. This means that if you specify your own prior, you have to be sure not only to specify the distribution you want, but also any relevant scale parameters. For example, the 0.5 in `Prior('Normal', mu=0, sd=0.5)` will be specified on the scale of the data, not the bounded partial correlation scale that Bambi uses for default priors. This means that if your outcome variable has a mean value of 10,000 and a standard deviation of, say, 1,000, you could potentially have some problems getting the model to produce reasonable estimates, since from the perspective of the data, you're specifying an extremely strong prior.

### Generalized linear mixed models
Bambi supports the construction of mixed models with non-normal error distributions (i.e., generalized linear mixed models, or GLMMs). GLMMs are specified in the same way as LMMs, except that the user must specify the distribution to use for the errors, and (optionally) the link function with which to transform the linear model prediction into the desired non-normal response. The easiest way to construct a GLMM is to simple set the `family` argument in the `fit()` call:

```python
model = Model(data)
model.fit('graduate ~ attendance_record + GPA', random='1|school', family='binomial')
```

If no `link` argument is explicitly set (see below), a sensible default will be used. The following table summarizes the currently available families and their associated links (the default is `gaussian`):

| Family name | Response distribution | Default link |
|-------------|-----------------------|--------------|
| gaussian    | Normal                | identity     |
| binomial    | Bernoulli             | logit        |
| poisson     | Poisson               | log          |
| t           | StudentT              | identity     |

#### Families
Following the convention used in many R packages, the error distribution to use for a GLMM is specified in a `Family` class that indicates how the response variable and its error are distributed, as well as the link function transforming the linear response to a non-linear one. Although the easiest way to specify a family is by name, using one of the options listed in the table above, users can also create and use their own family, providing enormous flexibility. In the following example, we show how the built-in 'binomial' family could be constructed on-the-fly:

```python
from bambi import Family, Prior
import theano.tensor as tt

# Specify how the Bernoulli p parameter is distributed
prior_p = Prior('Beta', alpha=2, beta=2)

# The response variable distribution
prior = Prior('Bernoulli', p=prior_p)

# Set the link function. Alternatively, we could just set
# the link to 'logit', since it's already built into Bambi.
# Note that we could pass in our own function here; the link
# function doesn't have to be predefined.
link = tt.nnet.sigmoid

# Construct the family
new_fam = Family('binomial', prior=prior, link=link, parent='p')

# Now it's business as usual
model = Model(data)
model.fit('graduate ~ attendance_record + GPA', random='1|school', family=new_fam)
```

The above example produces results identical to simply setting `family='binomial'`.

One (minor) complication in specifying a custom `Family` is that the link function must be able to operate over theano tensors rather than numpy arrays, so you'll probably need to rely on tensor operations provided in `theano.tensor` (many of which are also wrapped by PyMC3) when defining a new link.

### Results
When a model is fitted, it returns a `ModelResults` object containing methods for plotting and summarizing results. At present, functionality here is admittedly pretty thin, and Bambi simply wraps the most common plotting and summarization functions used in PyMC3.

### Plotting
To visualize a PyMC3-generated plot of the posterior estimates and sample traces for all parameters, simply call the result object's `.plot()` method:

```python
model = Model(data)
results = model.fit('value ~ condition', random='1|uid')
# Drop the first 500 burn-in samples from the plot
results.plot(500)
```

This produces a plot like the following:
![Sample PyMC3 trace plot](https://github.com/bambinos/bambi/blob/master/bambi/docs/images/sample_traceplot.png)

More details on this plot are available in the [PyMC3 documentation](http://pymc-devs.github.io/pymc3/notebooks/getting_started.html#Posterior-analysis).

### Summarizing
If you prefer numerical summaries of the posterior estimates, Bambi provides access to PyMC3's `summary()` function:

```python
# Omit the random effects intercepts because there's > 1,700 of them
names = ['b_Intercept', 'b_condition', 'u_uid_sd', 'likelihood_sd']
results.summary(500, names=names)
```

This produces a table that provides key descriptive statistics (including the 95% highest posterior density interval):

<pre>
b_Intercept:

  Mean             SD               MC Error         95% HPD interval
  -------------------------------------------------------------------
  
  4.137            0.973            0.097            [1.766, 4.754]

  Posterior quantiles:
  2.5            25             50             75             97.5
  |--------------|==============|==============|--------------|
  
  0.908          4.475          4.542          4.579          4.703


b_condition:

  Mean             SD               MC Error         95% HPD interval
  -------------------------------------------------------------------
  
  0.248            0.600            0.060            [-0.193, 1.795]

  Posterior quantiles:
  2.5            25             50             75             97.5
  |--------------|==============|==============|--------------|
  
  -0.228         -0.048         -0.007         0.103          1.770


u_uid_sd:

  Mean             SD               MC Error         95% HPD interval
  -------------------------------------------------------------------
  
  0.517            0.536            0.054            [0.044, 1.234]

  Posterior quantiles:
  2.5            25             50             75             97.5
  |--------------|==============|==============|--------------|
  
  0.048          0.072          0.104          1.177          1.247


likelihood_sd:

  Mean             SD               MC Error         95% HPD interval
  -------------------------------------------------------------------
  
  2.345            0.216            0.022            [2.079, 2.719]

  Posterior quantiles:
  2.5            25             50             75             97.5
  |--------------|==============|==============|--------------|
  
  2.077          2.112          2.393          2.452          2.718
</pre>

### Accessing back-end objects
Bambi is just a high-level interface to other statistical packages; as such, it uses other packages as computational back-ends. Internally, Bambi stores virtually all objects generated by backends like PyMC3, making it easy for users to retrieve, inspect, and modify those objects. For example, the `Model` class created by PyMC3 (as opposed to the Bambi class of the same name) is accessible from `model.backend.model`. For models fitted with a PyMC3 sampler, the resulting `MultiTrace` object is stored in `model.backend.trace` (though it can also be accessed via Bambi's `ModelResults` instance).
