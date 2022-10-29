import aesara.tensor as at
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


rng = np.random.default_rng(1234)

size = 1000
values = rng.normal(size=size)

left_censored = values < -2
interval_censored = np.logical_and(-0.2 < values, values < 0.2)
right_censored = values > 1.8
not_censored = ~np.logical_or.reduce((left_censored, interval_censored, right_censored))


print(sum(left_censored), sum(interval_censored), sum(right_censored), sum(not_censored))
sum(left_censored) + sum(interval_censored) + sum(right_censored) + sum(not_censored)


with pm.Model() as model:
    target = 0
    mu = pm.Normal("mu")
    sigma = pm.HalfNormal("sigma")

    # Contribution due to left censored values
    target += np.sum(left_censored) * pm.Normal.logcdf(-2, mu, sigma)

    # Contribution due to right censored values
    target += np.sum(right_censored) * at.log(1 - at.exp(pm.Normal.logcdf(1.8, mu, sigma)))

    # Contribution due to interval censored values
    target += np.sum(interval_censored) * (
        at.log(at.exp(pm.Normal.logcdf(0.2, mu, sigma)) - at.exp(pm.Normal.logcdf(-0.2, mu, sigma)))
    )

    # Contribution due to uncensored values
    likelihood = pm.Normal.dist(mu=mu, sigma=sigma)
    target += at.sum(pm.logp(likelihood, values[not_censored]))

    pm.Potential("y_potential", target)
    idata = pm.sample()


az.summary(idata)