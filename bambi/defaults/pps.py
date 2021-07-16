"""Utility functions to sample from the posterior predictive distributions"""

import numpy as np

from scipy import stats


def _get_mu_and_idxs(mu, draws): #pylint: disable=unused-argument
    # mu has shape (chain, draw, obs)
    idxs = np.random.randint(low=0, high=draws, size=draws)
    mu = mu[:, idxs, :]
    return mu, idxs


def pps_bernoulli(model, posterior, mu, draws): #pylint: disable=unused-argument
    mu, _ = _get_mu_and_idxs(mu, draws)
    return np.random.binomial(1, mu)


def pps_beta(model, posterior, mu, draws):
    mu, idxs = _get_mu_and_idxs(mu, draws)
    kappa = posterior[model.response.name + "_kappa"].values[:, idxs, np.newaxis]
    alpha = mu * kappa
    beta = (1 - mu) * kappa
    return np.random.beta(alpha, beta)


def pps_gamma(model, posterior, mu, draws):
    mu, idxs = _get_mu_and_idxs(mu, draws)
    alpha = posterior[model.response.name + "_alpha"].values[:, idxs, np.newaxis]
    beta = alpha / mu
    return np.random.gamma(alpha, 1 / beta)


def pps_gaussian(model, posterior, mu, draws):
    mu, idxs = _get_mu_and_idxs(mu, draws)
    sigma = posterior[model.response.name + "_sigma"].values[:, idxs, np.newaxis]
    return np.random.normal(mu, sigma)


def pps_negativebinomial(model, posterior, mu, draws):
    mu, idxs = _get_mu_and_idxs(mu, draws)
    n = posterior[model.response.name + "_alpha"].values[:, idxs, np.newaxis]
    p = n / (mu + n)
    return np.random.negative_binomial(n, p)


def pps_poisson(model, posterior, mu, draws): #pylint: disable=unused-argument
    mu, _ = _get_mu_and_idxs(mu, draws)
    return np.random.poisson(mu)


def pps_t(model, posterior, mu, draws):
    mu, idxs = _get_mu_and_idxs(mu, draws)
    if isinstance(model.family.likelihood.priors["nu"], (int, float)):
        nu = model.family.likelihood.priors["nu"]
    else:
        nu = posterior[model.response.name + "_nu"].values[:, idxs, np.newaxis]

    lam = posterior[model.response.name + "_lam"].values[:, idxs, np.newaxis]
    sigma = lam ** -0.5
    return stats.t.rvs(nu, mu, sigma)


def pps_wald(model, posterior, mu, draws):
    mu, idxs = _get_mu_and_idxs(mu, draws)
    lam = posterior[model.response.name + "_lam"].values[:, idxs, np.newaxis]
    return np.random.wald(mu, lam)
