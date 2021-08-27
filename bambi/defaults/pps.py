"""Utility functions to sample from the posterior predictive distributions"""

import numpy as np

from scipy import stats


def _get_mu_and_idxs(mu, draws, draw_n):  # pylint: disable=unused-argument
    """Sample values from the posterior of the mean, return sampled values and their indexes

    This function is used within ``pps_*`` auxiliary functions. Its goal is to simplify how we
    sample from the posterior trace and the other parameters that may define the conditional
    distribution of the response.

    Parameters
    ----------
    mu: np.array
        A 3-dimensional numpy array with samples from the posterior distribution of the mean. The
        first dimension represents the chain, the second represents the draw, and the third the
        individual.
    draws: int
        Number of samples to take.
    draw_n: int
        Number of draws available in per chain of the posterior

    Returns
    -------
    mu: np.array
        A 3-dimensional numpy array with the new samples.
    idxs: np.array
        The indexes of the samples in the original sample of the posterior.
    """
    # mu has shape (chain, draw, obs)
    idxs = np.random.randint(low=0, high=draw_n, size=draws)
    mu = mu[:, idxs, :]
    return mu, idxs


def pps_bernoulli(model, posterior, mu, draws, draw_n):  # pylint: disable=unused-argument
    mu, _ = _get_mu_and_idxs(mu, draws, draw_n)
    return np.random.binomial(1, mu)


def pps_beta(model, posterior, mu, draws, draw_n):
    mu, idxs = _get_mu_and_idxs(mu, draws, draw_n)
    kappa = posterior[model.response.name + "_kappa"].values[:, idxs, np.newaxis]
    alpha = mu * kappa
    beta = (1 - mu) * kappa
    return np.random.beta(alpha, beta)


def pps_binomial(
    model, posterior, mu, draws, draw_n, trials=None
):  # pylint: disable=unused-argument
    mu, _ = _get_mu_and_idxs(mu, draws, draw_n)
    if trials is None:
        trials = model.response.data[:, 1]
    return np.random.binomial(trials.squeeze(), mu)


def pps_gamma(model, posterior, mu, draws, draw_n):
    mu, idxs = _get_mu_and_idxs(mu, draws, draw_n)
    alpha = posterior[model.response.name + "_alpha"].values[:, idxs, np.newaxis]
    beta = alpha / mu
    return np.random.gamma(alpha, 1 / beta)


def pps_gaussian(model, posterior, mu, draws, draw_n):
    mu, idxs = _get_mu_and_idxs(mu, draws, draw_n)
    sigma = posterior[model.response.name + "_sigma"].values[:, idxs, np.newaxis]
    return np.random.normal(mu, sigma)


def pps_negativebinomial(model, posterior, mu, draws, draw_n):
    mu, idxs = _get_mu_and_idxs(mu, draws, draw_n)
    n = posterior[model.response.name + "_alpha"].values[:, idxs, np.newaxis]
    p = n / (mu + n)
    return np.random.negative_binomial(n, p)


def pps_poisson(model, posterior, mu, draws, draw_n):  # pylint: disable=unused-argument
    mu, _ = _get_mu_and_idxs(mu, draws, draw_n)
    return np.random.poisson(mu)


def pps_t(model, posterior, mu, draws, draw_n):
    mu, idxs = _get_mu_and_idxs(mu, draws, draw_n)
    if isinstance(model.family.likelihood.priors["nu"], (int, float)):
        nu = model.family.likelihood.priors["nu"]
    else:
        nu = posterior[model.response.name + "_nu"].values[:, idxs, np.newaxis]

    lam = posterior[model.response.name + "_lam"].values[:, idxs, np.newaxis]
    sigma = lam ** -0.5
    return stats.t.rvs(nu, mu, sigma)


def pps_wald(model, posterior, mu, draws, draw_n):
    mu, idxs = _get_mu_and_idxs(mu, draws, draw_n)
    lam = posterior[model.response.name + "_lam"].values[:, idxs, np.newaxis]
    return np.random.wald(mu, lam)
