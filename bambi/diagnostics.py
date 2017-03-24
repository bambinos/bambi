import numpy as np
import pandas as pd

# code adapted from pymc3.diagnostics:
# https://github.com/pymc-devs/pymc3/blob/master/pymc3/diagnostics.py


def _vhat_w(mcmc):
    # Calculate between-chain variance
    B = mcmc.n_samples * mcmc.data.mean(axis=0).var(axis=0, ddof=1)

    # Calculate within-chain variance
    W = mcmc.data.var(axis=0, ddof=1).mean(axis=0)

    # Estimate of marginal posterior variance
    Vhat = W * (mcmc.n_samples - 1) / mcmc.n_samples + B / mcmc.n_samples

    return Vhat, W


def gelman_rubin(mcmc):
    """
    mcmc (MCMCResults): Pre-sliced MCMC samples to compute diagnostics for.
    """

    if mcmc.n_chains < 2:
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains '
            'of the same length.')

    # get Vhat and within-chain variance
    Vhat, W = _vhat_w(mcmc)

    # compute and return Gelman-Rubin statistic
    Rhat = np.sqrt(Vhat / W)
    return pd.DataFrame({'gelman_rubin': Rhat}, index=mcmc.levels)


def effective_n(mcmc):
    """
    mcmc (MCMCResults): Pre-sliced MCMC samples to compute diagnostics for.
    """

    if mcmc.n_chains < 2:
        raise ValueError(
            'Calculation of effective sample size requires multiple chains '
            'of the same length.')

    def get_neff(x, Vhat):
        n_samples = x.shape[0]
        n_chains = x.shape[1]

        negative_autocorr = False
        t = 1

        rho = np.ones(n_samples)
        # Iterate until the sum of consecutive estimates of autocorrelation is
        # negative
        while not negative_autocorr and (t < n_samples):
            variogram = np.mean((x[t:, :] - x[:-t, :])**2)
            rho[t] = 1. - variogram / (2. * Vhat)
            negative_autocorr = sum(rho[t - 1:t + 1]) < 0
            t += 1

        if t % 2:
            t -= 1

        return min(n_chains * n_samples,
                   int(n_chains * n_samples / (1. + 2 * rho[1:t-1].sum())))

    Vhat, W = _vhat_w(mcmc)
    n_eff = [get_neff(mcmc.data[:, :, i], x) for i, x in enumerate(Vhat)]

    return pd.DataFrame({'effective_n': n_eff}, index=mcmc.levels)
