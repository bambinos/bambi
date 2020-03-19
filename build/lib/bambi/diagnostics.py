import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

# code adapted from pymc3.diagnostics:
# https://github.com/pymc-devs/pymc3/blob/master/pymc3/diagnostics.py


def autocorr(x):
    """
    Compute autocorrelation using FFT for every lag for the input array
    https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation.

    Args:
        x (array-like): An array containing MCMC samples.

    Returns:
        np.ndarray: An array of the same size as the input array.
    """

    y = x - x.mean()
    n = len(y)
    result = fftconvolve(y, y[::-1])
    acorr = result[len(result) // 2:]
    acorr /= np.arange(n, 0, -1)
    acorr /= acorr[0]
    return acorr


def autocov(x):
    """Compute autocovariance estimates for every lag for the input array.

    Args:
        x (array-like): An array containing MCMC samples.

    Returns:
        np.ndarray: An array of the same size as the input array.
    """

    acorr = autocorr(x)
    varx = np.var(x, ddof=1) * (len(x) - 1) / len(x)
    acov = acorr * varx
    return acov


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
    Args:
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
    Args:
        mcmc (MCMCResults): Pre-sliced MCMC samples to compute diagnostics for.
    """

    if mcmc.n_chains < 2:
        raise ValueError(
            'Calculation of effective sample size requires multiple chains '
            'of the same length.')

    def get_neff(x):
        """Compute the effective sample size for a 2D array."""
        trace_value = x.T
        nchain, n_samples = trace_value.shape

        acov = np.asarray([autocov(trace_value[chain]) for chain in range(nchain)])

        chain_mean = trace_value.mean(axis=1)
        chain_var = acov[:, 0] * n_samples / (n_samples - 1.)
        acov_t = acov[:, 1] * n_samples / (n_samples - 1.)
        mean_var = np.mean(chain_var)
        var_plus = mean_var * (n_samples - 1.) / n_samples
        var_plus += np.var(chain_mean, ddof=1)

        rho_hat_t = np.zeros(n_samples)
        rho_hat_even = 1.
        rho_hat_t[0] = rho_hat_even
        rho_hat_odd = 1. - (mean_var - np.mean(acov_t)) / var_plus
        rho_hat_t[1] = rho_hat_odd
        # Geyer's initial positive sequence
        max_t = 1
        t = 1
        while t < (n_samples - 2) and (rho_hat_even + rho_hat_odd) >= 0.:
            rho_hat_even = 1. - (mean_var - np.mean(acov[:, t + 1])) / var_plus
            rho_hat_odd = 1. - (mean_var - np.mean(acov[:, t + 2])) / var_plus
            if (rho_hat_even + rho_hat_odd) >= 0:
                rho_hat_t[t + 1] = rho_hat_even
                rho_hat_t[t + 2] = rho_hat_odd
            max_t = t + 2
            t += 2

        # Geyer's initial monotone sequence
        t = 3
        while t <= max_t - 2:
            if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
                rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.
                rho_hat_t[t + 2] = rho_hat_t[t + 1]
            t += 2
        ess = nchain * n_samples
        ess = ess / (-1. + 2. * np.sum(rho_hat_t))
        return ess

    nvar = mcmc.data.shape[-1]
    n_eff = [get_neff(mcmc.data[:, :, i]) for i in range(nvar)]

    return pd.DataFrame({'effective_n': n_eff}, index=mcmc.levels)
