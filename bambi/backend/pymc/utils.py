import inspect
import functools

import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.dist_math import normal_lccdf
from pytensor.tensor.special import softmax

from bambi.backend.pymc.links import (
    cloglog,
    identity,
    inverse_squared,
    logit,
    probit,
)


def horseshoe(name, tau_nu=3, lam_nu=1, dims=None):
    """Define coefficients with a horseshoe prior.

    This is an internal helper that constructs the PyMC random variables
    corresponding to a horseshoe prior for regression coefficients.
    It is not intended to be called directly by users.

    Parameters
    ----------
    name : str
        Base name of the coefficient as registered in the PyMC model.
    tau_nu : int or float
        Degrees of freedom of the global scale parameter `tau`. Default is 3.
    lam_nu : int or float
        Degrees of freedom of the local scale parameter `lam`.
        Default is 1 (equivalent to a Half-Cauchy).
    dims : str or sequence of str, optional
        Dimensions passed to PyMC. Default is `None`.

    Returns
    -------
    pm.Deterministic
        Deterministic PyMC variable representing coefficients with a horseshoe prior.
    """
    tau = pm.HalfStudentT(f"{name}_tau", nu=tau_nu)
    lam = pm.HalfStudentT(f"{name}_lam", nu=lam_nu, dims=dims)
    beta_raw = pm.Normal(f"{name}_raw", 0, 1, dims=dims)
    beta = pm.Deterministic(name, beta_raw * tau * lam, dims=dims)
    return beta


MAPPING = {"Cumulative": pm.Categorical, "StoppingRatio": pm.Categorical, "Horseshoe": horseshoe}

INVERSE_LINKS = {
    "cloglog": cloglog,
    "identity": identity,
    "inverse_squared": inverse_squared,
    "inverse": pt.reciprocal,
    "log": pt.exp,
    "logit": logit,
    "probit": probit,
    "softmax": functools.partial(softmax, axis=-1),
}


def get_distribution(dist):
    """Return a PyMC distribution."""
    if isinstance(dist, str):
        if dist in MAPPING:
            dist = MAPPING[dist]
        elif hasattr(pm, dist):
            dist = getattr(pm, dist)
        else:
            raise ValueError(f"The Distribution '{dist}' was not found in PyMC")
    return dist


def get_distribution_from_prior(prior):
    if prior.dist is not None:
        distribution = prior.dist
    else:
        distribution = get_distribution(prior.name)
    return distribution


def get_distribution_from_likelihood(likelihood):
    """
    It works because both `Prior` and `Likelihood` instances have a `name` and a `dist` argument.
    """
    return get_distribution_from_prior(likelihood)


def make_weighted_logp(dist: pm.Distribution):
    """Create a function to compute a weighted logp

    Parameters
    ----------
    dist : pm.Distribution
        The PyMC distribution for which we want to get the weighted logp.

    Returns
    -------
    A function that computes the weighted logp.
    """

    # NOTE: Should we weight the logp or weight the p? This does the first.
    def logp(value, *dist_params, weights):
        weights = pt.as_tensor_variable(weights)
        return weights * pm.logp(dist.dist(*dist_params), value)

    return logp


def get_dist_args(dist: pm.Distribution) -> list[str]:
    """Get the argument names of a PyMC distribution

    The argument names are the names of the parameters of the distribution.

    Parameters
    ----------
    dist : pm.Distribution
        The PyMC distribution for which we want to extract the argument names.

    Returns
    -------
    list[str]
        The names of the arguments.
    """
    # Get all args but the first one which is usually 'cls'
    return inspect.getfullargspec(dist.dist).args[1:]


def create_cdist(dist: pm.Distribution):
    def fun(*params):
        *dist_params, size = params
        return dist.dist(*dist_params, size=size)

    return fun


def make_weighted_distribution(dist: pm.Distribution):
    wlogp = make_weighted_logp(dist)
    dist_args = get_dist_args(dist)
    cdist = create_cdist(dist)

    # Get distribution name in a safe way.
    rv_op = getattr(dist, "rv_op", None)
    print_name = getattr(rv_op, "_print_name", ())
    dist_name = print_name[0] if print_name else "Dist"
    class_name = f"Weighted{dist_name}"

    class WeightedDistribution:
        # We pass 'logp' to get the weighted logp, and we pass 'dist' to make sure
        # the random draws are generated using the correct parameter values.
        # Distribution.dist is the method that handles the parameters and with this approach
        # we are sure that we use it.
        def __new__(cls, name, weights, **kwargs):
            # Get parameter values in the order required by the distribution as they are passed
            # by position to `pm.CustomDist`
            dist_params = [kwargs.pop(arg) for arg in dist_args if arg in kwargs]
            return pm.CustomDist(
                name,
                *dist_params,
                logp=functools.partial(wlogp, weights=weights),
                dist=cdist,
                class_name=class_name,
                **kwargs,
            )

        @classmethod
        def dist(cls, **kwargs):
            dist_params = [kwargs.pop(arg) for arg in dist_args if arg in kwargs]
            weights = 1 if "weights" not in kwargs else kwargs.pop("weights")
            return pm.CustomDist.dist(
                *dist_params,
                logp=functools.partial(wlogp, weights=weights),
                dist=cdist,
                class_name=class_name,
                **kwargs,
            )

    return WeightedDistribution


def make_competing_risks_logp(dist: pm.Distribution, parameter_names: list[str]):
    """Create a right-censored competing-risks log-likelihood from a PyMC distribution.

    For an event of cause k, p(y, J=k) = f_k(y) prod_{j != k} S_j(y).
    For right censoring, p(T > y) = prod_j S_j(y).

    The distribution parameters must have a final axis indexing causes.
    Each cause follows the same distributional family,
    with its own parameters and conditionally independent latent event time.
    """

    def logp(value, *dist_params, status, cause):
        parameters = dict(zip(parameter_names, dist_params, strict=True))
        base_dist = dist.dist(**parameters)  # (n, K)
        value = pt.shape_padright(value)  # (n, 1)
        log_density = pm.logp(base_dist, value)  # (n, K)

        log_survival_by_cause = _compute_logccdf(dist, base_dist, value, dist_params)

        log_survival = pt.sum(log_survival_by_cause, axis=-1)  # (n, )

        # Causes use one-based codes, while right-censored rows use zero.
        # Clipping makes zero a valid dummy index for the selections below.
        cause_index = pt.maximum(cause - 1, 0)
        rows = pt.arange(log_density.shape[0])

        # Gather the density indexed by `cause`: log f_k(y)
        selected_density = log_density[rows, cause_index]

        # P(T = y, J = k) = f_k(y) prod_{j != k} S_j(y).
        # Sum the other causes directly rather than subtracting log S_k(y) from the total.
        # The latter can produce NaN when log S_k(y) and the total both underflow to -inf.
        is_selected_cause = pt.eq(pt.arange(log_density.shape[-1]), cause_index[:, None])
        log_survival_other_causes = pt.sum(
            pt.where(is_selected_cause, 0, log_survival_by_cause), axis=-1
        )
        event_logp = selected_density + log_survival_other_causes

        # Exact events use their cause-specific density. Right censoring uses total survival.
        return pt.switch(pt.eq(status, 0), event_logp, log_survival)

    return logp


def _compute_logccdf(
    dist: pm.Distribution, base_dist: pt.TensorVariable, value: pt.TensorVariable, params: tuple
) -> pt.TensorVariable:
    """Compute stable log survival probabilities."""
    if dist is pm.Exponential:
        (lam,) = params
        return -lam * value

    if dist is pm.Weibull:
        alpha, beta = params
        return -((value / beta) ** alpha)

    if dist is pm.Gamma:
        mu, sigma = params
        alpha = (mu / sigma) ** 2
        beta = mu / sigma**2
        return _log_gamma_survival(alpha, beta * value)

    if dist is pm.Wald:
        mu, lam = params
        q = value / mu
        root = pt.sqrt(value / lam)
        log_first_term = normal_lccdf(0, 1, (q - 1.0) / root)
        log_second_term = 2.0 * lam / mu + pm.logcdf(pm.Normal.dist(0, 1), -(q + 1.0) / root)
        return pm.math.logdiffexp(log_first_term, log_second_term)

    return pm.logccdf(base_dist, value)


def _log_gamma_survival(alpha: pt.TensorVariable, value: pt.TensorVariable) -> pt.TensorVariable:
    """Compute log(Q(alpha, value)), including a stable upper-tail approximation."""
    direct = pt.log(pt.gammaincc(alpha, value))

    # `gammaincc` underflows in the extreme upper tail.
    # There the first two terms of its asymptotic expansion are accurate and stay on the log scale.
    # Keep the direct expression whenever it is representable.
    # It is more accurate around the body of the distribution, especially for large alpha.
    tail = (
        -value + (alpha - 1.0) * pt.log(value) - pt.gammaln(alpha) + pt.log1p((alpha - 1.0) / value)
    )
    return pt.switch(pt.isinf(direct), tail, direct)


def create_competing_risks_dist(dist: pm.Distribution, parameter_names: list[str]):
    """Create the distribution of the first event among the competing causes."""

    def fun(*params):
        *dist_params, _ = params
        parameters = dict(zip(parameter_names, dist_params, strict=True))
        return pt.min(dist.dist(**parameters), axis=-1)

    return fun


def make_competing_risks_distribution(dist: pm.Distribution):
    """Wrap a PyMC distribution as a right-censored competing-risks likelihood."""
    dist_args = get_dist_args(dist)

    # Get distribution name in a safe way.
    rv_op = getattr(dist, "rv_op", None)
    print_name = getattr(rv_op, "_print_name", ())
    dist_name = print_name[0] if print_name else "Dist"
    class_name = f"CompetingRisks{dist_name}"

    class CompetingRisksDistribution:
        def __new__(cls, name, status, cause, **kwargs):
            parameter_names = [arg for arg in dist_args if arg in kwargs]
            dist_params = [kwargs.pop(arg) for arg in parameter_names]
            cr_logp = make_competing_risks_logp(dist, parameter_names)
            # The symbolic distribution derives random draws automatically.
            # logp is supplied separately.
            cr_dist = create_competing_risks_dist(dist, parameter_names)
            signature = ",".join("(cause)" for _ in parameter_names) + "->()"
            return pm.CustomDist(
                name,
                *dist_params,
                logp=functools.partial(cr_logp, status=status, cause=cause),
                dist=cr_dist,
                class_name=class_name,
                signature=signature,
                **kwargs,
            )

    return CompetingRisksDistribution
