import inspect

import pytensor.tensor as pt
import pymc as pm

MAPPING = {"Cumulative": pm.Categorical, "StoppingRatio": pm.Categorical}


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


def has_hyperprior(kwargs):
    """Determines if a Prior has an hyperprior"""
    return (
        "sigma" in kwargs
        and "observed" not in kwargs
        and isinstance(kwargs["sigma"], pt.TensorVariable)
    )


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


def get_linkinv(link, invlinks):
    """Get the inverse of the link function as needed by PyMC

    Parameters
    ----------
    link : bmb.Link
        A link function object. It may contain the linkinv function that the backend uses.
    invlinks : dict
        Keys are names of link functions. Values are the built-in link functions.

    Returns
    -------
        callable
        The link function
    """
    # If the name is in the backend, get it from there
    if link.name in invlinks:
        invlink = invlinks[link.name]
    # If not, use whatever is in `linkinv_backend`
    else:
        invlink = link.linkinv_backend
    return invlink


def exp_quad(sigma, ell, input_dim=1):
    return sigma**2 * pm.gp.cov.ExpQuad(input_dim, ls=ell)


def matern32(sigma, ell, input_dim=1):
    return sigma**2 * pm.gp.cov.Matern32(input_dim, ls=ell)


def matern52(sigma, ell, input_dim=1):
    return sigma**2 * pm.gp.cov.Matern52(input_dim, ls=ell)


GP_KERNELS = {
    "ExpQuad": {"fn": exp_quad, "params": ("sigma", "ell")},
    "Matern32": {"fn": matern32, "params": ("sigma", "ell")},
    "Matern52": {"fn": matern52, "params": ("sigma", "ell")},
}


def make_logp_and_random(dist: pm.Distribution):
    """Create functions to compute a weighted logp and generate random draws for a distribution

    Parameters
    ----------
    dist : pm.Distribution
        The PyMC distribution for which we want to get the logp and random functions.

    Returns
    -------
    A tuple with two functions. The first computes the logp, the second generates random draws.
    """

    def logp(value, *dist_params):
        weights, *dist_params = dist_params
        return weights * dist.logp(value, *dist_params)

    def random(*dist_params, rng=None, size=None):
        # Weights don't make sense when generating new observations,
        # they are a property of observations
        _, *dist_params = dist_params
        rng_fn = getattr(rng, dist.rv_op.name)
        return rng_fn(*dist_params, size=size)

    return logp, random


def get_dist_args(dist: pm.Distribution) -> list[str]:
    """Get the argument names of a PyMC distribution.

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


def make_weighted_distribution(dist: pm.Distribution):
    logp, random = make_logp_and_random(dist)
    dist_args = get_dist_args(dist)

    # pylint: disable=protected-access
    try:
        dname = dist.rv_op._print_name[0] # pylint: disable=protected-access
    except:
        dname = "Dist"

    class WeightedDistribution:
        def __new__(cls, name, weights, **kwargs):
            # Get parameter values in the order required by the distribution as they are passed
            # by position to `pm.CustomDist`
            dist_params = [kwargs.pop(arg) for arg in dist_args if arg in kwargs]
            return pm.CustomDist(
                name,
                weights,
                *dist_params,
                logp=logp,
                random=random,
                class_name=f"Weighted{dname}",
                **kwargs,
            )

        @classmethod
        def dist(cls, **kwargs):
            dist_params = [kwargs.pop(arg) for arg in dist_args if arg in kwargs]
            if "weights" in kwargs:
                dist_params.insert(0, kwargs.pop("weights"))
            return pm.CustomDist.dist(
                *dist_params, logp=logp, random=random, class_name=f"Weighted{dname}", **kwargs
            )

    return WeightedDistribution


# WeightedNormal = make_weighted_distribution(pm.Normal)

# with pm.Model() as model:
#     weights = 2
#     mu = pm.Normal("mu", 0, 1)
#     sigma = pm.HalfNormal("sigma", 1)
#     WeightedNormal("y", weights, mu=mu, sigma=sigma, observed=np.random.randn(100))
#     idata = pm.sample_prior_predictive()
#     idata.extend(pm.sample(100))

# pm.draw(WeightedNormal.dist(mu=0, sigma=1), draws=10)
# pm.logp(WeightedNormal.dist(mu=0.25, sigma=2.5, weights=[1, 2.5]), value=[0, 0]).eval()
# pm.logp(pm.Normal.dist(mu=0.25, sigma=2.5), value=[0, 0]).eval()

# weights = 1 + np.random.poisson(lam=3, size=100)
# y = np.random.exponential(scale=3, size=100)

# WeightedExponential = make_weighted_dist(pm.Exponential)

# with pm.Model() as model:
#     lam = pm.math.exp(pm.Normal("log_lam", 1))
#     WeightedExponential("y", weights, lam=lam, observed=y)
#     idata = pm.sample_prior_predictive()
#     idata.extend(pm.sample(100))
# pm.draw(WeightedExponential.dist(lam=2), draws=10)
# pm.logp(WeightedExponential.dist(lam=2, weights=[1, 2.5]), value=[1, 1]).eval()
# pm.logp(pm.Exponential.dist(lam=2), value=[1, 1]).eval()
