import logging

import numpy as np
import theano.tensor as tt
import pymc3 as pm

from bambi.priors import Prior
import bambi.version as version

from .base import BackEnd
from .utils import probit, cloglog

_log = logging.getLogger("bambi")


class PyMC3BackEnd(BackEnd):
    """PyMC3 model-fitting backend."""

    # Available link functions
    links = {
        "identity": lambda x: x,
        "logit": tt.nnet.sigmoid,
        "probit": probit,
        "cloglog": cloglog,
        "inverse": tt.inv,
        "inverse_squared": lambda x: tt.inv(tt.sqrt(x)),
        "log": tt.exp,
    }

    dists = {"HalfFlat": pm.Bound(pm.Flat, lower=0)}

    def __init__(self):
        self.name = pm.__name__
        self.version = pm.__version__

        # Attributes defined elsewhere
        self.model = None
        self.mu = None  # build()
        self.spec = None  # build()
        self.trace = None  # build()
        self.advi_params = None  # build()
        self.fit = False  # run()

    def build(self, spec):  # pylint: disable=arguments-differ
        """Compile the PyMC3 model from an abstract model specification.

        Parameters
        ----------
        spec : Bambi model
            A Bambi ``Model`` instance containing the abstract specification of the model
            to compile.
        """

        coords = spec._get_pymc_coords()  # pylint: disable=protected-access
        self.model = pm.Model(coords=coords)
        noncentered = spec.noncentered

        with self.model:
            self.mu = 0.0
            for term in spec.terms.values():
                data = term.data
                label = term.name
                name = term.prior.name
                args = term.prior.args
                shape = term.data.shape[1]

                if term.group_specific:
                    # Group-specific terms always have pymc_coords, at least for the group.
                    dims = list(term.pymc_coords.keys())
                    coef = self.build_group_specific_distribution(
                        name, label, noncentered, dims=dims, **args
                    )
                    # term.predictor.shape[1] is larger than one when the expression is a
                    # categorical variable with more than one level.
                    # This is not the most beautiful alternative, but it resulted to be the
                    # fastest. Doing matrix multiplication, pm.math.dot(data, coef), is slower.
                    coef_ = coef[term.group_index]
                    if term.predictor.shape[1] > 1:
                        for col in range(term.predictor.shape[1]):
                            self.mu += coef_[:, col][:, None] * term.predictor[:, col]
                    else:
                        self.mu += coef_[:, None] * term.predictor
                else:
                    if term.pymc_coords:
                        # Common effects have at most ONE coord.
                        dims = list(term.pymc_coords.keys())
                        coef = self.build_common_distribution(name, label, dims=dims, **args)
                    else:
                        shape = () if shape == 1 else shape
                        coef = self.build_common_distribution(name, label, shape=shape, **args)
                    self.mu += pm.math.dot(data, coef)[:, None]

            self._build_response(spec)
            self.spec = spec

    # pylint: disable=arguments-differ, inconsistent-return-statements
    def run(
        self, start=None, method="mcmc", init="auto", n_init=50000, omit_offsets=True, **kwargs
    ):
        """Run the PyMC3 MCMC sampler.

        Parameters
        ----------
        start: dict, or array of dict
            Starting parameter values to pass to sampler; see ``pm.sample()`` for details.
        method: str
            The method to use for fitting the model. By default, ``'mcmc'``, in which case the
            PyMC3 sampler will be used. Alternatively, ``'advi'``, in which case the model will be
            fitted using  automatic differentiation variational inference as implemented in PyMC3.
            Finally, ``'laplace'``, in which case a laplace approximation is used, ``'laplace'`` is
            not recommended other than for pedagogical use.
        init: str
            Initialization method (see PyMC3 sampler documentation). Currently, this is
            ``'jitter+adapt_diag'``, but this can change in the future.
        n_init: int
            Number of initialization iterations if ``init = 'advi'`` or '``init = 'nuts'``.
            Default is kind of in PyMC3 for the kinds of models we expect to see run with Bambi,
            so we lower it considerably.
        omit_offsets: bool
            Omits offset terms in the ``InferenceData`` object when the model includes
            group specific effects. Defaults to ``True``.

        Returns
        -------
        An ArviZ ``InferenceData`` instance.
        """
        model = self.model

        if method.lower() == "mcmc":
            draws = kwargs.pop("draws", 1000)
            with model:
                idata = pm.sample(
                    draws,
                    start=start,
                    init=init,
                    n_init=n_init,
                    return_inferencedata=True,
                    **kwargs,
                )

            if omit_offsets:
                offset_vars = [var for var in idata.posterior.var() if var.endswith("_offset")]
                idata.posterior = idata.posterior.drop_vars(offset_vars)

            for group in idata.groups():
                getattr(idata, group).attrs["modeling_interface"] = "bambi"
                getattr(idata, group).attrs["modeling_interface_version"] = version.__version__

            # Reorder coords
            # pylint: disable=protected-access
            coords_original = list(self.spec._get_pymc_coords().keys())
            coords_group = [c for c in coords_original if c.endswith("_coord_group_factor")]
            for coord in coords_group:
                coords_original.remove(coord)
            coords_new = ["chain", "draw"] + coords_original + coords_group
            idata.posterior = idata.posterior.transpose(*coords_new)

            self.fit = True
            return idata

        elif method.lower() == "advi":
            with model:
                self.advi_params = pm.variational.ADVI(start, **kwargs)
            # this should return an InferenceData object (once arviz adds support for VI)
            return self.advi_params

        elif method.lower() == "laplace":
            return _laplace(model)

    def build_common_distribution(self, dist, label, **kwargs):
        """Build and return a PyMC3 Distribution for a common term."""
        # We are sure prior arguments aren't hyperpriors because it is already checked in the model
        distribution = self.get_distribution(dist)
        return distribution(label, **kwargs)

    def build_group_specific_distribution(self, dist, label, noncentered, **kwargs):
        """Build and return a PyMC3 Distribution."""
        dist = self.get_distribution(dist)
        if "dims" in kwargs:
            group_dim = [dim for dim in kwargs["dims"] if dim.endswith("_group_expr")]
            kwargs = {
                k: self.expand_prior_args(k, v, label, noncentered, dims=group_dim)
                for (k, v) in kwargs.items()
            }
        else:
            kwargs = {
                k: self.expand_prior_args(k, v, label, noncentered) for (k, v) in kwargs.items()
            }

        # Non-centered parameterization for hyperpriors
        if noncentered and has_hyperprior(kwargs):
            old_sigma = kwargs["sigma"]
            _offset = pm.Normal(label + "_offset", mu=0, sigma=1, dims=kwargs["dims"])
            return pm.Deterministic(label, _offset * old_sigma, dims=kwargs["dims"])

        return dist(label, **kwargs)

    def get_distribution(self, dist):
        """Return a PyMC3 distribution."""
        if isinstance(dist, str):
            if hasattr(pm, dist):
                dist = getattr(pm, dist)
            elif dist in self.dists:
                dist = self.dists[dist]
            else:
                raise ValueError(
                    f"The Distribution {dist} was not found in PyMC3 or the PyMC3BackEnd."
                )
        return dist

    def _build_response(self, spec):
        """Build and return a response distribution."""

        data = spec.response.data
        name = spec.response.name
        prior = spec.family.prior
        link = spec.family.link
        if isinstance(link, str):
            link = self.links[link]
        prior.args[spec.family.parent] = link(self.mu)
        prior.args["observed"] = data

        dist = self.get_distribution(prior.name)
        kwargs = {k: self.expand_prior_args(k, v, name, False) for (k, v) in prior.args.items()}

        if spec.family.name == "gamma":
            # Gamma distribution is specified using mu and sigma, but we request prior for alpha.
            # We need to build sigma from mu and alpha.
            # kwargs["mu"] ** 2 / kwargs["alpha"] would also work
            beta = kwargs["alpha"] / kwargs["mu"]
            sigma = (kwargs["mu"] / beta) ** 0.5
            return dist(name, mu=kwargs["mu"], sigma=sigma, observed=kwargs["observed"])

        return dist(name, **kwargs)

    def expand_prior_args(self, key, value, label, noncentered, **kwargs):
        # Inspect all args in case we have hyperparameters
        # kwargs are used to pass 'dims' for group specific terms.
        if isinstance(value, Prior):
            return self.build_group_specific_distribution(
                value.name, f"{label}_{key}", noncentered, **value.args, **kwargs
            )
        return value


def _laplace(model):
    """Fit a model using a Laplace approximation.

    Mainly for pedagogical use. ``mcmc`` and ``advi`` are better approximations.

    Parameters
    ----------
    model: PyMC3 model

    Returns
    -------
    Dictionary, the keys are the names of the variables and the values tuples of modes and standard
    deviations.
    """
    with model:
        varis = [v for v in model.unobserved_RVs if not pm.util.is_transformed_name(v.name)]
        maps = pm.find_MAP(start=model.test_point, vars=varis)
        hessian = pm.find_hessian(maps, vars=varis)
        if np.linalg.det(hessian) == 0:
            raise np.linalg.LinAlgError("Singular matrix. Use mcmc or advi method")
        stds = np.diag(np.linalg.inv(hessian) ** 0.5)
        maps = [v for (k, v) in maps.items() if not pm.util.is_transformed_name(k)]
        modes = [v.item() if v.size == 1 else v for v in maps]
        names = [v.name for v in varis]
        shapes = [np.atleast_1d(mode).shape for mode in modes]
        stds_reshaped = []
        idx0 = 0
        for shape in shapes:
            idx1 = idx0 + sum(shape)
            stds_reshaped.append(np.reshape(stds[idx0:idx1], shape))
            idx0 = idx1
    return dict(zip(names, zip(modes, stds_reshaped)))


def has_hyperprior(kwargs):
    """Determines if a Prior has an hyperprior"""

    return (
        "sigma" in kwargs
        and "observed" not in kwargs
        and isinstance(kwargs["sigma"], pm.model.TransformedRV)
    )
