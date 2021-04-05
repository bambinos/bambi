import logging

import numpy as np
import theano
import pymc3 as pm

from bambi.priors import Prior
import bambi.version as version
from .base import BackEnd

_log = logging.getLogger("bambi")


class PyMC3BackEnd(BackEnd):
    """PyMC3 model-fitting backend."""

    # Available link functions
    links = {
        "identity": lambda x: x,
        "logit": theano.tensor.nnet.sigmoid,
        "inverse": theano.tensor.inv,
        "inverse_squared": lambda x: theano.tensor.inv(theano.tensor.sqrt(x)),
        "log": theano.tensor.exp,
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

    # Inspect all args in case we have hyperparameters
    def _expand_args(self, key, value, label, noncentered):
        if isinstance(value, Prior):
            label = f"{label}_{key}"
            return self._build_dist(noncentered, label, value.name, **value.args)
        return value

    def _build_dist(self, noncentered, label, dist, **kwargs):
        """Build and return a PyMC3 Distribution."""
        if isinstance(dist, str):
            if hasattr(pm, dist):
                dist = getattr(pm, dist)
            elif dist in self.dists:
                dist = self.dists[dist]
            else:
                raise ValueError(
                    f"The Distribution {dist} was not found in PyMC3 or the PyMC3BackEnd."
                )

        kwargs = {k: self._expand_args(k, v, label, noncentered) for (k, v) in kwargs.items()}
        # Non-centered parameterization for hyperpriors
        if (
            noncentered
            and "sigma" in kwargs
            and "observed" not in kwargs
            and isinstance(kwargs["sigma"], pm.model.TransformedRV)
        ):
            old_sigma = kwargs["sigma"]
            _offset = pm.Normal(label + "_offset", mu=0, sigma=1, shape=kwargs["shape"])
            return pm.Deterministic(label, _offset * old_sigma)

        return dist(label, **kwargs)

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
                dist_name = term.prior.name
                dist_args = term.prior.args
                dist_shape = term.data.shape[1]
                if dist_shape == 1:
                    dist_shape = ()
                coef = self._build_dist(
                    noncentered, label, dist_name, shape=dist_shape, **dist_args
                )

                if term.group_specific:
                    self.mu += coef[term.group_index][:, None] * term.predictor
                else:
                    self.mu += pm.math.dot(data, coef)[:, None]

            response = spec.response.data
            response_name = spec.response.name
            response_prior = spec.family.prior
            link_f = spec.family.link
            if isinstance(link_f, str):
                link_f = self.links[link_f]
            response_prior.args[spec.family.parent] = link_f(self.mu)
            response_prior.args["observed"] = response
            self._build_dist(noncentered, response_name, response_prior.name, **response_prior.args)
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
                offset_dims = [vn for vn in idata.posterior.dims if "offset" in vn]
                idata.posterior = idata.posterior.drop_dims(offset_dims)

            for group in idata.groups():
                getattr(idata, group).attrs["modeling_interface"] = "bambi"
                getattr(idata, group).attrs["modeling_interface_version"] = version.__version__

            self.fit = True
            return idata

        elif method.lower() == "advi":
            with model:
                self.advi_params = pm.variational.ADVI(start, **kwargs)
            return (
                self.advi_params
            )  # this should return an InferenceData object (once arviz adds support for VI)

        elif method.lower() == "laplace":
            return _laplace(model)


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
