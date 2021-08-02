import logging
import traceback

import numpy as np
import theano.tensor as tt
import pymc3 as pm

from bambi import version
from bambi.priors import Prior

from .base import BackEnd
from .utils import probit, cloglog

_log = logging.getLogger("bambi")


class PyMC3BackEnd(BackEnd):
    """PyMC3 model-fitting backend."""

    # Available inverse link functions
    INVLINKS = {
        "cloglog": cloglog,
        "identity": lambda x: x,
        "inverse_squared": lambda x: tt.inv(tt.sqrt(x)),
        "inverse": tt.inv,
        "log": tt.exp,
        "logit": tt.nnet.sigmoid,
        "probit": probit,
    }

    def __init__(self):
        self.name = pm.__name__
        self.version = pm.__version__

        # Attributes defined elsewhere
        self.model = None
        self.has_intercept = None  # build()
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

        self.has_intercept = spec.intercept_term is not None

        ## Add common effects
        # Common effects have at most ONE coord.
        with self.model:
            self.mu = 0.0
            coef_list = []

            # Iterate through terms and add their coefficients to coef_list
            for term in spec.common_terms.values():
                data = term.data
                label = term.name
                dist = term.prior.name
                args = term.prior.args
                if term.pymc_coords:
                    dims = list(term.pymc_coords.keys())
                    coef = self.build_common_distribution(dist, label, dims=dims, **args)
                else:
                    coef = self.build_common_distribution(dist, label, shape=data.shape[1], **args)
                coef_list.append(coef)

            # If there are predictors, use design matrix (w/o intercept)
            if coef_list:
                coefs = tt.concatenate(coef_list)
                X = spec._design.common.design_matrix  # pylint: disable=protected-access

            if self.has_intercept:
                term = spec.intercept_term
                distribution = self.get_distribution(term.prior.name)
                # If there are predictors, "Intercept" is a the intercept for centered predictors
                # This is intercept is re-scaled later.
                intercept = distribution("Intercept", shape=1, **term.prior.args)
                self.mu += intercept

                if spec.common_terms:
                    # Remove intercept from design matrix
                    # pylint: disable=protected-access
                    idx = spec._design.common.terms_info["Intercept"]["cols"]
                    X = np.delete(X, idx, axis=1)
                    self.mu += tt.dot(X - X.mean(0), coefs)
            elif coef_list:
                self.mu += tt.dot(X, coefs)

        ## Add group-specific effects
        # Group-specific effects always have pymc_coords. At least for the group.
        # The loop through predictor columns is not the most beautiful alternative.
        # But it's the fastest. Doing matrix multiplication, pm.math.dot(data, coef), is slower.

        # Add group specific terms that have prior for their correlation matrix
        with self.model:
            for group, eta in spec.priors_cor.items():
                # pylint: disable=protected-access
                terms = [spec.terms[name] for name in spec._get_group_specific_groups()[group]]
                self.mu += add_lkj(terms, eta)

        # Add group specific terms that don't have a prior for their correlation matrix
        terms = [
            term
            for term in spec.group_specific_terms.values()
            if term.name.split("|")[1] not in spec.priors_cor
        ]
        with self.model:
            for term in terms:
                label = term.name
                dist = term.prior.name
                args = term.prior.args
                predictor = term.predictor.squeeze()
                dims = list(term.pymc_coords.keys())
                coef = self.build_group_specific_distribution(
                    dist, label, noncentered, dims=dims, **args
                )
                coef = coef[term.group_index]

                if predictor.ndim > 1:
                    for col in range(predictor.shape[1]):
                        self.mu += coef[:, col] * predictor[:, col]
                else:
                    self.mu += coef * predictor

        # Build response distribution
        with self.model:
            self.build_response(spec)

        # Add potentials to the model
        if spec.potentials is not None:
            with self.model:
                count = 0
                for variable, constraint in spec.potentials:
                    if isinstance(variable, (list, tuple)):
                        lambda_args = [self.model[var] for var in variable]
                        potential = constraint(*lambda_args)
                    else:
                        potential = constraint(self.model[variable])
                    pm.Potential(f"pot_{count}", potential)
                    count += 1

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
            Initialization method. Defaults to ``'auto'``. The available methods are:
            * auto: Use ``'jitter+adapt_diag'`` and if this method fails it uses ``'adapt_diag'``.
            * adapt_diag: Start with a identity mass matrix and then adapt a diagonal based on the
              variance of the tuning samples. All chains use the test value (usually the prior mean)
              as starting point.
            * jitter+adapt_diag: Same as ``adapt_diag``, but use test value plus a uniform jitter in
              [-1, 1] as starting point in each chain.
            * advi+adapt_diag: Run ADVI and then adapt the resulting diagonal mass matrix based on
              the sample variance of the tuning samples.
            * advi+adapt_diag_grad: Run ADVI and then adapt the resulting diagonal mass matrix based
              on the variance of the gradients during tuning. This is **experimental** and might be
              removed in a future release.
            * advi: Run ADVI to estimate posterior mean and diagonal mass matrix.
            * advi_map: Initialize ADVI with MAP and use MAP as starting point.
            * map: Use the MAP as starting point. This is strongly discouraged.
            * adapt_full: Adapt a dense mass matrix using the sample covariances. All chains use the
              test value (usually the prior mean) as starting point.
            * jitter+adapt_full: Same as ``adapt_full``, but use test value plus a uniform jitter in
              [-1, 1] as starting point in each chain.
        n_init: int
            Number of initialization iterations. Only works for 'advi' init methods.
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
                try:
                    idata = pm.sample(
                        draws,
                        start=start,
                        init=init,
                        n_init=n_init,
                        return_inferencedata=True,
                        **kwargs,
                    )

                except (RuntimeError, ValueError):
                    if (
                        "ValueError: Mass matrix contains" in traceback.format_exc()
                        and init == "auto"
                    ):
                        _log.info(
                            "\nThe default initialization using init='auto' has failed, trying to "
                            "recover by switching to init='adapt_diag'",
                        )
                        idata = pm.sample(
                            draws,
                            start=start,
                            init="adapt_diag",
                            n_init=n_init,
                            return_inferencedata=True,
                            **kwargs,
                        )
                    else:
                        raise

            if omit_offsets:
                offset_vars = [var for var in idata.posterior.var() if var.endswith("_offset")]
                idata.posterior = idata.posterior.drop_vars(offset_vars)

            for group in idata.groups():
                getattr(idata, group).attrs["modeling_interface"] = "bambi"
                getattr(idata, group).attrs["modeling_interface_version"] = version.__version__

            # Drop variables and dimensions associated with LKJ prior
            vars_to_drop = [var for var in idata.posterior.var() if var.startswith("_LKJ")]
            dims_to_drop = [dim for dim in idata.posterior.dims if dim.startswith("_LKJ")]

            idata.posterior = idata.posterior.drop_vars(vars_to_drop)
            idata.posterior = idata.posterior.drop_dims(dims_to_drop)

            # Reorder coords
            # pylint: disable=protected-access
            coords_to_drop = [dim for dim in idata.posterior.dims if dim.endswith("_dim_0")]
            idata.posterior = idata.posterior.squeeze(coords_to_drop).reset_coords(
                coords_to_drop, drop=True
            )
            coords_original = list(self.spec._get_pymc_coords().keys())
            coords_group = [c for c in coords_original if c.endswith("_coord_group_factor")]
            for coord in coords_group:
                coords_original.remove(coord)
            coords_new = ["chain", "draw"] + coords_original + coords_group

            idata.posterior = idata.posterior.transpose(*coords_new)

            # Compute the actual intercept
            if self.has_intercept and self.spec.common_terms:
                chain_n = len(idata.posterior["chain"])
                draw_n = len(idata.posterior["draw"])

                # Design matrix without intercept
                X = self.spec._design.common.design_matrix
                idx = self.spec._design.common.terms_info["Intercept"]["cols"]
                X = np.delete(X, idx, axis=1)

                # Re-scale intercept for centered predictors
                posterior_ = idata.posterior.stack(sample=["chain", "draw"])
                coefs_list = [np.atleast_2d(posterior_[name]) for name in self.spec.common_terms]
                coefs = np.vstack(coefs_list)
                idata.posterior["Intercept"] -= np.dot(X.mean(0), coefs).reshape((chain_n, draw_n))

            # Sort variable names so Intercept is in the beginning
            var_names = list(idata.posterior.var())
            if "Intercept" in var_names:
                var_names.insert(0, var_names.pop(var_names.index("Intercept")))
                idata.posterior = idata.posterior[var_names]

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
        """Build and return a PyMC3 Distribution for a group specific term."""
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

    def build_response(self, spec):
        """Build and return a response distribution."""
        data = spec.response.data.squeeze()
        name = spec.response.name

        if spec.family.link.name in self.INVLINKS:
            linkinv = self.INVLINKS[spec.family.link.name]
        else:
            linkinv = spec.family.link.linkinv_backend

        likelihood = spec.family.likelihood
        dist = self.get_distribution(likelihood.name)
        kwargs = {likelihood.parent: linkinv(self.mu), "observed": data}
        if likelihood.priors:
            kwargs.update(
                {
                    k: self.expand_prior_args(k, v, name, False)
                    for (k, v) in likelihood.priors.items()
                }
            )
        if spec.family.name == "beta":
            # Beta distribution is specified using alpha and beta, but we have mu and kappa.
            # alpha = mu * kappa and beta = (1 - mu) * kappa
            alpha = kwargs["mu"] * kwargs["kappa"]
            beta = (1 - kwargs["mu"]) * kwargs["kappa"]
            return dist(name, alpha=alpha, beta=beta, observed=kwargs["observed"])

        if spec.family.name == "binomial":
            successes = data[:, 0].squeeze()
            trials = data[:, 1].squeeze()
            return dist(name, p=kwargs["p"], observed=successes, n=trials)

        if spec.family.name == "gamma":
            # Gamma distribution is specified using mu and sigma, but we request prior for alpha.
            # We need to build sigma from mu and alpha.
            # kwargs["mu"] ** 2 / kwargs["alpha"] would also work
            beta = kwargs["alpha"] / kwargs["mu"]
            sigma = (kwargs["mu"] / beta) ** 0.5
            return dist(name, mu=kwargs["mu"], sigma=sigma, observed=kwargs["observed"])

        return dist(name, **kwargs)

    def get_distribution(self, dist):
        """Return a PyMC3 distribution."""
        if isinstance(dist, str):
            if hasattr(pm, dist):
                dist = getattr(pm, dist)
            else:
                raise ValueError(f"The Distribution '{dist}' was not found in PyMC3")
        return dist

    def expand_prior_args(self, key, value, label, noncentered, **kwargs):
        # Inspect all args in case we have hyperparameters.
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


def add_lkj(terms, eta=1):
    """Add correlated prior for group-specific effects.

    This function receives a list of group-specific terms that share their `grouper`, constructs
    a multivariate Normal prior with LKJ prior on the correlation matrix, and adds the necessary
    variables to the model. It uses a non-centered parametrization.

    Parameters
    ----------
    terms: list
        A list of terms that share a common grouper (i.e. ``1|Group`` and ``Variable|Group`` in
        formula notation).
    eta: num
        The value for the eta parameter in the LKJ distribution.

    Parameters
    ----------
    mu:
        The contribution to the linear predictor of the roup-specific terms in ``terms``.
    """

    # Parameters
    # grouper: The name of the grouper.
    # rows: Sum of the number of columns in all the "Xi" matrices for a given grouper.
    #       Same than the order of L
    # cols: Number of groups in the grouper variable
    mu = 0
    grouper = terms[0].name.split("|")[1]
    rows = int(np.sum([term.predictor.shape[1] for term in terms]))
    cols = int(terms[0].grouper.shape[1])  # not the most beautiful, but works

    # Construct sigma
    # Horizontally stack the sigma values for all the hyperpriors
    sigma = np.hstack([term.prior.args["sigma"].args["sigma"] for term in terms])

    # Reconstruct the hyperprior for the standard deviations, using one variable
    sigma = pm.HalfNormal.dist(sigma=sigma, shape=rows)

    # Obtain Cholesky factor for the covariance
    lkj_decomp, corr, sigma = pm.LKJCholeskyCov(  # pylint: disable=unused-variable
        "_LKJCholeskyCov_" + grouper,
        n=rows,
        eta=eta,
        sd_dist=sigma,
        compute_corr=True,
        store_in_trace=False,
    )

    coefs_offset = pm.Normal("_LKJ_" + grouper + "_offset", mu=0, sigma=1, shape=(rows, cols))
    coefs = tt.dot(lkj_decomp, coefs_offset).T

    ## Separate group-specific terms
    start = 0
    for term in terms:
        label = term.name
        dims = list(term.pymc_coords.keys())
        predictor = term.predictor.squeeze()
        delta = term.predictor.shape[1]

        if delta == 1:
            idx = start
        else:
            idx = slice(start, start + delta)

        # Add prior for the parameter
        coef = pm.Deterministic(label, coefs[:, idx], dims=dims)
        coef = coef[term.group_index]

        # Add standard deviation of the hyperprior distribution
        group_dim = [dim for dim in dims if dim.endswith("_group_expr")]
        pm.Deterministic(label + "_sigma", sigma[idx], dims=group_dim)

        # Account for the contribution of the term to the linear predictor
        if predictor.ndim > 1:
            for col in range(predictor.shape[1]):
                mu += coef[:, col] * predictor[:, col]
        else:
            mu += coef * predictor
        start += delta

    # TO DO: Add correlations
    return mu
