import logging
import traceback

import numpy as np
import theano.tensor as tt
import pymc3 as pm

from bambi import version
from bambi.priors import Prior

from bambi.backends.utils import probit, cloglog, has_hyperprior, get_pymc_distribution

_log = logging.getLogger("bambi")


class PyMC3BackEnd:
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
        self.has_intercept = spec.intercept_term is not None
        self.mu = 0.0

        ## Add common effects
        with self.model:
            self.mu += Common(spec.intercept_term, spec.common_terms).build()

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
            self.mu += GroupSpecific(terms, spec.noncentered).build()

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

    def run(
        self,
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,
        method="mcmc",
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        **kwargs,
    ):
        """Run PyMC3 sampler."""
        model = self.model

        if method.lower() == "mcmc":
            with model:
                try:
                    idata = pm.sample(
                        draws=draws,
                        tune=tune,
                        discard_tuned_samples=discard_tuned_samples,
                        init=init,
                        n_init=n_init,
                        chains=chains,
                        cores=cores,
                        random_seed=random_seed,
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
                            draws=draws,
                            tune=tune,
                            discard_tuned_samples=discard_tuned_samples,
                            init="adapt_diag",
                            n_init=n_init,
                            chains=chains,
                            cores=cores,
                            random_seed=random_seed,
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
                self.advi_params = pm.variational.ADVI(**kwargs)
            # this should return an InferenceData object (once arviz adds support for VI)
            return self.advi_params

        else:
            return _laplace(model)

    def build_response(self, spec):
        """Build and return a response distribution."""
        data = spec.response.data.squeeze()
        name = spec.response.name

        if spec.family.link.name in self.INVLINKS:
            linkinv = self.INVLINKS[spec.family.link.name]
        else:
            linkinv = spec.family.link.linkinv_backend

        likelihood = spec.family.likelihood
        dist = get_pymc_distribution(likelihood.name)
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


class Common:
    def __init__(self, intercept, terms):
        self.intercept = intercept
        self.terms = terms

    def build(self):
        """
        intercept: Intercept term or None.
        terms: A dictionary with common terms from the Bambi model.
        model: A PyMC3 model.
        """
        mu = 0.0
        if self.intercept and self.terms:
            # Add intercept for centered predictors.
            # This is intercept is re-scaled later.
            mu += self.build_intercept()

            # Add centered predictors
            coefs, data = self.build_terms()
            data = data - data.mean(0)
            mu += tt.dot(data, coefs)
        elif self.intercept:
            # Add intercept
            mu += self.build_intercept()
        elif self.terms:
            # Add non-centered predictors
            coefs, data = self.build_terms()
            mu += tt.dot(data, coefs)
        return mu

    def build_term(self, term):
        """Build and return a PyMC3 Distribution for a common term."""
        data = term.data
        label = term.name
        dist = term.prior.name
        args = term.prior.args
        distribution = get_pymc_distribution(dist)
        if term.pymc_coords:
            dims = list(term.pymc_coords.keys())
            coef = distribution(label, dims=dims, **args)
        else:
            coef = distribution(label, shape=data.shape[1], **args)
        return coef

    def build_terms(self):
        """Build a dictionary of common terms"""
        coefs = []
        columns = []
        for term in self.terms.values():
            columns.append(term.data)
            coefs.append(self.build_term(term))

        # Column vector of coefficients
        coefs = tt.concatenate(coefs)

        # Design matrix
        data = np.hstack(columns)
        return coefs, data

    def build_intercept(self):
        distribution = get_pymc_distribution(self.intercept.prior.name)
        return distribution("Intercept", shape=1, **self.intercept.prior.args)


class GroupSpecific:
    def __init__(self, terms, noncentered):
        self.terms = terms
        self.noncentered = noncentered

    def build(self):
        mu = 0
        for term in self.terms:
            coef, predictor = self.build_term(term)
            if predictor.ndim > 1:
                for col in range(predictor.shape[1]):
                    mu += coef[:, col] * predictor[:, col]
            else:
                mu += coef * predictor
        return mu


    def build_term(self, term):
        label = term.name
        dist = term.prior.name
        args = term.prior.args
        predictor = term.predictor.squeeze()
        dims = list(term.pymc_coords.keys())
        coef = self.build_distribution(dist, label, dims=dims, **args)
        coef = coef[term.group_index]

        return coef, predictor

    def build_distribution(self, dist, label, **kwargs):
        """Build and return a PyMC3 Distribution for a group specific term."""

        dist = get_pymc_distribution(dist)

        if "dims" in kwargs:
            group_dim = [dim for dim in kwargs["dims"] if dim.endswith("_group_expr")]
            kwargs = {
                k: self.expand_prior_args(k, v, label, dims=group_dim) for (k, v) in kwargs.items()
            }
        else:
            kwargs = {k: self.expand_prior_args(k, v, label) for (k, v) in kwargs.items()}

        # Non-centered parameterization for hyperpriors
        if self.noncentered and has_hyperprior(kwargs):
            sigma = kwargs["sigma"]
            offset = pm.Normal(label + "_offset", mu=0, sigma=1, dims=kwargs["dims"])
            return pm.Deterministic(label, offset * sigma, dims=kwargs["dims"])
        return dist(label, **kwargs)

    def expand_prior_args(self, key, value, label, **kwargs):
        # Inspect all args in case we have hyperparameters.
        # kwargs are used to pass 'dims' for group specific terms.
        if isinstance(value, Prior):
            return self.build_distribution(value.name, f"{label}_{key}", **value.args, **kwargs)
        return value


class Response:
    def __init__(self):
        return None


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
    # grouper: The name of the grouper.build_group_specific_distribution
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
