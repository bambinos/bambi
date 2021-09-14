import logging
import traceback

import numpy as np
import theano.tensor as tt
import pymc3 as pm

from bambi import version

from bambi.backend.links import cloglog, identity, inverse_squared, probit
from bambi.backend.terms import CommonTerm, GroupSpecificTerm, InterceptTerm, ResponseTerm

_log = logging.getLogger("bambi")


class PyMC3Model:
    """PyMC3 model-fitting backend."""

    INVLINKS = {
        "cloglog": cloglog,
        "identity": identity,
        "inverse_squared": inverse_squared,
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
        self.has_intercept = False  # build()
        self.mu = None  # build()
        self.spec = None  # build()
        self.trace = None  # build()
        self.advi_params = None  # build()
        self.fit = False  # run()

    def build(self, spec):
        """Compile the PyMC3 model from an abstract model specification.

        Parameters
        ----------
        spec : Bambi model
            A Bambi ``Model`` instance containing the abstract specification of the model
            to compile.
        """
        self.model = pm.Model()
        self.has_intercept = spec.intercept_term is not None
        self.mu = 0.0

        with self.model:
            self._build_intercept(spec)
            self._build_common_terms(spec)
            self._build_group_specific_terms(spec)
            self._build_response(spec)
            self._build_potentials(spec)
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
            idata = self._run_mcmc(
                draws,
                tune,
                discard_tuned_samples,
                omit_offsets,
                init,
                n_init,
                chains,
                cores,
                random_seed,
                **kwargs,
            )
            self.fit = True
            return idata
        elif method.lower() == "advi":
            # This should return an InferenceData object (once arviz adds support for VI)
            with model:
                self.advi_params = pm.variational.ADVI(**kwargs)
            return self.advi_params
        else:
            return _laplace(model)

    def _build_intercept(self, spec):
        if self.has_intercept:
            self.mu += InterceptTerm(spec.intercept_term).build()

    def _build_common_terms(self, spec):
        if spec.common_terms:
            coords = {}
            coefs = []
            columns = []
            for term in spec.common_terms.values():
                common_term = CommonTerm(term)
                coef, data = common_term.build()
                coefs.append(coef)
                columns.append(data)
                coords.update(**common_term.coords)

            # Column vector of coefficients and Design matrix
            coefs = tt.concatenate(coefs)
            data = np.hstack(columns)

            # If there's an intercept, center the data
            if self.has_intercept:
                data = data - data.mean(0)

            # Add coords
            self.model.add_coords(coords)

            # Add term to linear predictor
            self.mu += tt.dot(data, coefs)

    def _build_group_specific_terms(self, spec):
        # Add group specific terms that have prior for their correlation matrix
        for group, eta in spec.priors_cor.items():
            # pylint: disable=protected-access
            terms = [spec.terms[name] for name in spec._get_group_specific_groups()[group]]
            self.mu += add_lkj(terms, eta)

        terms = [
            term
            for term in spec.group_specific_terms.values()
            if term.name.split("|")[1] not in spec.priors_cor
        ]
        for term in terms:
            group_specific_term = GroupSpecificTerm(term, spec.noncentered)
            coef, predictor = group_specific_term.build()

            # Add coords
            self.model.add_coords(group_specific_term.coords)

            # Add to the linear predictor
            # The loop through predictor columns is not the most beautiful alternative.
            # But it's the fastest. Doing matrix multiplication, pm.math.dot(data, coef), is slower.
            if predictor.ndim > 1:
                for col in range(predictor.shape[1]):
                    self.mu += coef[:, col] * predictor[:, col]
            else:
                self.mu += coef * predictor

    def _build_response(self, spec):
        ResponseTerm(spec.response, spec.family).build(self.mu, self.INVLINKS)

    def _build_potentials(self, spec):
        if spec.potentials is not None:
            count = 0
            for variable, constraint in spec.potentials:
                if isinstance(variable, (list, tuple)):
                    lambda_args = [self.model[var] for var in variable]
                    potential = constraint(*lambda_args)
                else:
                    potential = constraint(self.model[variable])
                pm.Potential(f"pot_{count}", potential)
                count += 1

    def _run_mcmc(
        self,
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        **kwargs,
    ):
        with self.model:
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
                if "ValueError: Mass matrix contains" in traceback.format_exc() and init == "auto":
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

        idata = self._clean_mcmc_results(idata, omit_offsets)
        return idata

    def _clean_mcmc_results(self, idata, omit_offsets):
        for group in idata.groups():
            getattr(idata, group).attrs["modeling_interface"] = "bambi"
            getattr(idata, group).attrs["modeling_interface_version"] = version.__version__

        if omit_offsets:
            offset_vars = [var for var in idata.posterior.var() if var.endswith("_offset")]
            idata.posterior = idata.posterior.drop_vars(offset_vars)

        # Drop variables and dimensions associated with LKJ prior
        vars_to_drop = [var for var in idata.posterior.var() if var.startswith("_LKJ")]
        dims_to_drop = [dim for dim in idata.posterior.dims if dim.startswith("_LKJ")]

        idata.posterior = idata.posterior.drop_vars(vars_to_drop)
        idata.posterior = idata.posterior.drop_dims(dims_to_drop)

        # Reorder coords
        # pylint: disable=protected-access
        # WHY DO WE DO THIS? I DON'T REMEMBER
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
            posterior = idata.posterior.stack(sample=["chain", "draw"])
            coefs = np.vstack([np.atleast_2d(posterior[name]) for name in self.spec.common_terms])
            idata.posterior["Intercept"] -= np.dot(X.mean(0), coefs).reshape((chain_n, draw_n))

        return idata


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
