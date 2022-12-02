import functools
import logging
import traceback

from copy import deepcopy

import numpy as np
import pymc as pm

import aesara.tensor as at

from bambi import version

from bambi.backend.links import cloglog, identity, inverse_squared, logit, probit, arctan_2
from bambi.backend.terms import ResponseTerm
from bambi.backend.model_components import ConstantComponent, DistributionalComponent
from bambi.families.multivariate import MultivariateFamily

_log = logging.getLogger("bambi")


class PyMCModel:
    """PyMC model-fitting backend."""

    INVLINKS = {
        "cloglog": cloglog,
        "identity": identity,
        "inverse_squared": inverse_squared,
        "inverse": at.reciprocal,
        "log": at.exp,
        "logit": logit,
        "probit": probit,
        "tan_2": arctan_2,
        "softmax": functools.partial(at.nnet.softmax, axis=-1),
    }

    def __init__(self):
        self.name = pm.__name__
        self.version = pm.__version__

        # Attributes defined elsewhere
        self.vi_approx = None
        self.fit = False
        self.model = None
        self.spec = None

    def build(self, spec):
        """Compile the PyMC model from an abstract model specification.

        Parameters
        ----------
        spec: bambi.Model
            A Bambi ``Model`` instance containing the abstract specification of the model
            to compile.
        """
        self.model = pm.Model()
        self.components = {}

        for name, values in spec.response_component.response_term.coords.items():
            if name not in self.model.coords:
                self.model.add_coords({name: values})

        with self.model:
            for name, component in spec.constant_components.items():
                self.components[name] = ConstantComponent(component)
                self.components[name].build(self, spec)

            for name, component in spec.distributional_components.items():
                self.components[name] = DistributionalComponent(component)
                self.components[name].build(self, spec)
            self.build_response(spec)
            self.build_potentials(spec)

        self.spec = spec

    def run(
        self,
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,
        include_mean=False,
        inference_method="mcmc",
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        **kwargs,
    ):
        """Run PyMC sampler."""
        inference_method = inference_method.lower()
        # NOTE: Methods return different types of objects (idata, approximation, and dictionary)
        if inference_method in ["mcmc", "nuts_numpyro", "nuts_blackjax"]:
            result = self._run_mcmc(
                draws,
                tune,
                discard_tuned_samples,
                omit_offsets,
                include_mean,
                init,
                n_init,
                chains,
                cores,
                random_seed,
                inference_method,
                **kwargs,
            )
        elif inference_method == "vi":
            result = self._run_vi(**kwargs)
        elif inference_method == "laplace":
            result = self._run_laplace(draws, omit_offsets, include_mean)
        else:
            raise NotImplementedError(f"'{inference_method}' method has not been implemented")

        self.fit = True
        return result

    def build_response(self, spec):
        """Add response term to the PyMC model

        Parameters
        ----------
        spec : bambi.Model
            The model.
        """
        response_term = ResponseTerm(spec.response_component.response_term, spec.family)
        response_term.build(self, spec)

    def build_potentials(self, spec):
        """Add potentials to the PyMC model.

        Potentials are arbitrary quantities that are added to the model log likelihood.
        See 'Factor Potentials' in
        https://github.com/fonnesbeck/probabilistic_python/blob/main/pymc_intro.ipynb

        Parameters
        ----------
        spec : bambi.Model
            The model.
        """
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
        include_mean=False,
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        sampler_backend="mcmc",
        **kwargs,
    ):
        with self.model:
            if sampler_backend == "mcmc":
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
                            **kwargs,
                        )
                    else:
                        raise
            elif sampler_backend == "nuts_numpyro":
                import pymc.sampling_jax  # pylint: disable=import-outside-toplevel

                if not chains:
                    # sample_numpyro_nuts does not handle chains = None like pm.sample does
                    chains = 4
                idata = pymc.sampling_jax.sample_numpyro_nuts(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    **kwargs,
                )
            elif sampler_backend == "nuts_blackjax":
                import pymc.sampling_jax  # pylint: disable=import-outside-toplevel

                # sample_blackjax_nuts does not handle chains = None like pm.sample does
                if not chains:
                    chains = 4
                idata = pymc.sampling_jax.sample_blackjax_nuts(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"sampler_backend value {sampler_backend} is not valid. Please choose one of"
                    f"'mcmc', 'nuts_numpyro' or 'nuts_blackjax'"
                )
        idata = self._clean_results(idata, omit_offsets, include_mean)
        return idata

    def _clean_results(self, idata, omit_offsets, include_mean):
        for group in idata.groups():
            getattr(idata, group).attrs["modeling_interface"] = "bambi"
            getattr(idata, group).attrs["modeling_interface_version"] = version.__version__

        # NOTE: We were using .var() which calculated the variance!
        if omit_offsets:
            offset_vars = [var for var in idata.posterior.data_vars if var.endswith("_offset")]
            idata.posterior = idata.posterior.drop_vars(offset_vars)

        # Drop variables and dimensions associated with LKJ prior
        vars_to_drop = [var for var in idata.posterior.data_vars if var.startswith("_LKJ")]
        dims_to_drop = [dim for dim in idata.posterior.dims if dim.startswith("_LKJ")]

        idata.posterior = idata.posterior.drop_vars(vars_to_drop)
        idata.posterior = idata.posterior.drop_dims(dims_to_drop)

        # This does not add any new coordinate, it just changes the order so the ones
        # ending in "__factor_dim" are placed after the others.
        dims_original = list(self.model.coords)
        dims_group = [c for c in dims_original if c.endswith("__factor_dim")]

        # Keep the original order in dims_original
        dims_original_set = set(dims_original) - set(dims_group)
        dims_original = [c for c in dims_original if c in dims_original_set]
        dims_new = ["chain", "draw"] + dims_original + dims_group
        idata.posterior = idata.posterior.transpose(*dims_new)

        # Compute the actual intercept in all distributional components that have an intercept

        for pymc_component in self.distributional_components.values():
            bambi_component = pymc_component.component
            if bambi_component.intercept_term and bambi_component.common_terms:
                chain_n = len(idata.posterior["chain"])
                draw_n = len(idata.posterior["draw"])
                shape, dims = (chain_n, draw_n), ("chain", "draw")
                X = pymc_component.design_matrix_without_intercept

                common_terms = []
                for term in bambi_component.common_terms.values():
                    common_terms.append(get_backend_name(term))

                response_coords = self.spec.response_component.response_term.coords
                if response_coords:
                    # Grab the first object in a dictionary
                    levels = list(response_coords.values())[0]
                    shape += (len(levels),)
                    dims += tuple(response_coords)

                posterior = idata.posterior.stack(samples=dims)
                coefs = np.vstack([np.atleast_2d(posterior[name].values) for name in common_terms])
                name = get_backend_name(bambi_component.intercept_term)
                center_factor = np.dot(X.mean(0), coefs).reshape(shape)
                idata.posterior[name] = idata.posterior[name] - center_factor

        if include_mean:
            self.spec.predict(idata)

        return idata

    def _run_vi(self, **kwargs):
        with self.model:
            self.vi_approx = pm.fit(**kwargs)
        return self.vi_approx

    def _run_laplace(self, draws, omit_offsets, include_mean):
        """Fit a model using a Laplace approximation.

        Mainly for pedagogical use, provides reasonable results for approximately
        Gaussian posteriors. The approximation can be very poor for some models
        like hierarchical ones. Use ``mcmc``, ``nuts_numpyro``, ``nuts_blackjax``
        or ``vi`` for better approximations.

        Parameters
        ----------
        draws: int
            The number of samples to draw from the posterior distribution.
        omit_offsets: bool
            Omits offset terms in the ``InferenceData`` object returned when the model includes
            group specific effects.
        include_mean: bool
            Compute the posterior of the mean response.

        Returns
        -------
        An ArviZ's InferenceData object.
        """
        with self.model:
            maps = pm.find_MAP()
            n_maps = deepcopy(maps)
            for m in maps:
                if pm.util.is_transformed_name(m):
                    n_maps.pop(pm.util.get_untransformed_name(m))

            hessian = pm.find_hessian(n_maps)

        if np.linalg.det(hessian) == 0:
            raise np.linalg.LinAlgError("Singular matrix. Use mcmc or vi method")

        cov = np.linalg.inv(hessian)
        modes = np.concatenate([np.atleast_1d(v) for v in n_maps.values()])

        samples = np.random.multivariate_normal(modes, cov, size=draws)

        idata = _posterior_samples_to_idata(samples, self.model)
        idata = self._clean_results(idata, omit_offsets, include_mean)
        return idata

    @property
    def constant_components(self):
        return {k: v for k, v in self.components.items() if isinstance(v, ConstantComponent)}

    @property
    def distributional_components(self):
        return {k: v for k, v in self.components.items() if isinstance(v, DistributionalComponent)}


def _posterior_samples_to_idata(samples, model):
    """Create InferenceData from samples.

    Parameters
    ----------
    samples: array
        Posterior samples
    model: PyMC model

    Returns
    -------
    An ArviZ's InferenceData object.
    """
    initial_point = model.initial_point()
    variables = model.value_vars

    var_info = {}
    for name, value in initial_point.items():
        var_info[name] = (value.shape, value.size)

    length_pos = len(samples)
    varnames = [v.name for v in variables]

    with model:
        strace = pm.backends.ndarray.NDArray(name=model.name)  # pylint:disable=no-member
        strace.setup(length_pos, 0)
    for i in range(length_pos):
        value = []
        size = 0
        for varname in varnames:
            shape, new_size = var_info[varname]
            var_samples = samples[i][size : size + new_size]
            value.append(var_samples.reshape(shape))
            size += new_size
        strace.record(point=dict(zip(varnames, value)))

    idata = pm.to_inference_data(pm.backends.base.MultiTrace([strace]), model=model)
    return idata


def add_lkj(backend, terms, eta=1):
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
    mu
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
    # pylint: disable=unused-variable, disable=unpacking-non-sequence
    (lkj_decomp, corr, sigma,) = pm.LKJCholeskyCov(
        "_LKJCholeskyCov_" + grouper,
        n=rows,
        eta=eta,
        sd_dist=sigma,
        compute_corr=True,
        store_in_trace=False,
    )

    coefs_offset = pm.Normal("_LKJ_" + grouper + "_offset", mu=0, sigma=1, shape=(rows, cols))
    coefs = at.dot(lkj_decomp, coefs_offset).T

    ## Separate group-specific terms
    start = 0
    for term in terms:
        label = term.name
        dims = list(term.coords)

        # Add coordinates to the model, only if they are not added yet.
        for name, values in term.coords.items():
            if name not in backend.model.coords:
                backend.model.add_coords({name: values})
        backend.coords.update(**term.coords)

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


def get_backend_name(term):
    if term.alias:
        return term.alias
    return term.name
