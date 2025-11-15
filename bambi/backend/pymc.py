import functools
import logging
import traceback
import warnings
from copy import deepcopy
from importlib.metadata import version

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.util import get_default_varnames
from pytensor.tensor.special import softmax

from bambi.backend.links import (
    cloglog,
    identity,
    inverse_squared,
    logit,
    probit,
)
from bambi.backend.model_components import (
    ConstantComponent,
    DistributionalComponent,
    ResponseComponent,
)
from bambi.utils import get_aliased_name

_log = logging.getLogger("bambi")


__version__ = version("bambi")


_SUPPORTED_METHODS = {"pymc", "numpyro", "blackjax", "nutpie", "vi", "laplace"}
_DEPRECATION_MAP = {
    "mcmc": "pymc",
    "nuts_numpyro": "numpyro",
    "numpyro_nuts": "numpyro",
    "nuts_blackjax": "blackjax",
    "blackjax_nuts": "blackjax",
}


class PyMCModel:
    """PyMC model-fitting backend."""

    INVLINKS = {
        "cloglog": cloglog,
        "identity": identity,
        "inverse_squared": inverse_squared,
        "inverse": pt.reciprocal,
        "log": pt.exp,
        "logit": logit,
        "probit": probit,
        "softmax": functools.partial(softmax, axis=-1),
    }

    def __init__(self):
        self.name = pm.__name__
        self.version = pm.__version__
        self.vi_approx = None
        self.fit = False
        self.model = None
        self.spec = None
        self.components = {}
        self.response_component = None

    def build(self, spec):
        """Compile the PyMC model from an abstract model specification

        Parameters
        ----------
        spec : bambi.Model
            A Bambi `Model` instance containing the abstract specification of the model to compile.
        """
        self.model = pm.Model()
        self.components = {}

        for name, values in spec.response_component.term.coords.items():
            if name not in self.model.coords:
                self.model.add_coords({name: values})

        with self.model:
            # Add constant components
            for name, component in spec.constant_components.items():
                self.components[name] = ConstantComponent(component)
                self.components[name].build(self, spec)

            # Add distributional components
            for name, component in spec.distributional_components.items():
                self.components[name] = DistributionalComponent(component)
                self.components[name].build(self, spec)

            # Add response
            self.response_component = ResponseComponent(spec.response_component)
            self.response_component.build(self, spec)

            # Add potentials
            self.build_potentials(spec)

        self.spec = spec

    def run(
        self,
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,
        include_response_params=False,
        inference_method="pymc",
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        **kwargs,
    ):
        """Run PyMC sampler."""
        inference_method = inference_method.lower()

        # Handle deprecated inference methods
        if inference_method in _DEPRECATION_MAP:
            new_method = _DEPRECATION_MAP[inference_method]
            warnings.warn(
                f"'{inference_method}' has been replaced by '{new_method}' and will be "
                "removed in a future release.",
                category=FutureWarning,
            )
            inference_method = new_method

        # Validate the inference method
        if inference_method not in _SUPPORTED_METHODS:
            # Use sorted() for a predictable, user-friendly error message
            supported = ", ".join(sorted(_SUPPORTED_METHODS))
            raise ValueError(
                f"'{inference_method}' is not a supported inference method. "
                f"Must be one of: {supported}"
            )

        # Ensure the appropriate dependencies are installed for the selected inference method
        self._check_dependencies(inference_method)

        # NOTE: Methods return different types of objects (idata, approximation, and dictionary)
        if inference_method == "vi":
            result = self._run_vi(random_seed=random_seed, **kwargs)
        elif inference_method == "laplace":
            result = self._run_laplace(
                draws=draws,
                omit_offsets=omit_offsets,
                include_response_params=include_response_params,
            )
        else:
            result = self._run_mcmc(
                draws=draws,
                tune=tune,
                discard_tuned_samples=discard_tuned_samples,
                omit_offsets=omit_offsets,
                include_response_params=include_response_params,
                init=init,
                n_init=n_init,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                sampler_backend=inference_method,
                **kwargs,
            )

        self.fit = True
        return result

    def build_potentials(self, spec):
        """Add potentials to the PyMC model

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
        draws,
        tune,
        discard_tuned_samples,
        omit_offsets,
        include_response_params,
        init,
        n_init,
        chains,
        cores,
        random_seed,
        sampler_backend,
        **kwargs,
    ):
        # Don't include the parameters of the likelihood, which are deterministics.
        # They can take lot of space in the trace and increase RAM requirements.
        vars_to_sample = get_default_varnames(
            self.model.unobserved_value_vars, include_transformed=False
        )
        vars_to_sample = [variable.name for variable in vars_to_sample]

        for name, variable in self.model.named_vars.items():
            is_likelihood_param = name in self.spec.family.likelihood.params
            is_deterministic = variable in self.model.deterministics
            if is_likelihood_param and is_deterministic:
                vars_to_sample.remove(name)

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
                    var_names=vars_to_sample,
                    nuts_sampler=sampler_backend,
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
                        var_names=vars_to_sample,
                        nuts_sampler=sampler_backend,
                        **kwargs,
                    )
                else:
                    raise

        idata = self._clean_results(idata, omit_offsets, include_response_params)
        return idata

    def _clean_results(self, idata, omit_offsets, include_response_params):
        # Before doing anything, make sure we compute deterministics.
        # But, don't include those determinisics for parameters of the likelihood.

        for group in idata.groups():
            getattr(idata, group).attrs["modeling_interface"] = "bambi"
            getattr(idata, group).attrs["modeling_interface_version"] = __version__

        if omit_offsets:
            offset_vars = [var for var in idata.posterior.data_vars if var.endswith("_offset")]
            idata.posterior = idata.posterior.drop_vars(offset_vars)

        dims_original = list(self.model.coords)

        # Don't select dims that are in the model but unused in the posterior
        dims_original = [dim for dim in dims_original if dim in idata.posterior.dims]

        # This does not add any new coordinate, it just changes the order so the ones
        # ending in "__factor_dim" are placed after the others.
        dims_group = [dim for dim in dims_original if dim.endswith("__factor_dim")]

        # Keep the original order in dims_original
        dims_original_set = set(dims_original) - set(dims_group)
        dims_original = [dim for dim in dims_original if dim in dims_original_set]
        dims_new = ["chain", "draw"] + dims_original + dims_group

        # Drop unused dimensions before transposing
        dims_to_drop = [dim for dim in idata.posterior.dims if dim not in dims_new]
        idata.posterior = idata.posterior.drop_dims(dims_to_drop).transpose(*dims_new)

        # Compute the actual intercept in all distributional components that have an intercept
        for pymc_component in self.distributional_components.values():
            bambi_component = pymc_component.component
            if (
                bambi_component.intercept_term
                and bambi_component.common_terms
                and self.spec.center_predictors
            ):
                chain_n = len(idata.posterior["chain"])
                draw_n = len(idata.posterior["draw"])
                shape, dims = (chain_n, draw_n), ("chain", "draw")
                X = pymc_component.design_matrix_without_intercept

                common_terms = []
                for term in bambi_component.common_terms.values():
                    common_terms.append(get_aliased_name(term))

                response_coords = self.spec.response_component.term.coords
                if response_coords:
                    # Grab the first object in a dictionary
                    levels = list(response_coords.values())[0]
                    shape += (len(levels),)
                    dims += tuple(response_coords)

                posterior = idata.posterior.stack(samples=dims)
                coefs = np.vstack([np.atleast_2d(posterior[name].values) for name in common_terms])
                name = get_aliased_name(bambi_component.intercept_term)
                center_factor = np.dot(X.mean(0), coefs).reshape(shape)
                idata.posterior[name] = idata.posterior[name] - center_factor

        if include_response_params:
            self.spec.predict(idata)

        return idata

    def _run_vi(self, random_seed, **kwargs):
        with self.model:
            self.vi_approx = pm.fit(random_seed=random_seed, **kwargs)
        return self.vi_approx

    def _run_laplace(self, draws, omit_offsets, include_response_params):
        """Fit a model using a Laplace approximation.

        Mainly for pedagogical use, provides reasonable results for approximately Gaussian
        posteriors. The approximation can be very poor for some models  like hierarchical ones.

        Parameters
        ----------
        draws : int
            The number of samples to draw from the posterior distribution.
        omit_offsets : bool
            Omits offset terms in the `InferenceData` object returned when the model includes
            group specific effects.
        include_response_params : bool
            Compute the posterior of the mean response.

        Returns
        -------
        An ArviZ's InferenceData object.
        """
        with self.model:
            maps = pm.find_MAP()
            n_maps = deepcopy(maps)

            # Remove deterministics for parent parameters
            n_maps = {
                key: value
                for key, value in n_maps.items()
                if key not in self.spec.family.likelihood.params
            }

            for m in maps:
                if pm.util.is_transformed_name(m):
                    untransformed_name = pm.util.get_untransformed_name(m)
                    if untransformed_name in n_maps:
                        n_maps.pop(untransformed_name)

            hessian = pm.find_hessian(n_maps)

        if np.linalg.det(hessian) == 0:
            raise np.linalg.LinAlgError("Singular matrix. Use mcmc or vi method")

        cov = np.linalg.inv(hessian)
        modes = np.concatenate([np.atleast_1d(v) for v in n_maps.values()])

        samples = np.random.multivariate_normal(modes, cov, size=draws)

        idata = _posterior_samples_to_idata(samples, self.model)
        idata = self._clean_results(idata, omit_offsets, include_response_params)
        return idata

    def _check_dependencies(self, inference_method):
        """Dependency checking given the selected inference method."""
        required_packages = {
            "numpyro": ["numpyro", "jax"],
            "blackjax": ["blackjax", "jax"],
            "nutpie": ["nutpie"],
        }

        if inference_method in required_packages:
            missing = []
            for package in required_packages[inference_method]:
                try:
                    __import__(package)
                except ImportError:
                    missing.append(package)

            if missing:
                raise ImportError(
                    f"'{inference_method}' requires package(s): {', '.join(missing)}. "
                )

    @property
    def constant_components(self):
        return {k: v for k, v in self.components.items() if isinstance(v, ConstantComponent)}

    @property
    def distributional_components(self):
        return {k: v for k, v in self.components.items() if isinstance(v, DistributionalComponent)}


def _posterior_samples_to_idata(samples, model):
    """Create InferenceData from samples

    Parameters
    ----------
    samples : array
        Posterior samples
    model : PyMC model

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
