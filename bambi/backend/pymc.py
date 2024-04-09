import functools
import importlib
import logging
import operator
import traceback
import warnings

from copy import deepcopy
from importlib.metadata import version

import numpy as np
import pymc as pm

import pytensor.tensor as pt
from pytensor.tensor.special import softmax

from bambi.backend.links import cloglog, identity, inverse_squared, logit, probit, arctan_2
from bambi.backend.model_components import ConstantComponent, DistributionalComponent
from bambi.utils import get_aliased_name

_log = logging.getLogger("bambi")


__version__ = version("bambi")


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
        "tan_2": arctan_2,
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
        self.bayeux_methods = _get_bayeux_methods()
        self.pymc_methods = {"mcmc": ["mcmc"], "vi": ["vi"]}

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

        if inference_method == "nuts_numpyro":
            inference_method = "numpyro_nuts"
            warnings.warn(
                "'nuts_numpyro' has been replaced by 'numpyro_nuts' and will be "
                "removed in a future release",
                category=FutureWarning,
            )
        elif inference_method == "nuts_blackjax":
            inference_method = "blackjax_nuts"
            warnings.warn(
                "'nuts_blackjax' has been replaced by 'blackjax_nuts' and will "
                "be removed in a future release",
                category=FutureWarning,
            )

        # NOTE: Methods return different types of objects (idata, approximation, and dictionary)
        if inference_method in (self.pymc_methods["mcmc"] + self.bayeux_methods["mcmc"]):
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
        elif inference_method in self.pymc_methods["vi"]:
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
        spec : bambi.Modelf
            The model.
        """
        response_component = self.components[spec.response_name]
        response_component.build_response(self, spec)

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
        if sampler_backend in self.pymc_methods["mcmc"]:
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
                idata_from = "pymc"
        elif sampler_backend in self.bayeux_methods["mcmc"]:
            import bayeux as bx  # pylint: disable=import-outside-toplevel
            import jax  # pylint: disable=import-outside-toplevel

            # Set the seed for reproducibility if provided
            if random_seed is not None:
                if not isinstance(random_seed, int):
                    random_seed = random_seed[0]
                np.random.seed(random_seed)

            jax_seed = jax.random.PRNGKey(np.random.randint(2**32 - 1))

            bx_model = bx.Model.from_pymc(self.model)
            bx_sampler = operator.attrgetter(sampler_backend)(
                bx_model.mcmc  # pylint: disable=no-member
            )
            idata = bx_sampler(seed=jax_seed, **kwargs)
            idata_from = "bayeux"
        else:
            raise ValueError(
                f"sampler_backend value {sampler_backend} is not valid. Please choose one of"
                f" {self.pymc_methods['mcmc'] + self.bayeux_methods['mcmc']}"
            )

        idata = self._clean_results(idata, omit_offsets, include_mean, idata_from)
        return idata

    def _clean_results(self, idata, omit_offsets, include_mean, idata_from):
        # Before doing anything, make sure we compute deterministics.
        if idata_from == "bayeux":
            idata.posterior = pm.compute_deterministics(
                idata.posterior, model=self.model, merge_dataset=True, progressbar=False
            )

        for group in idata.groups():
            getattr(idata, group).attrs["modeling_interface"] = "bambi"
            getattr(idata, group).attrs["modeling_interface_version"] = __version__

        if omit_offsets:
            offset_vars = [var for var in idata.posterior.data_vars if var.endswith("_offset")]
            idata.posterior = idata.posterior.drop_vars(offset_vars)

        # NOTE:
        # This has not had an effect for a while since we haven't been supporting LKJ prior lately.

        # Drop variables and dimensions associated with LKJ prior
        # vars_to_drop = [var for var in idata.posterior.data_vars if var.startswith("_LKJ")]
        # dims_to_drop = [dim for dim in idata.posterior.dims if dim.startswith("_LKJ")]
        # idata.posterior = idata.posterior.drop_vars(vars_to_drop)
        # idata.posterior = idata.posterior.drop_dims(dims_to_drop)

        dims_original = list(self.model.coords)

        # Identify bayeux idata and rename dims and coordinates to match PyMC model
        if idata_from == "bayeux":
            cleaned_dims = {
                f"{dim}_0": dim
                for dim in dims_original
                if not dim.endswith("_obs") and f"{dim}_0" in idata.posterior.dims
            }
            idata = idata.rename(cleaned_dims)

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

                response_coords = self.spec.response_component.response_term.coords
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
        like hierarchical ones. Use ``mcmc``, ``vi``, or JAX based MCMC methods
        for better approximations.

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
        idata = self._clean_results(idata, omit_offsets, include_mean, idata_from="pymc")
        return idata

    @property
    def response_component(self):
        return self.components[self.spec.response_name]

    @property
    def constant_components(self):
        return {k: v for k, v in self.components.items() if isinstance(v, ConstantComponent)}

    @property
    def distributional_components(self):
        return {k: v for k, v in self.components.items() if isinstance(v, DistributionalComponent)}

    @property
    def inference_methods(self):
        return {"pymc": self.pymc_methods, "bayeux": self.bayeux_methods}


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


def _get_bayeux_methods():
    """Gets a dictionary of usable bayeux methods if the bayeux package is installed
    within the user's environment.

    Returns
    -------
    dict
        A dict where the keys are the module names and the values are the methods
        available in that module.
    """
    if importlib.util.find_spec("bayeux") is None:
        return {"mcmc": []}

    import bayeux as bx  # pylint: disable=import-outside-toplevel

    # Dummy log density to get access to all methods
    return bx.Model(lambda x: -(x**2), 0.0).methods
