import logging
import traceback
from copy import deepcopy
from importlib.metadata import version

from typing import Optional

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc.backends.arviz import apply_function_over_dataset, coords_and_dims_for_inferencedata
from pymc.model.fgraph import fgraph_from_model, model_from_fgraph
from pymc.model.transform.conditioning import remove_value_transforms
from xarray import DataTree

from bambi.backend.pymc.coords import coords_from_response
from bambi.backend.pymc.parameters import (
    build_conditional_parameter,
    build_marginal_parameter,
    remove_group_specific_contributions,
)
from bambi.backend.pymc.parameters.conditional import (
    ConditionalParameterInfo,
    DenseGroupSpecificFactorPlan,
    GroupSpecificGraphState,
    add_new_dense_group_specific_contributions,
    add_new_sparse_group_specific_contributions,
    build_new_dense_conditional_parameter_data,
    build_new_sparse_conditional_parameter_data,
    make_conditional_parameter_info,
)
from bambi.backend.pymc.terms import build_potentials, build_response_term
from bambi.backend.pymc.terms.response import (
    build_new_response_data,
    build_response_interventions,
    replace_response_variables,
)
from bambi.config import config as bmb_config
from bambi.utils import as_dataset

_logger = logging.getLogger("bambi")


__version__ = version("bambi")


_SUPPORTED_METHODS = {"pymc", "numpyro", "blackjax", "nutpie", "vi", "laplace"}


class PyMCModel:
    def __init__(self, model):
        """_summary_

        Parameters
        ----------
        model : bmb.Model
            The bambi model specification.
        """
        self.model = None
        self.spec = model
        self.fit = False
        self.vi_approx = None
        self._conditional_parameter_info: dict[str, ConditionalParameterInfo] = {}
        self._group_specific_state: GroupSpecificGraphState = GroupSpecificGraphState()

    def build(self) -> None:
        response_coords_data, response_coords, response_coords_reduced = coords_from_response(
            self.spec.response_term, self.spec.family
        )

        model = pm.Model(coords=response_coords_data | response_coords | response_coords_reduced)
        model.__bambi_attrs__ = {
            "response_ndim": self.spec.family.RESPONSE_NDIM,
            "response_coords_data": response_coords_data,
            "response_coords": response_coords,
            "response_coords_reduced": response_coords_reduced,
        }

        marginal_parameters = {}
        conditional_parameters = {}
        self._conditional_parameter_info = {}
        self._group_specific_state = GroupSpecificGraphState()
        for name, parameter in self.spec.marginal_parameters.items():
            marginal_parameters[name] = build_marginal_parameter(parameter, self.spec.family, model)

        for name, parameter in self.spec.conditional_parameters.items():
            parameter_info = make_conditional_parameter_info(parameter)
            self._conditional_parameter_info[name] = parameter_info
            conditional_parameters[name] = build_conditional_parameter(
                parameter_info, self.spec.family, self._group_specific_state, model
            )

        build_response_term(
            term=self.spec.response_term,
            parameters=marginal_parameters | conditional_parameters,
            family=self.spec.family,
            model=model,
        )

        build_potentials(self.spec.potentials, model)

        self.model = model

    def run(
        self,
        draws=1000,
        tune=1000,
        chains=None,
        cores=None,
        discard_tuned_samples=True,
        omit_offsets=True,
        include_response_params=False,
        inference_method=None,
        init="auto",
        n_init=50000,
        random_seed=None,
        nuts=None,
        **kwargs,
    ):
        """Run PyMC sampler."""
        if inference_method is not None:
            inference_method = inference_method.lower()

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
                chains=chains,
                cores=cores,
                discard_tuned_samples=discard_tuned_samples,
                omit_offsets=omit_offsets,
                include_response_params=include_response_params,
                init=init,
                n_init=n_init,
                random_seed=random_seed,
                nuts=nuts,
                sampler_backend=inference_method,
                **kwargs,
            )

        self.fit = True
        return result

    def predict(
        self,
        idata,
        data=None,
        include_group_specific=True,
        random_seed=None,
        kind="response",
        inplace=True,
        progressbar=True,
    ):
        if not inplace:
            idata = deepcopy(idata)

        output_groups = ()
        if data is not None:
            output_groups = ("predictions", "predictions_constant_data")
        elif kind in ("response", "response_latent"):
            output_groups = ("posterior_predictive",)

        for group in output_groups:
            if group in idata:
                del idata[group]

        parameters_names = [param.label for param in self.spec.conditional_parameters.values()]
        responses_names = [self.spec.response_term.label]

        # If group-specific offsets are discarded, we add them back.
        # They are needed for the computation of deterministics (model parameters).
        posterior = as_dataset(idata["posterior"])
        offset_values = {}
        for parameter_info in self._conditional_parameter_info.values():
            for term_info in parameter_info.group_specific_terms:
                term = term_info.term
                term_label = term.label
                offset_name = f"{term_label}_offset"
                if term.noncentered and offset_name not in posterior:
                    sigma_name = term.hyperprior_alias.get("sigma", "sigma")
                    offset_values[offset_name] = (
                        posterior[term_label] / posterior[f"{term_label}_{sigma_name}"]
                    )

        if data is None:
            self._predict_in_sample(
                idata,
                offset_values,
                parameters_names,
                responses_names,
                include_group_specific,
                random_seed,
                kind,
                progressbar,
            )
        else:
            self._predict_out_of_sample(
                idata,
                data,
                offset_values,
                parameters_names,
                responses_names,
                include_group_specific,
                random_seed,
                kind,
                progressbar,
            )

        if inplace:
            return None

        return idata

    def compute_log_likelihood(
        self,
        idata,
        data: Optional[pd.DataFrame],
        inplace: bool = True,
        progressbar: bool = True,
    ):
        if not inplace:
            idata = deepcopy(idata)

        if "log_likelihood" in idata:
            del idata["log_likelihood"]

        posterior = as_dataset(idata["posterior"])

        offset_values = {}
        for parameter_info in self._conditional_parameter_info.values():
            for term_info in parameter_info.group_specific_terms:
                term = term_info.term
                term_label = term.label
                offset_name = f"{term_label}_offset"
                if term.noncentered and offset_name not in posterior:
                    sigma_name = term.hyperprior_alias.get("sigma", "sigma")
                    offset_values[offset_name] = (
                        posterior[term_label] / posterior[f"{term_label}_{sigma_name}"]
                    )

        trace = idata
        if offset_values:
            trace = idata.copy()
            trace["posterior"] = posterior.assign(offset_values)

        if data is None:
            self._compute_log_likelihood_in_sample(trace, progressbar)
        else:
            self._compute_log_likelihood_out_of_sample(trace, data, progressbar)

        if offset_values:
            idata["log_likelihood"] = as_dataset(trace["log_likelihood"])

        idata["log_likelihood"] = as_dataset(idata["log_likelihood"]).assign_attrs(
            modeling_interface="bambi", modeling_interface_version=__version__
        )

        if inplace:
            return None

        return idata

    def compute_log_prior(self, idata, inplace: bool = True):
        if not inplace:
            idata = deepcopy(idata)

        # Reproduce PyMC's compute_log_prior logic so we can exclude posterior variables
        # that do not correspond to free random variables, such as omitted offsets.
        model = self.model
        untransformed_model = remove_value_transforms(model)
        coords, dims = coords_and_dims_for_inferencedata(untransformed_model)

        deterministic_names = {deterministic.name for deterministic in model.deterministics}
        posterior = as_dataset(idata["posterior"])
        target_rvs = [
            rv
            for rv in untransformed_model.free_RVs
            if rv.name not in deterministic_names and rv.name in posterior
        ]
        target_names = [rv.name for rv in target_rvs]
        value_vars = [untransformed_model.rvs_to_values[rv] for rv in target_rvs]

        elemwise_logprior_fn = untransformed_model.compile_fn(
            inputs=value_vars,
            outs=untransformed_model.logp(vars=target_rvs, sum=False),
            on_unused_input="ignore",
        )
        input_dataset = posterior[target_names].astype(
            {value_var.name: value_var.type.dtype for value_var in value_vars}, copy=False
        )
        logdens = apply_function_over_dataset(
            elemwise_logprior_fn,
            input_dataset,
            output_var_names=target_names,
            sample_dims=("chain", "draw"),
            dims=dims,
            coords=coords,
            progressbar=False,
        )
        log_prior = xr.Dataset({name: logdens[name] for name in target_names})

        if "log_prior" in idata:
            del idata["log_prior"]
        idata["log_prior"] = log_prior.assign_attrs(
            modeling_interface="bambi", modeling_interface_version=__version__
        )

        if inplace:
            return None

        return idata

    def _predict_in_sample(
        self,
        idata,
        offset_values,
        parameters_names,
        responses_names,
        include_group_specific,
        random_seed,
        kind,
        progressbar,
    ) -> None:
        if offset_values:
            posterior_for_prediction = as_dataset(idata["posterior"]).assign(offset_values)
        else:
            posterior_for_prediction = as_dataset(idata["posterior"])

        model = self.model
        if not include_group_specific:
            model = remove_group_specific_contributions(self._group_specific_state, model)

        # It's assumed the user always wants the parameter 'predictions' (mu, sigma, etc.)
        with model:
            posterior_for_prediction = pm.compute_deterministics(
                dataset=posterior_for_prediction,
                var_names=parameters_names,
                progressbar=progressbar,
                merge_dataset=True,
            )
        idata["posterior"] = as_dataset(idata["posterior"]).merge(
            posterior_for_prediction[parameters_names], compat="override"
        )

        if kind in ("response", "response_latent"):
            prediction_model = pm.model.fgraph.clone_model(model)
            prediction_model = replace_response_variables(
                self.spec.response_term, prediction_model, kind
            )
            interventions = build_response_interventions(
                self.spec.response_term, prediction_model, kind
            )
            if interventions:
                prediction_model = pm.do(prediction_model, interventions)

            with prediction_model:
                predictions = pm.sample_posterior_predictive(
                    trace=posterior_for_prediction,
                    var_names=responses_names,
                    random_seed=random_seed,
                )

            idata["posterior_predictive"] = as_dataset(predictions["posterior_predictive"])

    def _predict_out_of_sample(
        self,
        idata,
        data: pd.DataFrame,
        offset_values,
        parameters_names,
        responses_names,
        include_group_specific,
        random_seed,
        kind,
        progressbar,
    ) -> None:
        new_data, new_coords, factor_plans = self._build_new_data(data, "prediction", kind)
        out_of_sample_plans = [
            plan for plan in factor_plans if (plan.groups_index == -1).any() or plan.groups_new
        ]
        var_names = parameters_names[:]
        if kind in ("response", "response_latent"):
            var_names += responses_names

        trace = as_dataset(idata["posterior"]).assign(offset_values)

        model, group_specific_state = _clone_model_with_group_specific_state(
            self._group_specific_state, self.model
        )
        model = replace_response_variables(self.spec.response_term, model, kind)
        pm.set_data(new_data=new_data, coords=new_coords, model=model)

        if not include_group_specific:
            model = remove_group_specific_contributions(group_specific_state, model)
        elif out_of_sample_plans:
            if bmb_config["SPARSE_DOT"]:
                model = add_new_sparse_group_specific_contributions(
                    factor_plans, group_specific_state, model
                )
            else:
                model = add_new_dense_group_specific_contributions(
                    factor_plans, group_specific_state, model
                )

        interventions = build_response_interventions(self.spec.response_term, model, kind)
        if interventions:
            model = pm.do(model, interventions)

        with model:
            predictions = pm.sample_posterior_predictive(
                trace=trace,
                var_names=var_names,
                progressbar=progressbar,
                random_seed=random_seed,
                extend_inferencedata=False,
                predictions=True,
            )

        idata["predictions"] = as_dataset(predictions["predictions"])
        if "predictions_constant_data" in predictions:
            idata["predictions_constant_data"] = as_dataset(
                predictions["predictions_constant_data"]
            )

    def _compute_log_likelihood_in_sample(self, trace, progressbar) -> None:
        with self.model:
            pm.compute_log_likelihood(
                idata=trace, extend_inferencedata=True, progressbar=progressbar
            )

    def _compute_log_likelihood_out_of_sample(self, trace, data: pd.DataFrame, progressbar) -> None:
        new_data, new_coords, factor_plans = self._build_new_data(data, "log_likelihood")
        out_of_sample_plans = [
            plan for plan in factor_plans if (plan.groups_index == -1).any() or plan.groups_new
        ]

        if out_of_sample_plans:
            factors = tuple(plan.factor_name for plan in out_of_sample_plans)
            raise ValueError(
                f"Cannot compute log likelihood for new groups of the factors {factors}."
            )

        model = pm.model.fgraph.clone_model(self.model)
        pm.set_data(new_data, coords=new_coords, model=model)
        model = replace_response_variables(self.spec.response_term, model)

        with model:
            pm.compute_log_likelihood(
                idata=trace,
                var_names=[self.spec.response_term.label],
                extend_inferencedata=True,
                progressbar=progressbar,
            )

    def _build_new_data(self, data: pd.DataFrame, purpose: str, kind: str | None = None):
        new_coords = {"__obs__": range(len(data))}
        new_data = build_new_response_data(
            self.spec.response_term, data, self.spec.family, purpose, kind
        )
        factor_plans: list[DenseGroupSpecificFactorPlan] = []

        for parameter_info in self._conditional_parameter_info.values():
            if bmb_config["SPARSE_DOT"] and parameter_info.group_specific_terms:
                parameter_data, parameter_factor_plans, parameter_coords = (
                    build_new_sparse_conditional_parameter_data(parameter_info, data, self.model)
                )
                new_coords.update(parameter_coords)
            else:
                parameter_data, parameter_factor_plans = build_new_dense_conditional_parameter_data(
                    parameter_info, data, self.model
                )
            new_data.update(parameter_data)
            factor_plans.extend(parameter_factor_plans)

        return new_data, new_coords, factor_plans

    def _run_mcmc(
        self,
        draws,
        tune,
        chains,
        cores,
        discard_tuned_samples,
        omit_offsets,
        include_response_params,
        init,
        n_init,
        random_seed,
        nuts,
        sampler_backend,
        **kwargs,
    ):
        vars_to_sample = pm.util.get_default_varnames(
            self.model.unobserved_value_vars, include_transformed=False
        )
        vars_to_sample = [variable.name for variable in vars_to_sample]

        if not include_response_params:
            parameters_names = [param.label for param in self.spec.conditional_parameters.values()]
            vars_to_sample = [var for var in vars_to_sample if var not in parameters_names]

        if omit_offsets:
            vars_to_sample = [var for var in vars_to_sample if not var.endswith("_offset")]

        # pm.sample routes nuts settings via kwargs.pop("nuts", {}); only inject when provided
        # to avoid passing nuts=None which causes pm.sample's internal nuts_kwargs.copy() to fail.
        if nuts is not None:
            kwargs["nuts"] = nuts

        with self.model:
            try:
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    discard_tuned_samples=discard_tuned_samples,
                    init=init,
                    n_init=n_init,
                    random_seed=random_seed,
                    var_names=vars_to_sample,
                    nuts_sampler=sampler_backend,
                    **kwargs,
                )
            except (RuntimeError, ValueError):
                if "ValueError: Mass matrix contains" in traceback.format_exc() and init == "auto":
                    _logger.info(
                        "\nThe default initialization using init='auto' has failed, trying to "
                        "recover by switching to init='adapt_diag'",
                    )
                    idata = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        cores=cores,
                        discard_tuned_samples=discard_tuned_samples,
                        init="adapt_diag",
                        n_init=n_init,
                        random_seed=random_seed,
                        var_names=vars_to_sample,
                        nuts_sampler=sampler_backend,
                        **kwargs,
                    )
                else:
                    raise

        for group in idata.children:
            idata[group] = as_dataset(idata[group]).assign_attrs(
                modeling_interface="bambi", modeling_interface_version=__version__
            )

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
            Omits offset terms in the `DataTree` object returned when the model includes
            group specific effects.
        include_response_params : bool
            Include parameters of the response distribution in the output.

        Returns
        -------
        A DataTree containing the posterior draws.
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
        modes = np.concatenate(
            [np.atleast_1d(maps[value_var.name]) for value_var in self.model.value_vars]
        )

        samples = np.random.multivariate_normal(modes, cov, size=draws)

        response_parameter_names = [
            parameter.label for parameter in self.spec.conditional_parameters.values()
        ]
        idata = _posterior_samples_to_idata(
            samples,
            self.model,
            excluded_var_names=response_parameter_names,
        )

        if include_response_params:
            with self.model:
                posterior = pm.compute_deterministics(
                    dataset=as_dataset(idata["posterior"]),
                    var_names=response_parameter_names,
                    merge_dataset=True,
                    progressbar=False,
                )
            idata["posterior"] = posterior

        if omit_offsets:
            posterior = as_dataset(idata["posterior"])
            offset_vars = [var for var in posterior.data_vars if var.endswith("_offset")]
            idata["posterior"] = posterior.drop_vars(offset_vars)

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


def _clone_model_with_group_specific_state(
    group_specific_state: GroupSpecificGraphState, model: pm.Model
) -> tuple[pm.Model, GroupSpecificGraphState]:
    """Clone a model and map group-specific graph state to cloned variables."""
    fgraph, memo = fgraph_from_model(model)
    cloned_model = model_from_fgraph(fgraph, mutate_fgraph=True)
    return cloned_model, group_specific_state.clone(memo)


def _posterior_samples_to_idata(
    samples: np.ndarray,
    model: pm.Model,
    excluded_var_names: tuple[str, ...] = (),
) -> DataTree:
    """Convert Laplace samples in value-variable space to a `DataTree`.

    Parameters
    ----------
    samples : np.ndarray
        Posterior draws with shape `(draw, n_value_variables)`. Values must be ordered
        according to `model.value_vars` and stored in their unconstrained representation.
    model : pm.Model
        PyMC model that defines the value variables and the unobserved variables to record.
    excluded_var_names : tuple[str, ...], optional
        Names of unobserved variables not to evaluate or store in the posterior trace.

    Returns
    -------
    DataTree
        Posterior draws converted to constrained variables and deterministics selected for the
        trace.
    """
    initial_point = model.initial_point()
    variables = model.value_vars

    var_info = {}
    for name, value in initial_point.items():
        var_info[name] = (value.shape, value.size)

    length_pos = len(samples)
    varnames = [v.name for v in variables]

    variables = [
        variable
        for variable in pm.util.get_default_varnames(
            model.unobserved_value_vars, include_transformed=False
        )
        if variable.name not in excluded_var_names
    ]

    with model:
        # pylint:disable=no-member
        strace = pm.backends.ndarray.NDArray(name=model.name, vars=variables)
        strace.setup(length_pos, 0)

    for i in range(length_pos):
        value = []
        size = 0
        for varname in varnames:
            shape, new_size = var_info[varname]
            var_samples = samples[i][size : size + new_size]
            value.append(var_samples.reshape(shape))
            size += new_size
        strace.record(point=dict(zip(varnames, value)), in_warmup=False)

    idata = pm.to_inference_data(pm.backends.base.MultiTrace([strace]), model=model)
    return idata
