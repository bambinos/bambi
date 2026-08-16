import inspect
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from formulae.terms.call_utils import CallVarsExtractor
from formulae.terms.call_resolver import get_function_from_module
from pymc.distributions.continuous import TruncatedNormalRV
from pymc.distributions.truncated import TruncatedRV
from pymc.model.fgraph import fgraph_from_model, model_from_fgraph
from pymc.pytensorf import toposort_replace

from bambi.backend.pymc.utils import (
    make_weighted_distribution,
    get_distribution_from_likelihood,
)
from bambi.backend.pymc.transform import transforms_registry
from bambi.families.family import Family
from bambi.families.types import ResponseType
from bambi.terms.response import ResponseTerm


def build_response_term(
    term: ResponseTerm, parameters: dict, family: Family, model: pm.Model
) -> None:
    distribution = get_distribution_from_likelihood(family.likelihood)

    # All families get coordinates for observation indexes.
    # Multidimensional models also get additional coords, if available.
    dims = tuple(model.__bambi_attrs__["response_coords_data"])
    if family.RESPONSE_NDIM > 0:
        dims = dims + tuple(model.__bambi_attrs__["response_coords"])

    transform_parameters = transforms_registry.get_parameter_transform(family)
    if transform_parameters is not None:
        parameters = transform_parameters(parameters)

    if term.is_censored:
        data_mapping = _build_censored_data(term, dims, model)
        dist = distribution.dist(**parameters)
        with model:
            pm.Censored(term.label, dist, **data_mapping, dims=dims)
        return None

    if term.is_truncated or term.is_constrained:
        data_mapping = _build_truncated_data(term, dims, model)
        dist = distribution.dist(**parameters)
        with model:
            pm.Truncated(term.label, dist, **data_mapping, dims=dims)
        return None

    if term.is_weighted:
        data_mapping = _build_weighted_data(term, dims, model)
        weighted_dist = make_weighted_distribution(distribution)
        with model:
            weighted_dist(term.label, **data_mapping, **parameters, dims=dims)
        return None

    if term.is_counts:
        data_mapping = _build_counts_data(term, dims, model)
        with model:
            distribution(term.label, **parameters, **data_mapping, dims=dims)
        return None

    if term.is_binomial:
        data_mapping = _build_binomial_data(term, dims, model)
        with model:
            distribution(term.label, **parameters, **data_mapping, dims=dims)

        return None

    data = term.data
    if family.DATA_TYPE == ResponseType.BINARY and data.ndim > 1:
        # In a binary response model, when the user uses a categoric response without setting the
        # reference level, the data will be a 2D one-hot encoded matrix.
        # In that case, we select the corresponding column for the reference level.
        # Otherwise, the data is already a 1D binary array and no further action is needed.
        index = term.levels.index(term.reference)
        data = data[:, index]
    elif family.DATA_TYPE in (ResponseType.CATEGORICAL, ResponseType.ORDINAL):
        # In categorical and ordinal response models, the data is a 2D one-hot encoded matrix,
        # but PyMC needs a vector of observed category indices.
        data = np.nonzero(data)[1]

    transform_data = transforms_registry.get_data_transform(family)
    if transform_data is not None:
        data_mapping = transform_data(data)
    else:
        data_mapping = {"observed": data}

    data_vars = {}
    for name, value in data_mapping.items():
        if name == "observed":
            label = term.label + "_data"
        else:
            label = name + "_data"

        data_vars[name] = pm.Data(label, value, dims=dims, model=model)

    with model:
        distribution(term.label, **parameters, **data_vars, dims=dims)

    return None


Purpose = Literal["prediction", "log_likelihood"]


def _untruncate_response(response_name: str, model: pm.Model) -> pm.Model:
    """Return a copy of `model` with one truncated response made latent.

    The transformation preserves the response as an observed RV and removes the now-disconnected
    bound data containers.
    """
    fgraph, memo = fgraph_from_model(model)
    response = memo[model[response_name]]
    truncated_rv = response.owner.inputs[0]
    untruncated_rv = _get_untruncated_rv(truncated_rv)
    toposort_replace(fgraph, [(truncated_rv, untruncated_rv)], reverse=True)

    response = model[response_name]
    lower, upper = response.owner.inputs[-2:]
    bound_data_names = {
        bound.name
        for bound in (lower, upper)
        if bound.name is not None and bound.name.endswith("_data")
    }

    for index in reversed(range(len(fgraph.outputs))):
        if fgraph.outputs[index].name in bound_data_names:
            fgraph.remove_output(index)

    return model_from_fgraph(fgraph)


def replace_response_variables(
    term: ResponseTerm, model: pm.Model, kind: str | None = None
) -> pm.Model:
    """Apply response-variable replacements required by the current cloned data."""
    if term.is_truncated and kind == "response_latent":
        return _untruncate_response(term.label, model)

    return model


def build_response_interventions(
    term: ResponseTerm, model: pm.Model, kind: str
) -> dict[pt.TensorVariable, pt.TensorVariable]:
    """Build `pm.do` response interventions for posterior prediction.

    `model` must be an out-of-sample clone whose mutable data has already been
    updated.  This is important because the intervention uses the response data
    containers from that clone, rather than the data in the fitted model.
    """
    if term.is_censored and kind == "response_latent":
        return _build_intervention_censored(term, model)

    return {}


def build_new_response_data(
    term: ResponseTerm,
    data: pd.DataFrame,
    family: Family,
    purpose: Purpose,
    kind: str | None = None,
):  # pylint: disable=too-many-return-statements
    if purpose not in ("prediction", "log_likelihood"):
        raise ValueError(f"Unsupported purpose: {purpose}")

    if term.is_censored:
        return _build_new_censored_data(term, data, purpose, kind)

    if term.is_truncated:
        return _build_new_truncated_data(term, data, purpose, kind)

    if term.is_constrained:
        return _build_new_constrained_data(term, data, purpose)

    if term.is_weighted:
        return _build_new_weighted_data(term, data, purpose)

    if term.is_counts:
        return _build_new_counts_data(term, data, purpose)

    if term.is_binomial:
        return _build_new_binomial_data(term, data, purpose)

    return _build_new_generic_data(term, data, family, purpose)


def _get_untruncated_rv(truncated_rv: pt.TensorVariable) -> pt.TensorVariable:
    """Rebuild the base RV of a PyMC truncated random variable."""
    if truncated_rv.owner is None:
        raise ValueError("The response must be a PyMC random variable.")

    op = truncated_rv.owner.op
    inputs = truncated_rv.owner.inputs
    if isinstance(op, TruncatedRV):
        # Generic pm.Truncated stores the base RandomVariable Op and its RNG,
        # size, and distribution parameters before the final lower and upper
        # bound inputs.
        return op.base_rv_op.make_node(*inputs[:-2]).default_output()

    if isinstance(op, TruncatedNormalRV):
        rng, size, mu, sigma, _, _ = inputs
        return pm.Normal.dist(mu=mu, sigma=sigma, size=size, rng=rng)

    raise ValueError(f"Cannot reconstruct the base distribution for {op}.")


def _build_intervention_censored(
    term: ResponseTerm, model: pm.Model
) -> dict[pt.TensorVariable, pt.TensorVariable]:
    """Replace a censored response with its conditional latent distribution."""
    response = model[term.label]
    base_dist = response.owner.inputs[0]
    call_args = _get_call_bound_arguments(term)
    observed = model[call_args["x"] + "_data"]
    status = model[call_args["status"] + "_data"]

    # A lower censoring limit means that the latent response is below that
    # limit, and an upper censoring limit means it is above that limit.  The
    # bounds are consequently reversed when building the truncated latent
    # distribution.  Construct them from the clone's response data: the
    # Censored bounds use opposite infinities for observations of the other
    # censoring type, which cannot be exchanged directly.
    lower = pt.switch(pt.eq(status, 1), observed, -np.inf)
    upper = pt.switch(pt.eq(status, -1), observed, np.inf)
    intervention = pm.Truncated.dist(base_dist, lower=lower, upper=upper)

    return {response: intervention}


def _build_censored_data(
    term: ResponseTerm, dims: tuple[str], model: pm.Model
) -> dict[str, pt.TensorVariable]:
    # NOTE: Statuses could be more efficient (in some cases) if we allowed for scalars.
    #       For now, statuses are vectors of the same length as observed data.
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    status_name = call_args["status"]
    observed, status = term.data[:, 0], term.data[:, 1]
    observed_data = pm.Data(value_name + "_data", observed, dims=dims, model=model)
    status_data = pm.Data(status_name + "_data", status, dims=dims, model=model)

    lower, upper = -np.inf, np.inf
    if any(status == -1):
        is_left_censored = pt.eq(status_data, -1)
        lower = pt.switch(is_left_censored, observed_data, -np.inf)

    if any(status == 1):
        is_right_censored = pt.eq(status_data, 1)
        upper = pt.switch(is_right_censored, observed_data, np.inf)

    return {"lower": lower, "upper": upper, "observed": observed_data}


def _build_truncated_data(term: ResponseTerm, dims: tuple[str], model: pm.Model) -> dict:
    observed, lower, upper = term.data[:, 0], term.data[:, 1], term.data[:, 2]
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    observed_data = pm.Data(value_name + "_data", observed, dims=dims, model=model)

    if "lb" not in call_args:
        lower_data = None
    elif call_args["lb"] == "":
        # A literal, all observations share the same lower bound.
        lower_data = lower[0].item()
    else:
        # A variable name, lower bound is a vector of the same length as observed data.
        lower_name = call_args["lb"]
        lower_data = pm.Data(lower_name + "_data", lower, dims=dims, model=model)

    if "ub" not in call_args:
        upper_data = None
    elif call_args["ub"] == "":
        # A literal, all observations share the same upper bound.
        upper_data = upper[0].item()
    else:
        # A variable name, upper bound is a vector of the same length as observed data.
        upper_name = call_args["ub"]
        upper_data = pm.Data(upper_name + "_data", upper, dims=dims, model=model)

    return {"lower": lower_data, "upper": upper_data, "observed": observed_data}


def _build_weighted_data(term: ResponseTerm, dims: tuple[str], model: pm.Model) -> dict:
    observed, weights = term.data[:, 0], term.data[:, 1]
    call_args = _get_call_bound_arguments(term)

    value_name = call_args["x"]
    observed_data = pm.Data(value_name + "_data", observed, dims=dims, model=model)

    if call_args["weights"] == "":
        # A literal, all observations share the same weight.
        weights_data = weights[0].item()
    else:
        # A variable name, weights are a vector of the same length as observed data.
        weights_name = call_args["weights"]
        weights_data = pm.Data(weights_name + "_data", weights, dims=dims, model=model)

    return {"weights": weights_data, "observed": observed_data}


def _build_counts_data(term: ResponseTerm, dims: tuple[str], model: pm.Model) -> dict:
    observed_data = pm.Data(term.label + "_data", term.data, dims=dims, model=model)
    n_argument = term.components[0].call.kwargs.get("n")
    n_name = getattr(n_argument, "name", None)

    if n_argument is None:
        n_data = pt.sum(observed_data, axis=1)
    elif n_name is None:
        n_data = term.data.sum(axis=1)[0].item()
    else:
        obs_dims = tuple(model.__bambi_attrs__["response_coords_data"])
        n_data = pm.Data(n_name + "_data", term.data.sum(axis=1), dims=obs_dims, model=model)

    return {"observed": observed_data, "n": n_data}


def _build_binomial_data(term: ResponseTerm, dims: tuple[str], model: pm.Model) -> dict:
    successes, trials = term.data[:, 0], term.data[:, 1]
    call_args = _get_call_bound_arguments(term)

    successes_name = call_args["successes"]
    successes_data = pm.Data(successes_name + "_data", successes, dims=dims, model=model)

    if call_args["trials"] == "":
        # A literal, all observations share the same number of trials.
        trials_data = trials[0].item()
    else:
        # A variable name, trials are a vector of the same length as observed data.
        trials_name = call_args["trials"]
        trials_data = pm.Data(trials_name + "_data", trials, dims=dims, model=model)

    return {"observed": successes_data, "n": trials_data}


def _build_new_censored_data(
    term: ResponseTerm, data: pd.DataFrame, purpose: Purpose, kind: str | None
):
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    status_name = call_args["status"]
    n = data.shape[0]
    data_dict = {}

    # For posterior prediction, these data only provide the observed response
    # distribution. Whether a latent response is requested is determined by
    # ``kind`` in ``Model.predict``, not by which response columns are present.
    # Log-likelihood defaults status to "none".
    if purpose == "prediction":
        if kind == "response" and (
            value_name not in data.columns or status_name not in data.columns
        ):
            raise ValueError(
                "Censored response predictions require both "
                f"'{value_name}' and '{status_name}' in the data. "
                "Use kind='response_latent' for latent predictions."
            )
        should_evaluate = value_name in data.columns and status_name in data.columns
        if should_evaluate:
            data_dict[value_name] = data[value_name].to_numpy()
            data_dict[status_name] = data[status_name].to_numpy()
    else:
        if value_name not in data.columns:
            raise ValueError(f"Response term variable '{value_name}' must be present in the data.")

        should_evaluate = True
        data_dict[value_name] = data[value_name].to_numpy()
        if status_name in data.columns:
            data_dict[status_name] = data[status_name].to_numpy()
        else:
            data_dict[status_name] = np.full(n, "none")

    if should_evaluate:
        response_data = term.eval_new_data(pd.DataFrame(data_dict))
        value, status = response_data[:, 0], response_data[:, 1]
    else:
        value = np.full(n, term.data[0, 0])
        status = np.zeros(n, dtype=int)

    return {value_name + "_data": value, status_name + "_data": status}


def _build_new_truncated_data(
    term: ResponseTerm, data: pd.DataFrame, purpose: Purpose, kind: str | None
):
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    lower_name = call_args.get("lb", "")
    upper_name = call_args.get("ub", "")
    n = data.shape[0]

    requires_bounds = purpose == "log_likelihood" or kind == "response"
    if requires_bounds:
        missing_bounds = [name for name in (lower_name, upper_name) if name and name not in data]
        if missing_bounds:
            names = ", ".join(repr(name) for name in missing_bounds)
            if purpose == "log_likelihood":
                raise ValueError(
                    f"Truncated response log-likelihood requires bound variables {names} in data."
                )
            raise ValueError(
                f"Truncated response predictions require bound variables {names} in data. "
                "Use kind='response_latent' for latent predictions."
            )

    if purpose == "prediction" and kind == "response_latent":
        # The latent distribution is the untruncated base distribution, so its
        # named bounds are neither required nor evaluated. The response data
        # only keeps the cloned model's observation dimension in sync before
        # the observed truncated RV is replaced.
        return {value_name + "_data": np.full(n, term.data[0, 0])}

    # Re-evaluate the response call so transformed values and bounds stay consistent.
    # Literal bounds need no data value.
    data_dict = {}
    if lower_name:
        data_dict[lower_name] = data[lower_name].to_numpy()

    if upper_name:
        data_dict[upper_name] = data[upper_name].to_numpy()

    if purpose == "prediction":
        value_data = np.full(n, term.data[0, 0])
    else:
        if value_name not in data.columns:
            raise ValueError(f"Response term variable '{value_name}' must be present in the data.")
        value_data = data[value_name].to_numpy()

    data_dict = {value_name: value_data, **data_dict}
    response_data = term.eval_new_data(pd.DataFrame(data_dict))
    value, lower, upper = response_data[:, 0], response_data[:, 1], response_data[:, 2]

    output = {value_name + "_data": value}
    if lower_name:
        output[lower_name + "_data"] = lower
    if upper_name:
        output[upper_name + "_data"] = upper

    return output


def _build_new_constrained_data(term: ResponseTerm, data: pd.DataFrame, purpose: Purpose):
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    lower_name = call_args.get("lb", "")
    upper_name = call_args.get("ub", "")
    n = data.shape[0]

    bound_names = [name for name in (lower_name, upper_name) if name]
    data_dict = {}

    # Unlike truncation, named bounds must be present in new data.
    # Only the response value gets a default from the original data when predicting.
    if purpose == "prediction":
        var_names = bound_names
        data_dict[value_name] = np.full(n, term.data[0, 0])
    else:
        var_names = [value_name] + bound_names
        if value_name in data.columns:
            data_dict[value_name] = data[value_name].to_numpy()

    missing_var_names = [name for name in var_names if name not in data.columns]
    if missing_var_names:
        present_var_names = [name for name in var_names if name in data.columns]
        raise ValueError(
            "Response term variables must be present in the data.\n"
            f"Required variables: {var_names}.\n"
            f"Present variables: {present_var_names}."
        )

    for name in bound_names:
        data_dict[name] = data[name].to_numpy()

    response_data = term.eval_new_data(pd.DataFrame(data_dict))
    value, lower, upper = response_data[:, 0], response_data[:, 1], response_data[:, 2]
    output = {value_name + "_data": value}

    if lower_name:
        output[lower_name + "_data"] = lower
    if upper_name:
        output[upper_name + "_data"] = upper

    return output


def _build_new_weighted_data(term: ResponseTerm, data: pd.DataFrame, purpose: Purpose):
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    weights_name = call_args.get("weights", "")
    n = data.shape[0]

    # The response value is required for log-likelihood.
    # Missing weights default to one.
    if purpose == "prediction":
        value_data = np.full(n, term.data[0, 0])
    else:
        if value_name not in data.columns:
            raise ValueError(f"Response term variable '{value_name}' must be present in the data.")
        value_data = data[value_name].to_numpy()

    data_dict = {value_name: value_data}
    if weights_name:
        if weights_name in data.columns:
            data_dict[weights_name] = data[weights_name].to_numpy()
        else:
            data_dict[weights_name] = np.ones(n)

    response_data = term.eval_new_data(pd.DataFrame(data_dict))
    value, weights = response_data[:, 0], response_data[:, 1]
    output = {value_name + "_data": value}

    if weights_name:
        output[weights_name + "_data"] = weights

    return output


def _build_new_counts_data(term: ResponseTerm, data: pd.DataFrame, purpose: Purpose):
    component = term.components[0]
    count_names = [argument.name for argument in component.call.args]
    n_argument = component.call.kwargs.get("n")
    n_name = getattr(n_argument, "name", None)
    n = data.shape[0]

    if purpose == "prediction":
        observed = np.zeros((n, term.data.shape[1]), dtype=term.data.dtype)
        if n_argument is None:
            fixed_n = term.data[0].sum()
            observed[:, 0] = fixed_n
            warnings.warn(
                "Using the first training total for predictions from 'counts' without 'n'. "
                "Pass 'n' as a variable to update it with new data.",
                UserWarning,
            )

        output = {term.label + "_data": observed}
        if n_name is not None:
            if n_name not in data.columns:
                raise ValueError(f"Response term variable '{n_name}' must be present in the data.")
            output[n_name + "_data"] = data[n_name].to_numpy()
        return output

    var_names = count_names + ([n_name] if n_name is not None else [])
    missing_var_names = [name for name in var_names if name not in data.columns]
    if missing_var_names:
        present_var_names = [name for name in var_names if name in data.columns]
        raise ValueError(
            "Response term variables must be present in the data.\n"
            f"Required variables: {var_names}.\n"
            f"Present variables: {present_var_names}."
        )

    response_data = term.eval_new_data(data[var_names])
    output = {term.label + "_data": response_data}
    if n_name is not None:
        output[n_name + "_data"] = response_data.sum(axis=1)
    return output


def _build_new_binomial_data(term: ResponseTerm, data: pd.DataFrame, purpose: Purpose):
    call_args = _get_call_bound_arguments(term)
    successes_name = call_args["successes"]
    trials_name = call_args.get("trials", "")
    n = data.shape[0]
    var_names = [trials_name] if trials_name else []

    # Successes are required only for log-likelihood.
    # Trials must be provided if they were not provided as literals.
    if purpose == "log_likelihood":
        var_names = [successes_name] + var_names

    missing_var_names = [name for name in var_names if name not in data.columns]
    if missing_var_names:
        present_var_names = [name for name in var_names if name in data.columns]
        raise ValueError(
            "Response term variables must be present in the data.\n"
            f"Required variables: {var_names}.\n"
            f"Present variables: {present_var_names}."
        )

    if purpose == "prediction":
        data_dict = {successes_name: np.zeros(n, dtype=term.data.dtype)}
    else:
        data_dict = {successes_name: data[successes_name].to_numpy()}

    if trials_name:
        data_dict[trials_name] = data[trials_name].to_numpy()

    response_data = term.eval_new_data(pd.DataFrame(data_dict))
    successes, trials = response_data[:, 0], response_data[:, 1]
    output = {successes_name + "_data": successes}

    if trials_name:
        output[trials_name + "_data"] = trials

    return output


def _build_new_generic_data(
    term: ResponseTerm, data: pd.DataFrame, family: Family, purpose: Purpose
):
    var_names = list(term.term.var_names)
    n = data.shape[0]

    if purpose == "prediction":
        data_dict = {
            name: (
                data[name].to_numpy()
                if name in data.columns
                # Formulae validates raw category labels before Bambi derives PyMC category indexes.
                # Use a fitted level to evaluate a missing categorical response.
                else (
                    np.full(n, term.levels[0])
                    if term.categorical
                    else np.zeros(n, dtype=term.data.dtype)
                )
            )
            for name in var_names
        }
    else:
        missing_var_names = [name for name in var_names if name not in data.columns]
        if missing_var_names:
            present_var_names = [name for name in var_names if name in data.columns]
            raise ValueError(
                "Response term variables must be present in the data.\n"
                f"Required variables: {var_names}.\n"
                f"Present variables: {present_var_names}."
            )
        data_dict = {name: data[name].to_numpy() for name in var_names}

    response_data = term.eval_new_data(pd.DataFrame(data_dict))

    if family.DATA_TYPE == ResponseType.BINARY and response_data.ndim > 1:
        index = term.levels.index(term.reference)
        response_data = response_data[:, index]
    elif family.DATA_TYPE in (ResponseType.CATEGORICAL, ResponseType.ORDINAL):
        response_data = np.nonzero(response_data)[1]

    transform_data = transforms_registry.get_data_transform(family)
    if transform_data is not None:
        data_mapping = transform_data(response_data)
    else:
        data_mapping = {"observed": response_data}

    output = {}

    for name, value in data_mapping.items():
        if name == "observed":
            output[term.label + "_data"] = (
                np.zeros_like(value) if purpose == "prediction" else value
            )
        elif name in data.columns and purpose == "prediction":
            output[name + "_data"] = data[name]
        else:
            output[name + "_data"] = value

    return output


def _get_call_bound_arguments(term: ResponseTerm) -> dict:
    component = term.components[0]
    function = get_function_from_module(component.call.callee, component.env)
    bound = inspect.signature(function).bind(*component.call.args, **component.call.kwargs)
    parameters = list(dict(bound.arguments))
    arguments = CallVarsExtractor(component.call).get()
    return dict(zip(parameters, arguments))
