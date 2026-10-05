from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.utils import INVERSE_LINKS
from bambi.families import Family
from bambi.nonlinear import (
    BinaryOperation,
    FUNCTION_ALIASES,
    FunctionCall,
    Literal,
    SUPPORTED_FUNCTIONS,
    Symbol,
    UnaryOperation,
)

if TYPE_CHECKING:
    from bambi.parameters import ConditionalParameter

_BINARY_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
}


def _logit(value):
    return pt.log(value) - pt.log1p(-value)


def _normal_cdf(value):
    return 0.5 + 0.5 * pt.erf(value / pt.sqrt(2))


def _normal_ppf(value):
    return pt.sqrt(2) * pt.erfinv(2 * value - 1)


def _cloglog(value):
    return pt.log(-pt.log1p(-value))


def _invcloglog(value):
    return -pt.expm1(-pt.exp(value))


_CUSTOM_FUNCTIONS = {
    "logit": _logit,
    "normal_cdf": _normal_cdf,
    "norm_cdf": _normal_cdf,
    "normal_ppf": _normal_ppf,
    "norm_ppf": _normal_ppf,
    "probit": _normal_ppf,
    "invprobit": _normal_cdf,
    "cloglog": _cloglog,
    "invcloglog": _invcloglog,
}

_FUNCTIONS = {
    name: getattr(pt, FUNCTION_ALIASES.get(name, name))
    for name in SUPPORTED_FUNCTIONS
    if name not in _CUSTOM_FUNCTIONS
}
_FUNCTIONS.update(_CUSTOM_FUNCTIONS)


def nonlinear_data_name(parameter_label: str, symbol: str) -> str:
    """Return the backend data variable name for an expression input."""
    return f"{parameter_label}__{symbol}_data"


def build_nonlinear_parameter(
    parameter: ConditionalParameter,
    parameter_values: dict[str, pt.Variable],
    data,
    model: pm.Model,
    family: Family,
    parameters: dict[str, pt.Variable],
) -> pt.Variable:
    """Build a nonlinear likelihood parameter in a PyMC model.

    Parameters
    ----------
    parameter : ConditionalParameter
        Frontend description of the nonlinear parameter.
    parameter_values : dict of str to TensorVariable
        Already-built additive and nonlinear parameters keyed by their original names.
    data : pandas.DataFrame
        Observed data containing the expression inputs.
    model : pymc.Model
        Model in which to create data and deterministic variables.
    family : Family
        Family providing the parent link and predictor transformation.
    parameters : dict of str to TensorVariable
        Other built likelihood parameters used by predictor transformations.

    Returns
    -------
    TensorVariable
        Deterministic parent parameter on the response scale.
    """
    values = parameter_values.copy()
    for name in parameter.data_names:
        values[name] = pm.Data(
            nonlinear_data_name(parameter.label, name),
            np.asarray(data[name], dtype=float),
            dims="__obs__",
            model=model,
        )

    value = evaluate_expression(parameter.expression.root, values)
    if parameter.is_parent:
        link = family.link[parameter.name]
        inverse_link = INVERSE_LINKS.get(link.name, link.inverse_link)
        transform_predictor = transforms_registry.get_predictor_transform(family, parameter.name)
        if transform_predictor:
            value = transform_predictor(value, parameters, inverse_link)
        else:
            value = inverse_link(value)
    value = pt.as_tensor_variable(value)
    if any(value is variable for variable in model.deterministics):
        # Keep the parent distinct when PyMC clones direct deterministic views.
        value = value.copy()
    if value.ndim == 0:
        value = pt.broadcast_to(value, (model.dim_lengths["__obs__"],))
    return pm.Deterministic(parameter.label, value, dims="__obs__", model=model)


def build_new_nonlinear_data(parameter: ConditionalParameter, data) -> dict[str, np.ndarray]:
    """Build replacements for observed inputs to a nonlinear expression.

    Parameters
    ----------
    parameter : ConditionalParameter
        Nonlinear parameter whose expression inputs are required.
    data : pandas.DataFrame
        New prediction or log-likelihood data.

    Returns
    -------
    dict of str to numpy.ndarray
        Backend data variable names mapped to new values.

    Raises
    ------
    ValueError
        If a required expression input is missing.
    """
    missing = set(parameter.data_names) - set(data.columns)
    if missing:
        raise ValueError(f"New data is missing nonlinear expression column(s): {sorted(missing)}.")
    return {
        nonlinear_data_name(parameter.label, name): np.asarray(data[name], dtype=float)
        for name in parameter.data_names
    }


def evaluate_expression(node, values):
    """Evaluate an expression tree with PyTensor operations.

    Parameters
    ----------
    node : ExpressionNode
        Root of the expression tree to evaluate.
    values : Mapping
        Expression symbols mapped to PyTensor variables.

    Returns
    -------
    TensorVariable
        Evaluated expression.

    Raises
    ------
    TypeError
        If the tree contains an unknown node type.
    """
    if isinstance(node, Literal):
        return pt.as_tensor_variable(node.value)
    if isinstance(node, Symbol):
        return values[node.name]
    if isinstance(node, UnaryOperation):
        value = evaluate_expression(node.operand, values)
        return value if node.operator == "+" else -value
    if isinstance(node, BinaryOperation):
        left = evaluate_expression(node.left, values)
        right = evaluate_expression(node.right, values)
        return _BINARY_OPERATORS[node.operator](left, right)
    if isinstance(node, FunctionCall):
        arguments = [evaluate_expression(argument, values) for argument in node.arguments]
        return _FUNCTIONS[node.function](*arguments)
    raise TypeError(f"Unexpected nonlinear expression node: {type(node).__name__}.")
