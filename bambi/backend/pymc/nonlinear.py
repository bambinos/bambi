from __future__ import annotations

import operator

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.parameters import Conditional
from bambi.nonlinear import (
    BinaryOperation,
    FUNCTION_ALIASES,
    FunctionCall,
    Literal,
    SUPPORTED_FUNCTIONS,
    Symbol,
    UnaryOperation,
)

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
    return 0.5 * pt.erfc(-value / np.sqrt(2.0))


def _normal_ppf(value):
    return -np.sqrt(2.0) * pt.erfcinv(2 * value)


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


def build_nonlinear_predictor(
    parameter: Conditional,
    parameter_values: dict[str, pt.Variable],
    data,
    model: pm.Model,
    dims: tuple[str, ...] | None,
) -> tuple[pt.Variable, tuple[str, ...]]:
    """Build a nonlinear contribution on the predictor scale.

    Parameters
    ----------
    parameter : Conditional
        Frontend description of the nonlinear parameter.
    parameter_values : dict of str to TensorVariable
        Already-built additive and nonlinear parameters keyed by their original names.
    data : pandas.DataFrame
        Observed data containing the expression inputs.
    model : pymc.Model
        Model in which to create expression data variables.
    dims : tuple of str or None
        Predictor dimensions, or ``None`` to inherit dimensions from expression inputs.

    Returns
    -------
    TensorVariable
        Expression value before applying a parameter's inverse link.
    tuple of str
        Predictor dimensions, including the observation axis.
    """
    values = {
        name: parameter_values[name]
        for name in parameter.expression.symbols
        if name not in parameter.data_names
    }
    for name in parameter.data_names:
        values[name] = pm.Data(
            nonlinear_data_name(parameter.label, name),
            np.asarray(data[name], dtype=float),
            dims="__obs__",
            model=model,
        )

    if dims is None:
        input_dims = {
            tuple(dim for dim in model.named_vars_to_dims.get(value.name, ()) if dim != "__obs__")
            for value in values.values()
        } - {()}
        if len(input_dims) > 1:
            raise ValueError(
                f"Nonlinear coefficient '{parameter.name}' has incompatible input dimensions."
            )
        dims = ("__obs__",) + next(iter(input_dims), ())
    # Align observation and response vectors before elementwise arithmetic.
    if len(dims) > 1:
        response_dims = set(dims[1:])
        for name, value in values.items():
            if value.ndim != 1:
                continue
            value_dims = model.named_vars_to_dims.get(value.name, ())
            if "__obs__" in value_dims:
                values[name] = value[:, None]
            elif response_dims.intersection(value_dims):
                values[name] = value[None, :]
    value = evaluate_expression(parameter.expression.root, values)
    if value.ndim == 1 and len(dims) > 1:
        value = value[:, None]
    value = pt.broadcast_to(value, tuple(model.dim_lengths[dim] for dim in dims))
    return value, dims


def build_new_nonlinear_data(parameter: Conditional, data) -> dict[str, np.ndarray]:
    """Build replacements for observed inputs to a nonlinear expression.

    Parameters
    ----------
    parameter : Conditional
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
