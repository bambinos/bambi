from typing import Sequence

import ast
import textwrap

import formulae as fm
import numpy as np
from xarray import DataTree

from bambi.transformations import HSGP, SmoothTransform


def listify(obj):
    """Wrap all non-list or tuple objects in a list

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def indentify(string: str, n: int = 2) -> str:
    """Add spaces to the beginning of each line in a multi-line string."""
    space = n * " "
    return space + space.join(string.splitlines(True))


def multilinify(sequence: Sequence[str], sep: str = ",") -> str:
    """Make a multi-line string out of a sequence of strings."""
    sep += "\n"
    return "\n" + sep.join(sequence)


def wrapify(string: str, width: int = 100, indentation: int = 2) -> str:
    """Wraps long strings into multiple lines

    This function is used to print the model summary.
    """
    lines = string.splitlines(True)
    wrapper = textwrap.TextWrapper(width=width)
    for idx, line in enumerate(lines):
        if len(line) > width:
            leading_spaces = len(line) - len(line.lstrip(" "))
            wrapper.subsequent_indent = " " * (leading_spaces + indentation)
            wrapped = wrapper.wrap(line)
            lines[idx] = "\n".join(wrapped) + "\n"
    return "".join(lines)


def extract_argument_names(expr, accepted_funcs):
    """Extract the names of the arguments passed to a function

    This is used to extract the labels from function calls such as `c(y1, y2, y3, y3)`.

    Parameters
    ----------
    expr : str
        An expression that is parsed to extract the components of the call.
    accepted_funcs : list
        A list with the names of the functions that we accept to parse.

    Returns
    -------
    list
        If all criteria are met, the names of the arguments. Otherwise it returns `None`.
    """
    # Extract the first thing in the body
    parsed_expr = ast.parse(expr).body[0]
    if not isinstance(parsed_expr, ast.Expr):
        return None

    # Check the value is a call
    value = parsed_expr.value
    if not isinstance(value, ast.Call):
        return None

    # Check call name is the name of an exepcted function
    if value.func.id not in accepted_funcs:
        return None

    # Check all arguments are either names or constants
    args = value.args
    if not all(isinstance(arg, ast.Name) for arg in args):
        return None

    # We can safely build labels now
    labels = [arg.id for arg in args]

    if labels:
        return labels
    return None


def clean_formula_lhs(x):
    """Remove the left hand side of a model formula and the tilde

    Parameters
    ----------
    x : str
        A model formula that has '~' in it.

    Returns
    -------
    str
        The right hand side of the model formula.
    """
    assert "~" in x
    position = x.find("~")
    return x[position + 1 :]


def is_call_component(component) -> bool:
    """Determines if formulae component is the result of a function call."""
    return isinstance(component, fm.terms.call.Call)


def is_smooth_term(term):
    """Determine whether a formulae term is a smooth transform."""
    return any(
        is_call_component(component)
        and isinstance(component.call.stateful_transform, SmoothTransform)
        for component in getattr(term, "components", ())
    )


def is_hsgp_term(term):
    """Determine whether a formulae term is an HSGP transform."""
    return any(
        is_call_component(component) and isinstance(component.call.stateful_transform, HSGP)
        for component in getattr(term, "components", ())
    )


def remove_common_intercept(dm: fm.matrices.DesignMatrices) -> fm.matrices.DesignMatrices:
    """Removes the intercept from the common design matrix

    This function is used with families for ordinal responses,
    where the intercept is requested but not used because its inclusion,
    together with the cutpoints, would create a non-identifiability problem.
    """
    dm.common.terms.pop("Intercept")
    intercept_slice = dm.common.slices.pop("Intercept")
    dm.common.design_matrix = np.delete(dm.common.design_matrix, intercept_slice, axis=1)
    return dm


def as_dataset(group):
    """Return a Dataset from either a DataTree group or a Dataset."""
    return group.to_dataset() if isinstance(group, DataTree) else group
