import ast

import numpy as np


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def spacify(string, n=2):
    """Add spaces to the beginning of each line in a multi-line string."""
    space = n * " "
    return space + space.join(string.splitlines(True))


def multilinify(sequence, sep=","):
    """Make a multi-line string out of a sequence of strings."""
    sep += "\n"
    return "\n" + sep.join(sequence)


def c(*args):  # pylint: disable=invalid-name
    """Concatenate columns into a 2D NumPy Array"""
    return np.column_stack(args)


def extract_argument_names(expr, accepted_funcs):
    """Extract the names of the arguments passed to a function.

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
        If all criteria are met, the names of the arguments. Otherwise it returns None.
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


extra_namespace = {"c": c}
