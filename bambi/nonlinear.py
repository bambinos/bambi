import ast
from dataclasses import dataclass

import pandas as pd
import formulae as fm

FUNCTION_ARITIES = {
    "exp": 1,
    "log": 1,
    "sqrt": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "asin": 1,
    "acos": 1,
    "atan": 1,
    "arcsin": 1,
    "arccos": 1,
    "arctan": 1,
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "asinh": 1,
    "acosh": 1,
    "atanh": 1,
    "arcsinh": 1,
    "arccosh": 1,
    "arctanh": 1,
    "atan2": 2,
    "arctan2": 2,
    "log1p": 1,
    "expm1": 1,
    "softplus": 1,
    "erf": 1,
    "erfc": 1,
    "logit": 1,
    "invlogit": 1,
    "expit": 1,
    "normal_cdf": 1,
    "norm_cdf": 1,
    "normal_ppf": 1,
    "norm_ppf": 1,
    "probit": 1,
    "invprobit": 1,
    "cloglog": 1,
    "invcloglog": 1,
}

FUNCTION_ALIASES = {
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "asinh": "arcsinh",
    "acosh": "arccosh",
    "atanh": "arctanh",
    "atan2": "arctan2",
    "invlogit": "sigmoid",
    "expit": "sigmoid",
}

SUPPORTED_FUNCTIONS = frozenset(FUNCTION_ARITIES)


class ExpressionNode:
    """Base class for nodes in a nonlinear expression tree."""


@dataclass(frozen=True)
class Literal(ExpressionNode):
    """A numeric literal in a nonlinear expression."""

    value: int | float


@dataclass(frozen=True)
class Symbol(ExpressionNode):
    """A nonlinear parameter or observed data name."""

    name: str


@dataclass(frozen=True)
class UnaryOperation(ExpressionNode):
    """A unary arithmetic operation."""

    operator: str
    operand: ExpressionNode


@dataclass(frozen=True)
class BinaryOperation(ExpressionNode):
    """A binary arithmetic operation."""

    operator: str
    left: ExpressionNode
    right: ExpressionNode


@dataclass(frozen=True)
class FunctionCall(ExpressionNode):
    """A call to a supported function."""

    function: str
    arguments: tuple[ExpressionNode, ...]


@dataclass(frozen=True)
class NonlinearExpression:
    """A parsed nonlinear expression.

    Attributes
    ----------
    source : str
        Original expression source.
    root : ExpressionNode
        Root of the parsed expression tree.
    symbols : frozenset of str
        Nonlinear parameter and observed data names used by the expression.
    """

    source: str
    root: ExpressionNode
    symbols: frozenset[str]

    @classmethod
    def parse(cls, source: str) -> "NonlinearExpression":
        """Parse a nonlinear expression from its source.

        Parameters
        ----------
        source : str
            Expression using supported arithmetic operators and functions.

        Returns
        -------
        NonlinearExpression
            Parsed expression and the symbols it references.

        Raises
        ------
        ValueError
            If the expression is malformed or contains unsupported syntax.
        """
        try:
            parsed = ast.parse(source, mode="eval")
        except SyntaxError as error:
            raise ValueError(f"Malformed nonlinear expression: {source!r}.") from error

        symbols = set()
        root = _convert_node(parsed.body, symbols)
        return cls(source=source, root=root, symbols=frozenset(symbols))


def nonlinear_symbol_names(source: str) -> frozenset[str]:
    """Return variable names from an expression without validating its operations."""
    source = source.strip()
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Malformed nonlinear expression: {source!r}.") from error

    function_names = {
        node.func.id
        for node in ast.walk(parsed)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return (
        frozenset(node.id for node in ast.walk(parsed) if isinstance(node, ast.Name))
        - function_names
    )


@dataclass
class NonlinearParameter:
    """Description of a modeled parameter defined by a nonlinear expression.

    Attributes
    ----------
    name : str
        Original modeled parameter name.
    expression : NonlinearExpression
        Expression that defines the parameter. The parent expression is on its link scale;
        other parameter expressions are on the response scale.
    data_names : tuple of str
        Observed data columns referenced directly by the expression.
    alias : str or None
        Name used in the backend graph and posterior output.
    is_parent : bool
        Whether this is the likelihood's parent parameter, whose expression is on the link scale.
    """

    name: str
    expression: NonlinearExpression
    data_names: tuple[str, ...]
    alias: str | None = None
    is_parent: bool = True

    @property
    def label(self):
        """Return the aliased name when present, otherwise the original name."""
        return self.alias or self.name


@dataclass(frozen=True)
class ParameterDependencyGraph:
    """Dependency metadata for a nonlinear model's parameter-level expressions.

    Attributes
    ----------
    nodes : dict of str to NonlinearParameter
        Parameters defined by nonlinear expressions, keyed by their original names.
    dependencies : dict of str to tuple of str
        Direct parameter dependencies for every node.
    order : tuple of str
        Deterministic topological order in which to evaluate the nodes.
    """

    nodes: dict[str, NonlinearParameter]
    dependencies: dict[str, tuple[str, ...]]
    order: tuple[str, ...]


def split_nonlinear_formula(formula: str) -> tuple[str, str]:
    """Separate a nonlinear formula into its response and expression.

    Parameters
    ----------
    formula : str
        Formula in the form ``response ~ expression``.

    Returns
    -------
    response_formula : str
        Intercept-only formula used to build the response design.
    expression : str
        Nonlinear expression from the right-hand side.

    Raises
    ------
    ValueError
        If the formula does not contain one response and one expression.
    """
    lhs, separator, rhs = formula.partition("~")
    if not separator or not lhs.strip() or not rhs.strip() or "~" in rhs:
        raise ValueError("A nonlinear formula must have the form 'response ~ expression'.")
    return f"{lhs.strip()} ~ 1", rhs.strip()


def resolve_nonlinear_symbols(expression, parameter_names, data):
    """Partition expression symbols into modeled parameters and observed data columns.

    Parameters
    ----------
    expression : NonlinearExpression
        Parsed nonlinear expression.
    parameter_names : Collection of str
        Names available as modeled parameters.
    data : pandas.DataFrame
        Model data containing observed expression inputs.

    Returns
    -------
    dependencies : tuple of str
        Modeled parameters referenced directly by the expression.
    data_names : tuple of str
        Observed data columns referenced directly by the expression.

    Raises
    ------
    ValueError
        If a symbol is ambiguous, unresolved, or names non-numeric data.

    Examples
    --------
    >>> expression = NonlinearExpression.parse("sigma_y + attempts")
    >>> data = pd.DataFrame({"attempts": [10]})
    >>> resolve_nonlinear_symbols(expression, ("sigma_y",), data)
    (('sigma_y',), ('attempts',))
    """
    parameter_names = set(parameter_names)
    data_columns = set(data.columns)
    collisions = expression.symbols & parameter_names & data_columns
    if collisions:
        raise ValueError(
            "Nonlinear expression symbols must not be both modeled parameters and data columns: "
            f"{sorted(collisions)}."
        )

    dependencies = expression.symbols & parameter_names
    data_names = expression.symbols - dependencies
    unknown = data_names - data_columns
    if unknown:
        raise ValueError(
            "No nonlinear parameter formula or data column was found for symbol(s): "
            f"{sorted(unknown)}."
        )

    nonnumeric = [
        name for name in sorted(data_names) if not pd.api.types.is_numeric_dtype(data[name])
    ]
    if nonnumeric:
        raise ValueError(
            f"Nonlinear expression data must be numeric. Invalid column(s): {nonnumeric}."
        )
    return tuple(sorted(dependencies)), tuple(sorted(data_names))


def parameter_dependency_order(dependencies, declaration_order=None) -> tuple[str, ...]:
    """Return a deterministic topological order for modeled parameters.

    Parameters
    ----------
    dependencies : Mapping of str to Collection of str
        Direct dependencies for every modeled parameter node.
    declaration_order : Collection of str, optional
        Preferred order between otherwise independent nodes.

    Returns
    -------
    tuple of str
        Parameter names with every dependency preceding its dependent.

    Raises
    ------
    ValueError
        If a dependency is unknown, a node references itself, or the graph contains a cycle.

    Examples
    --------
    >>> parameter_dependency_order({"mu": (), "sigma": ("mu",)})
    ('mu', 'sigma')
    """
    dependencies = {name: tuple(values) for name, values in dependencies.items()}
    names = set(dependencies)
    unknown = {value for values in dependencies.values() for value in values if value not in names}
    if unknown:
        raise ValueError(f"Unknown nonlinear parameter reference(s): {sorted(unknown)}.")

    self_dependencies = sorted(name for name, values in dependencies.items() if name in values)
    if self_dependencies:
        raise ValueError(
            "Nonlinear parameters cannot depend on themselves: " f"{self_dependencies}."
        )

    preferred = list(dict.fromkeys(declaration_order or ()))
    preferred.extend(sorted(names - set(preferred)))
    rank = {name: index for index, name in enumerate(preferred)}
    state = {name: 0 for name in names}
    order = []
    path = []

    def visit(name):
        if state[name] == 2:
            return
        if state[name] == 1:
            start = path.index(name)
            cycle = path[start:] + [name]
            raise ValueError(
                "Cycle detected between nonlinear parameters: " + " -> ".join(cycle) + "."
            )

        state[name] = 1
        path.append(name)
        for dependency in sorted(dependencies[name], key=rank.__getitem__):
            visit(dependency)
        path.pop()
        state[name] = 2
        order.append(name)

    for name in sorted(names, key=rank.__getitem__):
        visit(name)
    return tuple(order)


def prepare_nonlinear_data(
    formula, expressions, data, dropna, include_response=True, parameter_names=()
):
    """Prepare aligned, complete observations for every part of a nonlinear model.

    Parameters
    ----------
    formula : Formula
        Nonlinear model formula and its parameter formulas.
    expressions : Mapping of str to NonlinearExpression or NonlinearExpression
        Parsed expressions keyed by the parameter they define. A single expression is accepted
        for backwards compatibility.
    data : pandas.DataFrame
        Model or prediction data.
    dropna : bool
        Whether to remove incomplete rows instead of raising an error.
    include_response : bool
        Whether the response is required in ``data``.
    parameter_names : Collection of str
        Names of modeled parameters, which are excluded from required data columns.

    Returns
    -------
    pandas.DataFrame
        Data with a shared complete-row mask applied when requested.

    Raises
    ------
    ValueError
        If required data are incomplete.
    """
    parameter_names = set(parameter_names) | set(formula.nlpars)
    if isinstance(expressions, NonlinearExpression):
        expressions = {"__parent__": expressions}
    variables = set()
    for expression in expressions.values():
        variables.update(expression.symbols - parameter_names)
    if include_response:
        response_formula, _ = split_nonlinear_formula(formula.main)
        variables.update(fm.model_description(response_formula).var_names)
    for name, predictor_formula in zip(formula.additionals_lhs, formula.additionals):
        if name in expressions:
            continue
        rhs = predictor_formula.partition("~")[2]
        predictor_variables = fm.model_description(rhs).var_names
        variables.update(set(predictor_variables) - parameter_names)

    columns = sorted(variables & set(data.columns))
    incomplete = data[columns].isna().any(axis=1)
    if incomplete.any():
        if not dropna:
            raise ValueError(f"'data' contains {incomplete.sum()} incomplete rows.")
        data = data.loc[~incomplete].copy()
    if len(data) == 0:
        raise ValueError("'data' does not contain any complete observation.")
    return data


_BINARY_OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
}

_UNARY_OPERATORS = {
    ast.UAdd: "+",
    ast.USub: "-",
}


def _convert_node(node: ast.AST, symbols: set[str]) -> ExpressionNode:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Nonlinear expressions only support numeric literals.")
        return Literal(node.value)

    if isinstance(node, ast.Name):
        symbols.add(node.id)
        return Symbol(node.id)

    if isinstance(node, ast.BinOp):
        operator = _BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(
                f"Unsupported operator '{type(node.op).__name__}' in nonlinear expression."
            )
        return BinaryOperation(
            operator,
            _convert_node(node.left, symbols),
            _convert_node(node.right, symbols),
        )

    if isinstance(node, ast.UnaryOp):
        operator = _UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(
                f"Unsupported operator '{type(node.op).__name__}' in nonlinear expression."
            )
        return UnaryOperation(operator, _convert_node(node.operand, symbols))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Nonlinear functions must be referenced by name.")
        if node.func.id not in SUPPORTED_FUNCTIONS:
            supported = ", ".join(sorted(SUPPORTED_FUNCTIONS))
            raise ValueError(
                f"Unsupported nonlinear function '{node.func.id}'. "
                f"Supported functions: {supported}."
            )
        arity = FUNCTION_ARITIES[node.func.id]
        if len(node.args) != arity or node.keywords:
            noun = "argument" if arity == 1 else "arguments"
            raise ValueError(
                f"Nonlinear function '{node.func.id}' requires exactly {arity} positional {noun}."
            )
        return FunctionCall(
            node.func.id, tuple(_convert_node(argument, symbols) for argument in node.args)
        )

    raise ValueError(f"Unsupported syntax '{type(node).__name__}' in nonlinear expression.")
