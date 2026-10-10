from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import pandas as pd
import formulae as fm
from formulae.parser import ParseError
from formulae.scanner import ScanError, Scanner

from bambi.parameters import Conditional, Marginal

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
    """Return variable names from an arithmetic expression or an additive formula."""
    source = source.strip()
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError as error:
        try:
            return frozenset(fm.model_description(source).var_names)
        except (ParseError, ScanError):
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


class _PredictorScanner(Scanner):
    """Formula tokens with Python numeric literals and underscore-prefixed names."""

    def scan_token(self):
        if self.peek() == "_":
            self.advance()
            self.identifier()
        else:
            super().scan_token()

    def number(self):
        match = re.match(
            r"(?:\d[\d_]*(?:\.[\d_]*)?|\.[\d_]+)(?:[eE][+-]?[\d_]+)?",
            self.code[self.start :],
        )
        self.current = self.start + match.end()
        self.add_token("NUMBER", float(match.group()))

    floatnum = number


def has_nl_wrapper(source: str) -> bool:
    """Whether a formula contains an explicit additive nonlinear contribution."""
    tokens = _PredictorScanner(source).scan(add_intercept=False)
    return any(
        token.lexeme == "nl" and following.kind == "LEFT_PAREN"
        for token, following in zip(tokens, tokens[1:])
    )


def split_predictor(source: str, parameter_names) -> tuple[str | None, NonlinearExpression | None]:
    """Separate additive formula terms from arithmetic contributions.

    Terms referencing modeled quantities are arithmetic. ``nl(expr)`` explicitly keeps an
    arithmetic expression together, including data-only terms and literal constants. Outside
    the wrapper, parentheses retain ordinary formula grouping semantics. Expression-only
    formulas without a wrapper retain their original arithmetic interpretation.
    """
    tokens = _PredictorScanner(source).scan(add_intercept=False)[:-1]
    additive = []
    nonlinear = []
    explicit = has_nl_wrapper(source)
    only_constants = True
    parameter_names = set(parameter_names)

    def collect(part, sign=1):
        nonlocal only_constants
        depth = 0
        start = 0
        current_sign = sign
        for index, token in enumerate(part):
            if token.kind in ("LEFT_PAREN", "LEFT_BRACKET", "LEFT_BRACE"):
                depth += 1
            elif token.kind in ("RIGHT_PAREN", "RIGHT_BRACKET", "RIGHT_BRACE"):
                depth -= 1
            elif (
                depth == 0
                and token.kind in ("PLUS", "MINUS")
                and index > start
                and part[index - 1].kind
                not in ("PLUS", "MINUS", "STAR", "SLASH", "STAR_STAR", "COLON", "PIPE")
            ):
                collect(part[start:index], current_sign)
                current_sign = sign if token.kind == "PLUS" else -sign
                start = index + 1
        if start:
            collect(part[start:], current_sign)
            return
        if not part:
            raise ValueError(f"Malformed nonlinear expression: {source!r}.")
        if part[0].kind in ("PLUS", "MINUS"):
            collect(part[1:], sign if part[0].kind == "PLUS" else -sign)
            return
        if part[0].kind == "LEFT_PAREN" and part[-1].kind == "RIGHT_PAREN":
            depth = 0
            enclosed = True
            group_specific = False
            for token in part[:-1]:
                depth += token.kind == "LEFT_PAREN"
                depth -= token.kind == "RIGHT_PAREN"
                group_specific |= token.kind == "PIPE" and depth == 1
                if depth == 0:
                    enclosed = False
                    break
            if enclosed and not group_specific:
                collect(part[1:-1], sign)
                return
        text = " ".join(str(token.lexeme) for token in part)
        if has_nl_wrapper(text):
            try:
                node = ast.parse(text, mode="eval").body
            except SyntaxError as error:
                raise ValueError("'nl(...)' must be a separate additive term.") from error
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "nl"
                and len(node.args) == 1
                and not node.keywords
            ):
                raise ValueError("'nl(...)' must be a separate additive term with one argument.")
            text = ast.unparse(node.args[0])
            nonlinear.append((sign, text))
        elif nonlinear_symbol_names(text) & parameter_names:
            nonlinear.append((sign, text))
        else:
            only_constants &= len(part) == 1 and part[0].kind == "NUMBER"
            additive.append((sign, text))

    collect(tokens)
    if not nonlinear:
        return source, None
    if not explicit and only_constants:
        return None, NonlinearExpression.parse(source.strip())
    expression_source = " ".join(
        ("+" if sign > 0 else "-") + f" ({text})" for sign, text in nonlinear
    )
    additive_source = " ".join(
        ("+" if sign > 0 else "-") + f" {text}" for sign, text in additive
    ).lstrip("+ ")
    return additive_source or "1", NonlinearExpression.parse(expression_source)


@dataclass(frozen=True)
class ParameterDependencyGraph:
    """Dependency metadata for a nonlinear model's parameter-level expressions.

    Attributes
    ----------
    nodes : dict of str to Conditional or Marginal
        Observational-model parameters and nonlinear coefficients, keyed by original names.
    dependencies : dict of str to tuple of str
        Direct parameter dependencies for every node.
    order : tuple of str
        Deterministic topological order in which to evaluate the nodes.
    """

    nodes: dict[str, Conditional | Marginal]
    dependencies: dict[str, tuple[str, ...]]
    order: tuple[str, ...]

    @property
    def expression_nodes(self):
        """Conditional quantities defined by expressions."""
        return {
            name: quantity
            for name, quantity in self.nodes.items()
            if isinstance(quantity, Conditional) and quantity.is_nonlinear
        }

    @property
    def nonlinear_coefficients(self):
        """Return the canonical coefficients owned by the graph's parameter nodes.

        Returns
        -------
        dict of str to ConditionalCoefficient or MarginalCoefficient
            Coefficients keyed by their original names.

        Raises
        ------
        ValueError
            If multiple parameter nodes define different coefficients with the same name.

        Examples
        --------
        The graph exposes one canonical object for each nonlinear coefficient.

        >>> import bambi as bmb
        >>> import pandas as pd
        >>> data = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]})
        >>> model = bmb.Model(bmb.Formula("y ~ a * x", nlpars=("a",)), data)
        >>> set(model.parameter_graph.nonlinear_coefficients)
        {'a'}
        """
        coefficients = {}
        for parameter in self.expression_nodes.values():
            for name, coefficient in parameter.nonlinear_coefficients.items():
                if name in coefficients and coefficients[name] is not coefficient:
                    raise ValueError(f"Nonlinear coefficient '{name}' has multiple definitions.")
                coefficients[name] = coefficient
        return coefficients


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
    formula,
    expressions,
    data,
    dropna,
    include_response=True,
    parameter_names=(),
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
    response_formula, parent_rhs = split_nonlinear_formula(formula.main)
    additive_rhs, _ = split_predictor(parent_rhs, parameter_names)
    if additive_rhs is not None:
        variables.update(set(fm.model_description(additive_rhs).var_names) - parameter_names)
    if include_response:
        variables.update(fm.model_description(response_formula).var_names)
    for predictor_formula in formula.additionals:
        rhs, _ = split_predictor(predictor_formula.partition("~")[2], parameter_names)
        if rhs is None:
            continue
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
