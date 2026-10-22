from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Collection, Sequence

import formulae as fm
from formulae.parser import ParseError
from formulae.scanner import ScanError, Scanner

from bambi.parameters import Conditional, ConditionalCoefficient, Marginal, MarginalCoefficient

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
        Variable names referenced by the expression, excluding function names.
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
            _PredictorScanner(source).scan(add_intercept=False)
            parsed = ast.parse(source, mode="eval")
        except (SyntaxError, ScanError) as error:
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
    """Formula tokens with decimal numeric literals and underscore-prefixed names."""

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
        if self.peek().isalnum() or self.peek() == "_":
            self.identifier()
            name = self.code[self.start : self.current]
            raise ValueError(f"Invalid variable name: {name!r}.")
        self.add_token("NUMBER", float(match.group()))

    floatnum = number


def has_nl_wrapper(source: str) -> bool:
    """Whether a formula contains an explicit additive nonlinear contribution."""
    tokens = _PredictorScanner(source).scan(add_intercept=False)
    return any(
        token.lexeme == "nl" and following.kind == "LEFT_PAREN"
        for token, following in zip(tokens, tokens[1:])
    )


def split_predictor(
    source: str, parameters: Collection[str]
) -> tuple[str | None, NonlinearExpression | None]:
    """Separate ordinary formula terms from nonlinear expressions.

    Ordinary terms are returned as a formula string for Formulae to build design matrices.
    Terms referencing names in `parameters` are automatically parsed as mathematical
    expressions, without requiring `nl(...)`. All such terms are combined into one expression
    evaluated using the values of those parameters and the observed data.

    The optional `nl(expr)` wrapper includes an entire expression in the nonlinear part,
    even terms containing only data or constants. Outside `nl`, parentheses retain ordinary
    formula grouping semantics. If an unwrapped predictor consists entirely of
    parameter-dependent expressions and constants, it has no linear formula part.

    Examples
    --------
    Separate an ordinary treatment effect from two parameter-dependent terms without a wrapper:

    >>> linear_formula, nlexpr = split_predictor(
    ...     "trt + x ** a + y ** b", ("a", "b")
    ... )
    >>> linear_formula
    'trt'
    >>> sorted(nlexpr.symbols)
    ['a', 'b', 'x', 'y']

    Use the optional wrapper to include a data-only expression in the nonlinear part:

    >>> linear_formula, nlexpr = split_predictor("trt + nl(x + 1)", ())
    >>> linear_formula
    'trt'
    >>> sorted(nlexpr.symbols)
    ['x']
    """
    tokens = _PredictorScanner(source).scan(add_intercept=False)[:-1]
    linear_terms = []
    nl_terms = []
    explicit = False
    only_constants = True
    parameter_names = set(parameters)

    def collect(part, sign=1):
        nonlocal explicit, only_constants
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

        text = " ".join(str(token.lexeme) for token in part)
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
                first_linear = len(linear_terms)
                collect(part[1:-1], sign)
                grouped_terms = linear_terms[first_linear:]
                if grouped_terms and not (len(grouped_terms) == 1 and grouped_terms[0][0] == sign):
                    # Formula subtraction removes terms rather than distributing signs.
                    grouped = " ".join(
                        ("+" if term_sign == sign else "-") + f" {term}"
                        for term_sign, term in grouped_terms
                    ).lstrip("+ ")
                    linear_terms[first_linear:] = [(sign, f"({grouped})")]
                return

        if any(
            token.lexeme == "nl" and following.kind == "LEFT_PAREN"
            for token, following in zip(part, part[1:])
        ):
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
            explicit = True
            nl_terms.append((sign, text))
        elif nonlinear_symbol_names(text) & parameter_names:
            nl_terms.append((sign, text))
        else:
            only_constants &= len(part) == 1 and part[0].kind == "NUMBER"
            linear_terms.append((sign, text))

    collect(tokens)
    if not nl_terms:
        return source, None

    if not explicit and only_constants:
        return None, NonlinearExpression.parse(source.strip())

    nlexpr_source = " ".join(("+" if sign > 0 else "-") + f" ({text})" for sign, text in nl_terms)

    linear_formula = " ".join(
        ("+" if sign > 0 else "-") + f" {text}" for sign, text in linear_terms
    ).lstrip("+ ")

    nlexpr = NonlinearExpression.parse(nlexpr_source)
    return linear_formula or "1", nlexpr


@dataclass(frozen=True)
class ParameterDependencyGraph:
    """Modeled quantities and their parameter dependencies.

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
        """Return nonlinear coefficient nodes keyed by their original names.

        Returns
        -------
        dict of str to ConditionalCoefficient or MarginalCoefficient
            Coefficients keyed by their original names.

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
        return {
            name: quantity
            for name, quantity in self.nodes.items()
            if isinstance(quantity, (ConditionalCoefficient, MarginalCoefficient))
        }


def parameter_dependency_order(dependencies, declaration_order: Sequence[str]) -> tuple[str, ...]:
    """Return a deterministic topological order for modeled parameters.

    Parameters
    ----------
    dependencies : Mapping of str to tuple of str
        Direct dependencies for every modeled parameter node. All referenced nodes must be keys.
    declaration_order : Sequence of str
        Every node exactly once, in preferred order between otherwise independent nodes.

    Returns
    -------
    tuple of str
        Parameter names with every dependency preceding its dependent.

    Raises
    ------
    ValueError
        If a node references itself or the graph contains a cycle.

    Examples
    --------
    >>> parameter_dependency_order({"mu": (), "sigma": ("mu",)}, ("mu", "sigma"))
    ('mu', 'sigma')
    """
    self_dependencies = sorted(name for name, values in dependencies.items() if name in values)
    if self_dependencies:
        raise ValueError(
            "Nonlinear parameters cannot depend on themselves: " f"{self_dependencies}."
        )

    rank = {name: index for index, name in enumerate(declaration_order)}
    state = {name: 0 for name in dependencies}
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

    for name in declaration_order:
        visit(name)

    return tuple(order)


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
