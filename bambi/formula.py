import keyword
import warnings
from typing import Sequence

import formulae as fm


class Formula:
    """Model formula

    Allows to describe a model with multiple formulas. The first formula describes the response
    variable and its predictors. The following formulas describe predictors for other parameters
    of the response distribution, allowing distributional models.

    Parameters
    ----------
    formula : str
        A model description written using the formula syntax from the `formulae` library.
    *additionals : tuple of str
        Additional formulas that describe model parameters rather than a response variable.
    nlpars : list or tuple of str, optional
        Names of parameters used in the nonlinear expression on the right-hand side of the main
        formula. An additional formula can describe how a nonlinear parameter varies. Parameters
        without an additional formula use an intercept-only formula. Additional formulas can also
        describe ordinary auxiliary likelihood parameters, such as `sigma ~ z`.
        The expression is on the parent parameter's link scale. The family's inverse link is
        applied once to the complete expression. An additional formula that references a modeled
        parameter is also treated as a nonlinear expression and evaluated after its dependencies;
        these dependent expressions define their parameter on the response scale. Separately
        modeled nonlinear predictors use identity links.

    Examples
    --------
    Model an exponential decay with three separately modeled parameters:

    >>> Formula("y ~ a + b * exp(-k * x)", "a ~ 1 + z", nlpars=("a", "b", "k"))
    Formula('y ~ a + b * exp(-k * x)', 'a ~ 1 + z', nlpars=('a', 'b', 'k'))
    """

    def __init__(
        self, formula: str, *additionals: str, nlpars: list[str] | tuple[str, ...] | None = None
    ):
        self.nlpars = self._check_nlpars(nlpars)
        self.additionals_lhs = []
        self.main = formula
        self.additionals = self.check_additionals(additionals)

        if self.nlpars:
            duplicates = {
                name for name in self.additionals_lhs if self.additionals_lhs.count(name) > 1
            }
            if duplicates:
                raise ValueError(f"Duplicate parameter formula(s): {sorted(duplicates)}.")

    @staticmethod
    def _check_nlpars(
        nlpars: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """Validate and normalize nonlinear parameter names."""
        if nlpars is None:
            return ()
        if not isinstance(nlpars, (list, tuple)):
            raise TypeError("'nlpars' must be a list or tuple of strings.")

        invalid = [
            name
            for name in nlpars
            if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name)
        ]
        if invalid:
            raise ValueError(f"'nlpars' entries must be valid Python identifiers: {invalid}.")

        duplicates = {name for name in nlpars if nlpars.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate nonlinear parameter name(s): {sorted(duplicates)}.")
        return tuple(nlpars)

    def check_additionals(self, additionals: Sequence[str]):
        """Check if the additional formulas match the expected format

        Parameters
        ----------
        additionals : sequence of str
            Model formulas that describe model parameters rather than a response variable.

        Returns
        -------
        additionals : sequence of str
            If all formulas match the required format, it returns them.
        """
        for additional in additionals:
            self.check_additional(additional)
        return additionals

    def check_additional(self, additional: str):
        """Check if an additional formula matches the expected format

        Parameters
        ----------
        additional : str
            A model formula that describes a model parameter.

        Raises
        ------
        ValueError
            If the formula does not contain a response term.
        ValueError
            If the response term is not a plain name.
        """
        response = fm.model_description(additional).response

        # There's a response in the formula
        if response is None:
            raise ValueError("Additional formulas must contain a response name.")

        # The response is a name, not a function call, for example
        if not isinstance(response.term.components[0], fm.terms.variable.Variable):
            raise ValueError("The response must be a name")

        self.additionals_lhs.append(response.term.name)

    def get_all_formulas(self):
        """Get all the model formulas

        Returns
        -------
        list of str
            All the formulas in the instance.
        """
        return [self.main] + list(self.additionals)

    def __str__(self):
        formulas = [self.main] + list(self.additionals)
        middle = ", ".join(formulas)
        if self.nlpars:
            middle += f", nlpars={self.nlpars!r}"
        return f"Formula({middle})"

    def __repr__(self):
        formulas = [self.main] + list(self.additionals)
        middle = ", ".join([f"'{formula}'" for formula in formulas])
        if self.nlpars:
            middle += f", nlpars={self.nlpars!r}"
        return f"Formula({middle})"


def formula_has_intercept(formula: str) -> bool:
    """Determines if a model formula describes a model with an intercept."""
    description = fm.model_description(formula)
    return any(isinstance(term, fm.terms.Intercept) for term in description.terms)


def check_ordinal_formula(formula: Formula) -> Formula:
    """Check if a supplied formula can be used with an ordinal model.

    Ordinal models have the following constraints (for the moment):

    - A single formula must be passed. This is because Bambi does not support modeling the
    thresholds as a function of predictors.
    - The intercept is omitted. This is to avoid non-identifiability issues between the intercept
    and the thresholds.
    """
    if len(formula.additionals) > 0:
        raise ValueError("Ordinal families don't accept multiple formulas")
    if formula_has_intercept(formula.main):
        warnings.warn("The intercept is omitted in ordinal families")
    return formula
