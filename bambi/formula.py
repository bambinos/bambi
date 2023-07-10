import warnings

from typing import Sequence

import formulae as fm


class Formula:
    """Model formula

    Allows to describe a model with multiple formulas. The first formula describes the response
    variable and its predictors. The following formulas describe predictors for other parameters
    of the likelihood function, allowing distributional models.

    Parameters
    ----------
    formula : str
        A model description written using the formula syntax from the ``formulae`` library.
    *additionals : str
        Additional formulas that describe the
    """

    def __init__(self, formula: str, *additionals: str):
        self.additionals_lhs = []
        self.main = formula
        self.additionals = self.check_additionals(additionals)

    def check_additionals(self, additionals: Sequence[str]):
        """Check if the additional formulas match the expected format

        Parameters
        ----------
        additionals : Sequence[str]
            Model formulas that describe model parameters rather than a response variable

        Returns
        -------
        additionals : Sequence[str]
            If all formulas match the required format, it return them.
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
            If the formula does not contain a response term
        ValueError
            If the response term is not a plain name
        """
        response = fm.model_description(additional).response

        # There's a response in the formula
        if response is None:
            raise ValueError("Additional formulas must contain a response name.")

        # The response is a name, not a function call for example
        if not isinstance(response.term.components[0], fm.terms.variable.Variable):
            raise ValueError("The response must be a name.")

        self.additionals_lhs.append(response.term.name)

    def get_all_formulas(self):
        """Get all the model formulas

        Returns
        -------
        list
            All the formulas in the instance
        """
        return [self.main] + list(self.additionals)

    def __str__(self):
        formulas = [self.main] + list(self.additionals)
        middle = ", ".join(formulas)
        return f"Formula({middle})"

    def __repr__(self):
        formulas = [self.main] + list(self.additionals)
        middle = ", ".join([f"'{formula}'" for formula in formulas])
        return f"Formula({middle})"


def formula_has_intercept(formula: str) -> bool:
    """Determines if a model formula results in a model with intercept."""
    description = fm.model_description(formula)
    return any(isinstance(term, fm.terms.Intercept) for term in description.terms)


def check_ordinal_formula(formula: Formula) -> Formula:
    """Check if a supplied formula can be used with an ordinal model.

    Ordinal models have the following constrains (for the moment):
    * A single formula must be passed. This is because Bambi does not support modeling the
    thresholds as a function of predictors.
    * The intercept is omitted. This is to avoid non-identifiability issues between the intercept
    and the thresholds.
    """
    if len(formula.additionals) > 0:
        raise ValueError("Ordinal families don't accept multiple formulas")
    if formula_has_intercept(formula.main):
        warnings.warn("The intercept is omitted in ordinal families")
    return formula
