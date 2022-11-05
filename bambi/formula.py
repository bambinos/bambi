from typing import Sequence

from formulae import model_description
from formulae.terms.variable import Variable


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
        self.additional_formulas_lhs = []
        self.formula = formula
        self.additional_formulas = self.check_additionals(additionals)

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
        """Check if an additional formula match the expected format

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
        response = model_description(additional).response

        # There's a response in the formula
        if response is None:
            raise ValueError("Additional formulas must contain a response name.")

        # The response is a name, not a function call for example
        if not isinstance(response.term.components[0], Variable):
            raise ValueError("The response must be a name.")

        self.additional_formulas_lhs.append(response.term.name)

    def __str__(self):
        formulas = [self.formula] + list(self.additional_formulas)
        middle = ", ".join(formulas)
        return f"Formula({middle})"

    def __repr__(self):
        formulas = [self.formula] + list(self.additional_formulas)
        middle = ", ".join([f"'{formula}'" for formula in formulas])
        return f"Formula({middle})"
