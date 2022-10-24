import numpy as np

import formulae.terms
from bambi.families.univariate import Bernoulli

from bambi.new_terms.base import BaseTerm
from bambi.new_terms.utils import get_reference_level, is_censored_response
from bambi.terms import get_success_level
from bambi.utils import extract_argument_names, extra_namespace
from bambi.families.multivariate import Categorical, Multinomial


class ResponseTerm(BaseTerm):
    def __init__(self, term, family):
        self.term = term
        self.family = family

        self.is_censored = is_censored_response(self.term.term)

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Response)
        self._term = value

    @property
    def name(self):
        return self.term.name

    @property
    def categorical(self):
        return self.term.kind == "categoric"

    @property
    def reference(self):
        # Should this be handled by the family?
        if self.categorical and not self.binary:
            return get_reference_level(self.term.term.term)
        return None

    @property
    def levels(self):
        # Some families, like Multinomial, override the levels attribute
        if hasattr(self.family, "get_levels"):
            return self.family.get_levels(self.term)
        if self.categorical:
            return self.term.levels
        return None

    @property
    def coords(self):
        if hasattr(self.family, "get_coords"):
            return self.family.get_coords(self.term)
        return {}

    @property
    def binary(self):
        if self.categorical:
            if self.term.levels is None:
                return True
            else:
                return len(self.term.levels) == 2
        return None

    @property
    def success(self):
        if self.binary:
            return get_success_level(self.term.term.term)
        elif isinstance(self.family, Bernoulli):
            return 1
        return None

    @property
    def data(self):
        if self.categorical:
            if self.binary:
                if self.term.design_matrix.ndim == 1:
                    return self.term.design_matrix
                else:
                    idx = self.levels.index(self.success)
                    return self.term.design_matrix[:, idx]
            else:
                return np.nonzero(self.term.design_matrix)[1]
        return self.term.design_matrix

    def __str__(self):
        extras = []
        if self.categorical:
            if self.binary:
                extras += [f"success: {self.success}"]
            else:
                extras += [f"reference: {self.reference}"]

        return super().__str__(extras)

    def __repr__(self):
        return self.__str__()


# How to make ResponseTerm more modular? For example, to have more flexibility when it
# comes to new families?
# Should that be a method of the family?
