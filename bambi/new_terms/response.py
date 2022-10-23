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

        # We use pymc coords when the response is multi-categorical.
        # These help to give the appropriate shape to coefficients and make the resulting
        # InferenceData object much cleaner
        self.coords = {}
        if isinstance(family, Categorical):
            name = self.name + "_dim"
            self.coords[name] = [level for level in term.levels if level != self.reference]
        elif isinstance(family, Multinomial):
            name = self.name + "_dim"
            labels = extract_argument_names(self.name, list(extra_namespace))
            if labels:
                self.levels = labels
            else:
                self.levels = [str(level) for level in range(self.data.shape[1])]
            labels = self.levels[1:]
            self.coords[name] = labels

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
        if self.categorical:
            return self.term.levels
        return None

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
