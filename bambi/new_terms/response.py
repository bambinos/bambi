import numpy as np

import formulae.terms

from bambi.new_terms.base import BaseTerm
from bambi.new_terms.utils import get_reference_level, is_censored_response
from bambi.terms import get_success_level


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
        if hasattr(self.family, "get_reference"):
            return self.family.get_reference(self)
        return None

    @property
    def levels(self):
        if hasattr(self.family, "get_levels"):
            return self.family.get_levels(self)
        return None

    @property
    def coords(self):
        if hasattr(self.family, "get_coords"):
            return self.family.get_coords(self)
        return {}

    @property
    def success(self):
        if hasattr(self.family, "get_success_level"):
            return self.family.get_success_level(self)
        return None

    @property
    def data(self):
        if hasattr(self.family, "get_data"):
            return self.family.get_data(self)
        return self.term.design_matrix

    @property
    def binary(self):
        # Maybe it's not needed here anymore
        if self.categorical:
            if self.term.levels is None:
                return True
            return len(self.term.levels) == 2
        return None

    def __str__(self):  # pylint: disable=signature-differs
        extras = []
        if self.categorical:
            if self.binary:
                extras += [f"success: {self.success}"]
            else:
                extras += [f"reference: {self.reference}"]
        return super().__str__(extras)


# TODO: Implement censored responses


# Categorical
# -> Nominal
#   -> Binary
# -> Ordinal

# These aren't actually used to do something with data, but mostly to give information to the user
# Well, the ordinal kind can be useful as well.
# class Categorical:
#     pass

# class Nominal(Categorical):
#     pass

# class Ordinal(Categorical):
#     pass

# class Binary(Nominal):
#     pass
