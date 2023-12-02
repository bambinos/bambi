import formulae.terms

from bambi.terms.base import BaseTerm

from bambi.terms.utils import is_response_of_kind


class ResponseTerm(BaseTerm):
    def __init__(self, response, family):
        self.term = response.term.term
        self.family = family
        self.is_censored = is_response_of_kind(self.term, "censored")
        self.is_constrained = is_response_of_kind(self.term, "constrained")
        self.is_truncated = is_response_of_kind(self.term, "truncated")
        self.is_weighted = is_response_of_kind(self.term, "weighted")

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Term)
        self._term = value

    @property
    def data(self):
        if hasattr(self.family, "get_data"):
            return self.family.get_data(self)
        return self.term.data

    @property
    def name(self):
        return self.term.name

    @property
    def shape(self):
        return self.data.shape

    @property
    def levels(self):
        if hasattr(self.family, "get_levels"):
            return self.family.get_levels(self)
        return self.term.levels

    @property
    def categorical(self):
        return self.term.kind == "categoric"

    @property
    def reference(self):
        if hasattr(self.family, "get_reference"):
            return self.family.get_reference(self)
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
    def binary(self):
        # Maybe it's not needed here anymore
        if self.categorical:
            if self.term.levels is None:
                return True
            return len(self.term.levels) == 2
        return None

    def __str__(self):
        extras = []
        if self.categorical:
            if self.binary:
                extras += [f"success: {self.success}"]
            else:
                extras += [f"reference: {self.reference}"]
        return self.make_str(extras)
