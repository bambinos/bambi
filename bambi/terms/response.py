import formulae.terms
from formulae.terms.call_resolver import get_function_from_module

from bambi.terms.base import BaseTerm
from bambi.terms.utils import is_response_of_kind


class ResponseTerm(BaseTerm):
    def __init__(self, response):
        self.term = response.term.term
        self.is_censored = is_response_of_kind(self.term, "censored")
        self.is_constrained = is_response_of_kind(self.term, "constrained")
        self.is_truncated = is_response_of_kind(self.term, "truncated")
        self.is_weighted = is_response_of_kind(self.term, "weighted")
        self.is_counts = is_response_of_kind(self.term, "counts")
        self.is_binomial = self.term.kind == "proportion"  # NOTE: could update formulae pkg

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Term)
        self._term = value

    @property
    def name(self):
        return self.term.name

    @property
    def data(self):
        return self.term.data

    @property
    def shape(self):
        return self.data.shape

    @property
    def categorical(self):
        return self.term.kind == "categoric"

    @property
    def levels(self):
        return self.term.levels

    @property
    def reference(self):
        """The reference level of the term.

        It returns `None` when the concept of "reference level" does not apply.
        """
        if self.term.kind != "categoric":
            return None

        if self.term.levels is None:
            return self.term.components[0].reference

        return self.term.levels[0]

    def eval_new_data(self, data):
        """Evaluate response data on a new data frame."""
        if len(self.components) != 1:
            return self.term.eval_new_data(data)

        component = self.components[0]
        if not hasattr(component, "call"):
            return self.term.eval_new_data(data)

        function = get_function_from_module(component.call.callee, component.env)
        args = [
            arg.eval(data, component.env) if hasattr(arg, "eval") else arg
            for arg in component.call.args
        ]
        kwargs = {
            name: value.eval(data, component.env) if hasattr(value, "eval") else value
            for name, value in component.call.kwargs.items()
        }
        value = function(*args, **kwargs)
        if hasattr(value, "eval"):
            return value.eval()
        return value

    def __str__(self):
        extras = []
        if self.categorical:
            extras.append(f"reference: {self.reference}")
        return self.make_str(extras)
