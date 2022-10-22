import numpy as np

import formulae.terms

from bambi.new_terms.base import BaseTerm

class CommonTerm(BaseTerm):
    """A common model term."""
    categorical = False
    data = None
    shape = None
    coords = {}

    def __init__(self, term, prior):
        self.term = term
        self.prior = prior
        self._name = term.name
        self.data = np.squeeze(term.data)
        self.kind = term.kind
        self.levels = term.levels

        # If the term has one component, it's categorical if the component is categorical.
        # If the term has more than one component (i.e. it is an interaction), it's categorical if
        # at least one of the components is categorical.
        if self.kind == "interaction":
            if any(component.kind == "categoric" for component in self.term.components):
                self.categorical = True
        else:
            self.categorical = self.kind == "categoric"

        # Flag constant terms
        if self.categorical and len(term.levels) == 1 and (self.data == self.data[0]).all():
            raise ValueError(f"The term '{name}' has only 1 category!")

        if not self.categorical and self.kind != "intercept" and np.all(self.data == self.data[0]):
            raise ValueError(f"The term '{name}' is constant!")

        # Obtain pymc coordinates, only for categorical components of a term.
        # A categorical component can have up to two coordinates if it is including with both
        # reduced and full rank encodings.
        if self.categorical:
            name = self.name + "_dim"
            self.coords[name] = self.term.levels
        elif self.data.ndim > 1 and self.data.shape[1] > 1:
            name = self.name + "_dim"
            self.coords[name] = np.arange(self.data.shape[1])

    @property
    def name(self):
        return self._name

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Term)
        self._term = value

    @property
    def shape(self):
        return self.data.shape

    def __str__(self):
        args = []
        if self.coords:
            args = [f"coords: {self.coords}"]
        return super().__str__(args)
        