import numpy as np

import formulae.terms

from bambi.terms.base import BaseTerm


class CommonTerm(BaseTerm):
    """A common model term."""

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.data = np.squeeze(term.data)
        self.prefix = prefix

        if self.categorical and len(self.levels) == 1 and (self.data == self.data[0]).all():
            raise ValueError(f"The term '{self.name}' has only 1 category!")

        if (
            not self.categorical
            and self.kind not in ("intercept", "offset")
            and np.all(self.data == self.data[0])
        ):
            raise ValueError(f"The term '{self.name}' is constant!")

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, (formulae.terms.terms.Term, formulae.terms.terms.Intercept))
        self._term = value

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def coords(self):
        # Obtain pymc coordinates, only for categorical components of a term.
        # A categorical component can have up to two coordinates in the same model if it is
        # includied with both reduced and full rank encodings.
        coords = {}
        if self.categorical:
            name = self.name + "_dim"
            coords[name] = self.levels
        elif self.data.ndim > 1 and self.data.shape[1] > 1:
            name = self.name + "_dim"
            coords[name] = np.arange(self.data.shape[1])
        return coords

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def kind(self):
        return self.term.kind

    @property
    def shape(self):
        return self.data.shape

    @property
    def categorical(self):
        # If the term has one component, it's categorical if the component is categorical.
        # If the term has more than one component (i.e. it is an interaction), it's categorical if
        # at least one of the components is categorical.
        if self.kind == "interaction":
            return any(component.kind == "categoric" for component in self.term.components)
        return self.kind == "categoric"

    @property
    def levels(self):
        return self.term.levels

    def __str__(self):
        args = []
        if self.coords:
            args = [f"coords: {self.coords}"]
        return self.make_str(args)
