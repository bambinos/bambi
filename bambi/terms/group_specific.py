import numpy as np

import formulae.terms

from bambi.terms.base import BaseTerm, VALID_PRIORS
from bambi.priors.prior import Prior


class GroupSpecificTerm(BaseTerm):  # pylint: disable=too-many-instance-attributes
    def __init__(self, term, prior, prefix=None):
        self._hyperprior_alias = {}
        self.term = term
        self.prior = prior
        self.data = np.squeeze(term.data)
        self.group_index = self.invert_dummies(self.grouper)
        self.prefix = prefix

    def invert_dummies(self, dummies):
        """
        For the sake of computational efficiency (i.e., to avoid lots of large matrix
        multiplications in the backend), invert the dummy-coding process and represent full-rank
        dummies as a vector of indices into the coefficients.
        """
        vector = np.zeros(len(dummies), dtype=int)
        for i in range(1, dummies.shape[1]):
            vector[dummies[:, i] == 1] = i
        return vector

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.GroupSpecificTerm)
        self._term = value

    @property
    def coords(self):
        # The group is _always_ added as a coordinate. Maybe there's a cleaner way
        coords = {}
        expr, factor = self.name.split("|")
        coords[factor + "__factor_dim"] = self.groups

        if self.categorical:
            coords[expr + "__expr_dim"] = self.term.expr.levels
        elif self.predictor.ndim == 2 and self.predictor.shape[1] > 1:
            coords[expr + "__expr_dim"] = np.arange(self.predictor.shape[1])
        return coords

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def kind(self):
        return self.term.kind

    @property
    def shape(self):
        return self.data.shape

    @property
    def categorical(self):
        # Determine if the expression is categorical
        if self.kind == "interaction":
            return any(component.kind == "categoric" for component in self.term.expr.components)
        return self.kind == "categoric"

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        # This does not check which argument has hyperprior (must be dispersion?)
        assert isinstance(value, VALID_PRIORS), f"Prior must be one of {VALID_PRIORS}"
        if isinstance(value, Prior):
            any_hyperprior = any(isinstance(x, Prior) for x in value.args.values())
            if not any_hyperprior:
                raise ValueError("Prior for group-specific terms must have hyperpriors")
        self._prior = value

    @property
    def groups(self):
        return self.term.groups

    @property
    def levels(self):
        return self.term.labels

    @property
    def predictor(self):
        return self.term.expr.data

    @property
    def grouper(self):
        return self.term.factor.data

    @property
    def hyperprior_alias(self):
        return self._hyperprior_alias

    @hyperprior_alias.setter
    def hyperprior_alias(self, values):
        assert all(isinstance(x, str) for x in values.keys())
        assert all(isinstance(x, str) for x in values.values())
        self._hyperprior_alias.update(values)

    def __str__(self):
        args = [f"groups: {self.groups}"]
        return self.make_str(args)
