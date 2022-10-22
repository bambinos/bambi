import numpy as np

import formulae.terms

from bambi.new_terms.base import BaseTerm

class GroupSpecificTerm:
    # pylint: disable=too-many-instance-attributes
    """Representation of a single (group specific) model term.

    Parameters
    ----------
    name: str
        Name of the term.
    term: dict
        A dictionary describing the components of the term. These can be found in
        ``formulae.design_matrices().group.terms_info``
    data: (DataFrame, Series, ndarray)
        The term values.
    prior: Prior
        A specification of the prior(s) to use. An instance of class ``priors.Prior``.
    """

    _hyperprior_alias = {}
    coords = {}

    def __init__(self, term, prior):
        self.term = term
        self.prior = prior

        self._name = term.name
        self.data = np.squeeze(term.data)
        self.kind = term.kind
        self.groups = term.groups
        self.levels = term.labels
        self.grouper = term.factor.data
        self.predictor = term.expr.data
        self.group_index = self.invert_dummies(self.grouper)

        # Determine if the expression is categorical
        if self.kind == "interaction":
            if any(component.kind == "categoric" for component in self.term.expr.components):
                self.categorical = True
        else:
            self.categorical = self.kind == "categoric"

        # The group is _always_ added as a coordinate
        expr, factor = self.name.split("|")
        self.coords[factor + "__factor_dim"] = self.groups

        if self.categorical:
            self.coords[expr + "__expr_dim"] = term.expr.levels
        elif self.predictor.ndim == 2 and self.predictor.shape[1] > 1:
            self.coords[expr + "__expr_dim"] = [str(i) for i in range(self.predictor.shape[1])]

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
    def hyperprior_alias(self):
        return self._hyperprior_alias

    @hyperprior_alias.setter
    def hyperprior_alias(self, values):
        assert all(isinstance(x, str) for x in values.keys())
        assert all(isinstance(x, str) for x in values.values())
        self._hyperprior_alias.update(values)

    @property
    def name(self):
        return self._name

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.GroupSpecificTerm)
        self._term = value

    def __str__(self):
        args = [f"groups: {self.groups}"]
        return super().__str__(args)

# TODO: Priors! Make sure they're hierarchical