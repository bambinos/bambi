"""Bambi front-end representation of a group-specific monotonic effect ``(mo(x) | g)``."""

import numpy as np

import formulae.terms

from bambi.terms.base import BaseTerm
from bambi.terms._monotonic_helpers import (
    extract_component_info,
    simplex_names,
    validate_prior_dict,
)


# pylint: disable = invalid-name
class MonotonicGroupSpecificTerm(BaseTerm):
    """Group-specific monotonic effect, ``(mo(x) | g)``.

    Per-group slopes ``r_g`` with a hierarchical Normal/HalfNormal prior:

        r_g ~ Normal(0, sigma_g), sigma_g ~ HalfNormal(...)
        contribution[i] = r_g[group[i]] * D * cumsum(simplex)[codes[i]]

    The simplex is shared across groups (matching brms). Use ``id=`` to also
    share it with standalone ``mo()`` terms.

    Parameters
    ----------
    term : formulae.terms.terms.GroupSpecificTerm
        The formulae group-specific term whose ``expr`` is a Monotonic call.
    prior : dict or None
        Optional dict with keys ``"slope"`` (Normal w/ HalfNormal hyperprior on
        sigma) and/or ``"simplex"`` (Dirichlet). Defaults supplied at build time.
    prefix : str, optional
        Prefix for non-parent distributional components.
    """

    _ALLOWED_PRIOR_KEYS = ("slope", "simplex")
    _PRIOR_KIND_LABEL = "group-specific mo() term"

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.prefix = prefix
        self._info = extract_component_info(self.term.expr.components[0])
        self._group_index = self._invert_dummies(self.grouper)

    @staticmethod
    def _invert_dummies(dummies):
        # `dummies` is a (n, n_groups) one-hot. Return the index per row.
        if hasattr(dummies, "toarray"):  # sparse
            dummies = dummies.toarray()
        return np.asarray(dummies).argmax(axis=1)

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        if not isinstance(value, formulae.terms.terms.GroupSpecificTerm):
            raise TypeError(
                "'MonotonicGroupSpecificTerm.term' must be a formulae "
                f"GroupSpecificTerm, got {type(value).__name__}."
            )
        self._term = value

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        self._prior = validate_prior_dict(value, self._ALLOWED_PRIOR_KEYS, self._PRIOR_KIND_LABEL)

    @property
    def data(self):
        return self.term.data

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def shape(self):
        if hasattr(self.term.data, "shape"):
            return self.term.data.shape
        return (None, None)

    @property
    def categorical(self):
        return False

    @property
    def levels(self):
        return self.term.labels

    @property
    def groups(self):
        return self.term.groups

    @property
    def grouper(self):
        return self.term.factor.data

    @property
    def codes(self):
        return self._info["codes"]

    @property
    def D(self):
        return self._info["D"]

    @property
    def K(self):
        return self._info["K"]

    @property
    def mo_levels(self):
        return self._info["levels"]

    @property
    def id(self):
        return self._info["id"]

    @property
    def transform(self):
        return self._info["transform"]

    @property
    def group_index(self):
        return self._group_index

    @property
    def simplex_name(self):
        name, _dim = simplex_names(self.id, self.name)
        return name

    @property
    def simplex_dim(self):
        _name, dim = simplex_names(self.id, self.name)
        return dim

    @property
    def factor_dim(self):
        # Reuse the same naming convention bambi uses for regular group-specific terms,
        # so xarray downstream tooling stays consistent.
        _, factor = self.name.split("|", 1)
        return f"{factor}__factor_dim"

    @property
    def coords(self):
        return {
            self.simplex_dim: np.asarray(self.mo_levels[1:]),
            self.factor_dim: list(self.groups),
        }
