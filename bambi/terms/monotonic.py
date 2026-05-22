"""Bambi front-end representation of a standalone monotonic effect ``mo(x)``."""

import numpy as np

import formulae.terms

from bambi.terms.base import BaseTerm
from bambi.terms._monotonic_helpers import (
    extract_component_info,
    simplex_names,
    validate_prior_dict,
)


class MonotonicTerm(BaseTerm):
    """Representation of a monotonic-effect term as in ``brms::mo()``.

    The predictor is an integer or ordered categorical with ``K`` distinct values.
    The linear-predictor contribution is ``slope * D * sum(simplex[1:x])`` where
    ``simplex`` is a length-``D`` simplex (``D = K - 1``) and ``slope`` is a scalar.

    Parameters
    ----------
    term : formulae.terms.terms.Term
        A formulae term wrapping a ``mo(...)`` call.
    prior : dict or None
        Dict with optional keys ``"slope"`` and ``"simplex"``, each mapping to a
        ``bambi.Prior``. Any missing key is filled by defaults at build time.
    prefix : str, optional
        Used when the term belongs to a non-parent distributional component.
    """

    _ALLOWED_PRIOR_KEYS = ("slope", "simplex")
    _PRIOR_KIND_LABEL = "monotonic 'mo()' term"

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.prefix = prefix
        self._info = extract_component_info(self.term.components[0])

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Term)
        self._term = value

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        self._prior = validate_prior_dict(
            value, self._ALLOWED_PRIOR_KEYS, self._PRIOR_KIND_LABEL
        )

    @property
    def data(self):
        # formulae stores the (n, 1) array returned by the stateful transform.
        return self.term.data

    @property
    def codes(self):
        """Integer codes (zero-indexed) of the monotonic predictor."""
        return self._info["codes"]

    @property
    def shape(self):
        return self.term.data.shape

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def categorical(self):
        return False

    @property
    def levels(self):
        return self._info["levels"]

    @property
    def K(self):
        return self._info["K"]

    @property
    def D(self):
        return self._info["D"]

    @property
    def kind(self):
        return self._info["kind"]

    @property
    def transform(self):
        return self._info["transform"]

    @property
    def id(self):
        """Optional shared-simplex group id (brms-style)."""
        return self._info["id"]

    @property
    def simplex_name(self):
        name, _dim = simplex_names(self.id, self.name)
        return name

    @property
    def simplex_dim(self):
        _name, dim = simplex_names(self.id, self.name)
        return dim

    @property
    def coords(self):
        # One coord for the simplex elements. Each element is the "step" from one
        # category to the next, so we label by destination level.
        return {self.simplex_dim: np.asarray(self.levels[1:])}
