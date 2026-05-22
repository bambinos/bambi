"""Bambi front-end representation of a monotonic effect term ``mo(x)``."""

import numpy as np

import formulae.terms

from bambi.priors.prior import Prior
from bambi.terms.base import BaseTerm


VALID_MONOTONIC_PRIOR_VALUES = (Prior, int, float, np.ndarray, type(None))


class MonotonicTerm(BaseTerm):
    """Representation of a monotonic-effect term as in ``brms::mo()``.

    The predictor is an integer or ordered categorical with ``K`` distinct values. The
    linear-predictor contribution is ``slope * D * sum(simplex[1:x])`` where
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

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.prefix = prefix
        self._mo_attrs = _get_monotonic_attributes(term)

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
        if value is None:
            self._prior = None
            return
        if not isinstance(value, dict):
            raise ValueError(
                "The prior for a monotonic 'mo()' term must be a dict with keys 'slope' "
                "and/or 'simplex', or None."
            )
        unknown = set(value) - {"slope", "simplex"}
        if unknown:
            raise ValueError(
                f"Unknown keys in monotonic prior dict: {sorted(unknown)}. "
                "Allowed keys: 'slope', 'simplex'."
            )
        for v in value.values():
            assert isinstance(
                v, VALID_MONOTONIC_PRIOR_VALUES
            ), f"Prior values must be one of {VALID_MONOTONIC_PRIOR_VALUES}"
        self._prior = value

    @property
    def data(self):
        # formulae stores the (n, 1) array returned by the stateful transform.
        return self.term.data

    @property
    def codes(self):
        """Integer codes (zero-indexed) of the monotonic predictor."""
        return np.asarray(self.term.data).squeeze().astype("int64")

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
        return self._mo_attrs["levels"]

    @property
    def K(self):
        return self._mo_attrs["K"]

    @property
    def D(self):
        return self._mo_attrs["D"]

    @property
    def kind(self):
        return self._mo_attrs["kind"]

    @property
    def coords(self):
        # One coord for the simplex elements. Each element is the "step" from one
        # category to the next, so we label by destination level.
        levels = self.levels
        # levels is either the categorical levels or the sorted unique integers; we
        # label the D simplex entries by levels[1:] (the "to" side of each step).
        return {f"{self.name}_simplex_dim": np.asarray(levels[1:])}


def _get_monotonic_attributes(term):
    """Pull the recorded state off the underlying ``Monotonic`` stateful transform."""
    attrs = term.components[0].call.stateful_transform.__dict__
    return {
        "levels": attrs["levels"],
        "K": attrs["K"],
        "D": attrs["D"],
        "kind": attrs["kind"],
        "min_value": attrs["min_value"],
    }
