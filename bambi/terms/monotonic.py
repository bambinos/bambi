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
    def id(self):
        """Optional shared-simplex group id (brms-style)."""
        return self._mo_attrs["id"]

    @property
    def simplex_name(self):
        """Name of the PyMC simplex variable.

        For shared-id terms, this is ``simplex_<id>`` so a single Dirichlet is
        emitted and reused. For un-shared terms, it's ``{term.name}_simplex``.
        """
        if self.id is not None:
            return f"simplex_{self.id}"
        return f"{self.name}_simplex"

    @property
    def simplex_dim(self):
        if self.id is not None:
            return f"simplex_{self.id}_dim"
        return f"{self.name}_simplex_dim"

    @property
    def coords(self):
        # One coord for the simplex elements. Each element is the "step" from one
        # category to the next, so we label by destination level.
        levels = self.levels
        return {self.simplex_dim: np.asarray(levels[1:])}


class MonotonicInteractionTerm(BaseTerm):
    """An interaction term containing at least one ``mo()`` component.

    The linear-predictor contribution for a row ``i`` is

        sum_k slope_k * (prod_m D_m * cumsum(simplex_m)[codes_m[i]]) * other_factor[i, k]

    where the product is over the ``mo()`` components in the interaction,
    ``other_factor`` is the matrix of the non-mo factors (recovered from the
    formulae design-matrix slice by dividing out the raw mo codes), and ``k``
    indexes the columns of the interaction design slice (single column for
    interactions with continuous variables, multiple for categorical).

    Each mo() component carries its own simplex (or a shared one when ``id=`` is
    set), reusing the same simplex registry as standalone ``MonotonicTerm``s.

    Parameters
    ----------
    term : formulae.terms.terms.Term
        The interaction term.
    prior : dict or None
        Optional ``{"slope": Prior}``. Simplex priors are taken from id-matched
        terms; if no id is set on a mo() component, the default Dirichlet(1) is
        used.
    prefix : str, optional
        Prefix for non-parent distributional components.
    """

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.prefix = prefix
        self._init_components()

    def _init_components(self):
        from bambi.utils import is_monotonic_component  # local import to avoid cycle

        mono = []
        for i, comp in enumerate(self.term.components):
            if is_monotonic_component(comp):
                tx = comp.call.stateful_transform
                codes = np.asarray(comp.value).squeeze().astype("int64")
                mono.append(
                    {
                        "idx": i,
                        "transform": tx,
                        "codes": codes,
                        "D": tx.D,
                        "K": tx.K,
                        "levels": tx.levels,
                        "id": tx.id,
                    }
                )
        if not mono:
            raise AssertionError("MonotonicInteractionTerm built with no mo() components")
        self._mono = mono

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
                "The prior for a monotonic interaction must be a dict with key 'slope' or None."
            )
        unknown = set(value) - {"slope"}
        if unknown:
            raise ValueError(
                f"Unknown keys in monotonic-interaction prior dict: {sorted(unknown)}. "
                "Only 'slope' is configurable here; simplex priors are set on the standalone "
                "mo() term (using id= to share)."
            )
        for v in value.values():
            assert isinstance(
                v, VALID_MONOTONIC_PRIOR_VALUES
            ), f"Prior values must be one of {VALID_MONOTONIC_PRIOR_VALUES}"
        self._prior = value

    @property
    def data(self):
        # Full interaction design-matrix slice as formulae built it.
        # Shape (n, k) where k is the number of dummy columns. Note: each row's
        # values include the raw mo() codes multiplied in -- we undo that during
        # build/predict.
        return None  # Not used directly; we work off the design slice in the component.

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def shape(self):
        # Derived later from the design matrix when needed.
        return (None, None)

    @property
    def categorical(self):
        return False

    @property
    def levels(self):
        return None

    @property
    def mono_components(self):
        return self._mono

    @property
    def all_ids(self):
        return [m["id"] for m in self._mono]

    @property
    def D_product(self):
        D = 1
        for m in self._mono:
            D *= m["D"]
        return D


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

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.prefix = prefix
        self._init_from_expr()
        self._group_index = self._invert_dummies(self.grouper)

    def _init_from_expr(self):
        component = self.term.expr.components[0]
        tx = component.call.stateful_transform
        self._transform = tx
        self._codes = np.asarray(component.value).squeeze().astype("int64")
        self._D = tx.D
        self._K = tx.K
        self._levels = tx.levels
        self._mo_id = tx.id

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
        assert isinstance(value, formulae.terms.terms.GroupSpecificTerm)
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
                "The prior for a group-specific mo() term must be a dict with keys "
                "'slope' and/or 'simplex', or None."
            )
        unknown = set(value) - {"slope", "simplex"}
        if unknown:
            raise ValueError(
                f"Unknown keys in group-specific mo() prior dict: {sorted(unknown)}. "
                "Allowed keys: 'slope', 'simplex'."
            )
        self._prior = value

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
        return self._codes

    @property
    def D(self):
        return self._D

    @property
    def K(self):
        return self._K

    @property
    def mo_levels(self):
        return self._levels

    @property
    def id(self):
        return self._mo_id

    @property
    def transform(self):
        return self._transform

    @property
    def group_index(self):
        return self._group_index

    @property
    def simplex_name(self):
        if self.id is not None:
            return f"simplex_{self.id}"
        return f"{self.name}_simplex"

    @property
    def simplex_dim(self):
        if self.id is not None:
            return f"simplex_{self.id}_dim"
        return f"{self.name}_simplex_dim"

    @property
    def factor_dim(self):
        # Reuse the same naming convention bambi uses for regular group-specific terms,
        # so xarray downstream tooling stays consistent.
        _, factor = self.name.split("|", 1)
        return f"{factor}__factor_dim"

    @property
    def coords(self):
        coords = {
            self.simplex_dim: np.asarray(self.mo_levels[1:]),
            self.factor_dim: list(self.groups),
        }
        return coords


def _get_monotonic_attributes(term):
    """Pull the recorded state off the underlying ``Monotonic`` stateful transform."""
    attrs = term.components[0].call.stateful_transform.__dict__
    return {
        "levels": attrs["levels"],
        "K": attrs["K"],
        "D": attrs["D"],
        "kind": attrs["kind"],
        "min_value": attrs["min_value"],
        "id": attrs.get("id"),
    }
