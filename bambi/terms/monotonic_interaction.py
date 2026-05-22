"""Bambi front-end representation of a monotonic interaction term.

Handles formulae interaction terms such as ``mo(x):z``, ``mo(x):mo(y)``, or
``mo(x):g`` (categorical right-hand side).
"""

import formulae.terms

from bambi.terms.base import BaseTerm
from bambi.terms._monotonic_helpers import (
    extract_component_info,
    validate_prior_dict,
)
from bambi.utils import is_monotonic_component


# pylint: disable = invalid-name
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
    set), reusing the same simplex registry as standalone ``MonotonicTerm`` s.

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

    _ALLOWED_PRIOR_KEYS = ("slope",)
    _PRIOR_KIND_LABEL = "monotonic-interaction term"

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.prefix = prefix
        self._mono = [
            extract_component_info(comp, idx=i)
            for i, comp in enumerate(self.term.components)
            if is_monotonic_component(comp)
        ]
        if not self._mono:
            raise AssertionError("MonotonicInteractionTerm built with no mo() components")

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        if not isinstance(value, formulae.terms.terms.Term):
            raise TypeError(
                "'MonotonicInteractionTerm.term' must be a formulae Term, "
                f"got {type(value).__name__}."
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
        # Not used directly — predict/build pull the design slice via the
        # component's slot in `design.common`.
        return None

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
        """List of per-mo() metadata dicts.

        Each entry has keys: ``transform, codes, D, K, levels, id, kind, idx``.
        See ``bambi.terms._monotonic_helpers.extract_component_info``.
        """
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
