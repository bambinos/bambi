"""Private helpers shared by the three front-end monotonic term classes.

Keeps the per-class files thin while letting each remain a direct subclass of
``BaseTerm`` (matching bambi's term-package convention).
"""

from typing import Iterable, Optional

import numpy as np

from bambi.priors.prior import Prior


VALID_MONOTONIC_PRIOR_VALUES = (Prior, int, float, np.ndarray, type(None))


def extract_component_info(call_component, idx: int = 0) -> dict:
    """Extract per-mo()-component metadata from a formulae Call component.

    Returns a plain ``dict`` (rather than a dataclass/namedtuple) because the
    predict path mutates it to cache re-encoded codes for new data.

    Used by ``MonotonicTerm`` (one component), ``MonotonicInteractionTerm``
    (one or more), and ``MonotonicGroupSpecificTerm`` (one, in the term's
    ``expr``).
    """
    tx = call_component.call.stateful_transform
    codes = np.asarray(call_component.value).squeeze().astype("int64")
    return {
        "transform": tx,
        "codes": codes,
        "D": tx.D,
        "K": tx.K,
        "levels": tx.levels,
        "id": tx.id,
        "kind": tx.kind,
        "idx": idx,
    }


def validate_prior_dict(
    value, allowed_keys: Iterable[str], kind_label: str
) -> Optional[dict]:
    """Validate a user-supplied prior dict for a monotonic term.

    Returns ``None`` if the input is ``None``; otherwise returns the dict
    unchanged after checking it has the expected shape and only allowed keys.

    Parameters
    ----------
    value :
        The candidate prior value to validate.
    allowed_keys :
        The keys this term type accepts (e.g. ``{"slope", "simplex"}``).
    kind_label :
        Human-readable label for the term type, used in error messages
        (e.g. ``"monotonic 'mo()' term"``).
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        keys_str = "', '".join(sorted(allowed_keys))
        raise ValueError(
            f"The prior for a {kind_label} must be a dict with keys '{keys_str}', or None."
        )
    allowed = set(allowed_keys)
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys in {kind_label} prior dict: {sorted(unknown)}. "
            f"Allowed keys: {sorted(allowed)}."
        )
    for v in value.values():
        assert isinstance(
            v, VALID_MONOTONIC_PRIOR_VALUES
        ), f"Prior values must be one of {VALID_MONOTONIC_PRIOR_VALUES}"
    return value


def simplex_names(mo_id: Optional[str], term_name: str) -> tuple[str, str]:
    """Compute ``(simplex_name, simplex_dim)`` for a monotonic term.

    For shared-id terms, the simplex is named ``simplex_<id>`` so a single
    Dirichlet variable is emitted and reused across terms with the same id.
    For un-shared terms, the names are scoped to the term's own name.
    """
    if mo_id is not None:
        return f"simplex_{mo_id}", f"simplex_{mo_id}_dim"
    return f"{term_name}_simplex", f"{term_name}_simplex_dim"
