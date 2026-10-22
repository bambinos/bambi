import formulae as fm
from formulae.terms.call_resolver import get_function_from_module

from bambi.transformations import HSGP, SmoothTransform


def is_single_component(term) -> bool:
    """Determines if formulae term contains a single component."""
    return hasattr(term, "components") and len(term.components) == 1


def is_call_component(component) -> bool:
    """Determines if formulae component is the result of a function call."""
    return isinstance(component, fm.terms.call.Call)


def is_call_of_kind(call, kind: str) -> bool:
    """Determines if formulae call component is of certain kind

    To do so, it checks whether the callee has metadata and whether the 'kind' slot matches the
    kind passed to the function.
    """
    function = get_function_from_module(call.call.callee, call.env)
    return hasattr(function, "__metadata__") and function.__metadata__["kind"] == kind


def is_response_of_kind(term, kind: str) -> bool:
    """Determines if a formulae term represents a response of a certain kind"""
    if not is_single_component(term):
        return False
    component = term.components[0]  # get the first (and single) component
    if not is_call_component(component):
        return False
    return is_call_of_kind(component, kind)


def is_smooth_term(term):
    """Determine whether a formulae term is a smooth transform."""
    return any(
        is_call_component(component)
        and isinstance(component.call.stateful_transform, SmoothTransform)
        for component in getattr(term, "components", ())
    )


def is_hsgp_term(term):
    """Determine whether a formulae term is an HSGP transform."""
    return any(
        is_call_component(component) and isinstance(component.call.stateful_transform, HSGP)
        for component in getattr(term, "components", ())
    )
