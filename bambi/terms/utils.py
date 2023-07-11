import formulae as fm
from formulae.terms.call_resolver import get_function_from_module


def is_single_component(term) -> bool:
    """Determines if formulae term contains a single component"""
    return hasattr(term, "components") and len(term.components) == 1


def is_call_component(component) -> bool:
    """Determines if formulae component is the result of a function call"""
    return isinstance(component, fm.terms.call.Call)


def is_call_of_kind(call, kind):
    """Determines if formulae call component is of certain kind

    To do so, it checks whether the callee has metadata and whether the 'kind' slot matches the
    kind passed to the function.
    """
    function = get_function_from_module(call.call.callee, call.env)
    return hasattr(function, "__metadata__") and function.__metadata__["kind"] == kind


def is_censored_response(term):
    """Determines if a formulae term represents a censored response"""
    if not is_single_component(term):
        return False
    component = term.components[0]  # get the first (and single) component
    if not is_call_component(component):
        return False
    return is_call_of_kind(component, "censored")
