from formulae.terms.call import Call
from formulae.terms.call_resolver import get_function_from_module


# pylint: disable = protected-access
def get_reference_level(term):
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return None

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]


# pylint: disable = protected-access
def get_success_level(term):
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return term.components[0].reference

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]


def is_single_component(term):
    return len(term.term.components) == 1


def extract_first_component(term):
    return term.term.components[0]


def is_call_component(component):
    return isinstance(component, Call)


def is_call_of_kind(call, kind):
    function = get_function_from_module(call.call.callee, call.env)
    return hasattr(function, "__metadata__") and function.__metadata__["kind"] == kind


def is_censored_response(term):
    if not is_single_component(term):
        return False
    component = extract_first_component(term)
    if not is_call_component(component):
        return False
    return is_call_of_kind(component, "censored")
