import numpy as np


def coords_from_response(term, family):
    coords_data = {"__obs__": range(term.shape[0])}
    coords = {}
    coords_reduced = {}

    if hasattr(family, "get_levels"):
        levels = family.get_levels(term)
    else:
        levels = term.levels

    if levels:
        coords[f"{term.name}_dim"] = levels
        if term.reference:
            # There's a restriction applied when there is a reference level
            levels_restricted = [level for level in levels if level != term.reference]
        else:
            levels_restricted = levels[1:]

        coords_reduced[f"{term.name}_dim_reduced"] = levels_restricted

    elif family.RESPONSE_NDIM > 1 and term.ndim > 1:
        # A multidimensional numeric outcome.
        # Both non-reduced and reduced dimensions are called the same.
        # Dict union will make sure we attempt to add the dimension only once.
        # We still need regular and reduced coords because that is what things such as
        # additive predictors expect.
        coords[f"{term.name}_dim"] = range(term.shape[1])
        coords_reduced[f"{term.name}_dim"] = range(term.shape[1])

    return coords_data, coords, coords_reduced


def coords_from_common(term):
    # Single numeric
    if term.kind == "numeric":
        if term.ndim == 1:
            return {}
        return {f"{term.name}_dim": range(term.shape[1])}

    # Single categoric
    if term.kind == "categoric":
        if term.spans_intercept:
            name = f"{term.name}_dim"
        else:
            name = f"{term.name}_dim_reduced"
        return {name: term.levels}

    # Interaction
    if term.kind == "interaction":
        # All numerics, return empty dict
        if all(el.kind == "numeric" for el in term.components):
            return {}

        coords = {}
        for el in term.components:
            # A numeric that spans multiple columns (e.g., a spline)
            if el.kind == "numeric" and el.value.ndim == 2 and el.value.shape[1] > 1:
                coords[f"{el.name}_dim"] = range(el.value.shape[1])

            if el.kind == "categoric":
                if el.spans_intercept:
                    name = f"{el.name}_dim"
                else:
                    name = f"{el.name}_dim_reduced"

                coords[name] = el.contrast_matrix.labels
        return coords

    # Otherwise
    return {}


def coords_from_group_specific(term):
    expr_coords = {}
    factor_coords = {}
    expr = term.expr
    factor = term.factor

    # Expression term
    if expr.kind == "numeric" and expr.data.ndim > 1:
        # If numeric, it's non empty only when the term spans multiple columns
        expr_coords = {f"{expr.name}_dim": range(expr.data.shape[1])}
    elif expr.kind == "categoric":
        # Single numeric
        if expr.spans_intercept:
            name = f"{expr.name}_dim"
        else:
            name = f"{expr.name}_dim_reduced"
        expr_coords = {name: expr.levels}
    elif expr.kind == "interaction" and any(el.kind == "categoric" for el in expr.components):
        for el in expr.components:
            # A numeric that spans multiple columns (e.g., a spline)
            if el.kind == "numeric" and el.value.ndim == 2 and el.value.shape[1] > 1:
                expr_coords[f"{el.name}_dim"] = range(el.value.shape[1])

            if el.kind == "categoric":
                if el.spans_intercept:
                    name = f"{el.name}_dim"
                else:
                    name = f"{el.name}_dim_reduced"
                expr_coords[name] = el.contrast_matrix.labels

    # Factor term
    # Factor terms are always non-numeric and non-reduced
    if factor.kind == "categoric":
        factor_coords = {f"{factor.name}_dim": factor.levels}
    elif factor.kind == "interaction":
        # NOTE: Is it true that these components are always categoric?
        #       They should. They should also always span the intercept.
        for el in factor.components:
            factor_coords[f"{el.name}_dim"] = el.contrast_matrix.labels

    return expr_coords, factor_coords


def coords_from_hsgp(term):
    # This handles univariate and multivariate cases
    coords = {f"{term.label}_weights_dim": range(np.prod(term.m))}

    if term.by_levels is not None:
        coords[f"{term.label}_by"] = term.by_levels

    if not term.iso and term.shape[1] > 1:
        coords[f"{term.label}_var"] = range(term.shape[1])

    return coords


def coords_for_cutpoints(parameter_label, response_levels):
    """Return the coordinate for the K - 1 cutpoints of an ordinal response."""
    cutpoint_levels = [
        f"{lower}->{upper}" for lower, upper in zip(response_levels[:-1], response_levels[1:])
    ]
    return {f"{parameter_label}_dim": cutpoint_levels}
