import pymc as pm


def build_potentials(potentials, model: pm.Model) -> None:
    """Add user-defined potentials to the PyMC model."""
    if potentials is None:
        return

    available_names = sorted(model.named_vars)

    for index, (variables, constraint) in enumerate(potentials):
        if not callable(constraint):
            raise TypeError(f"Potential constraint at index {index} must be callable.")

        if isinstance(variables, (list, tuple)):
            variable_names = variables
        else:
            variable_names = (variables,)

        missing_names = [name for name in variable_names if name not in model.named_vars]
        if missing_names:
            missing = ", ".join(missing_names)
            available = ", ".join(available_names)
            raise ValueError(
                f"Potential variable(s) not found in the PyMC model: {missing}. "
                f"Available variables are: {available}."
            )

        variable_values = [model[name] for name in variable_names]
        pm.Potential(f"pot_{index}", constraint(*variable_values), model=model)
