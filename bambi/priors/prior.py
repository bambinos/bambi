import pytensor
import numpy as np


class Prior:
    """Abstract specification of a term prior

    Parameters
    ----------
    name : str
        Name of prior distribution. Must be the name of a PyMC distribution
        (e.g., `"Normal"`, `"Bernoulli"`, etc.)
    auto_scale : bool, optional
        Whether to adjust the parameters of the prior or use them as passed. Default to `True`.
    dist : pymc.Distribution or callable or None, optional
        A callable that returns a valid PyMC distribution. The signature must contain `name`,
        `dims`, and `shape`, as well as its own keyworded arguments.
    noncentered : bool or None, optional
        Per-prior override for non-centered parameterization of group-specific terms. When
        `None` (default), the value of `Model.noncentered` is used. When `True` or `False`,
        this prior's setting overrides the model-level default. Only meaningful on the prior
        of a group-specific term; ignored otherwise. `noncentered=True` still requires a
        Normal prior with a random `sigma` hyperprior; `noncentered=False` works for any prior.
    kwargs : dict
        Optional keywords specifying the parameters of the named distribution.
    """

    def __init__(self, name, auto_scale=True, dist=None, noncentered=None, **kwargs):
        self.name = name
        self.auto_scale = auto_scale
        self.args = {}
        self.update(**kwargs)
        self.dist = dist
        self.noncentered = noncentered

    def update(self, **kwargs):
        """Update the arguments of the prior with additional arguments

        Parameters
        ----------
        kwargs : dict
            Optional keyword arguments to add to prior args.
        """
        # The backend expect numpy arrays, so make sure all numeric values are represented as such.
        kwargs_ = {}
        for key, val in kwargs.items():
            if isinstance(val, (int, float)):
                val = np.array(val, dtype=pytensor.config.floatX)  # pylint: disable = no-member
            elif isinstance(val, np.ndarray):
                val = val.squeeze().astype(pytensor.config.floatX)  # pylint: disable = no-member
            kwargs_[key] = val
        self.args.update(kwargs_)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.__dict__ == other.__dict__

    def __str__(self):
        args = ", ".join(
            [
                f"{k}: {format_arg(v, 4)}" if not isinstance(v, type(self)) else f"{k}: {v}"
                for k, v in self.args.items()
            ]
        )
        if self.noncentered is not None:
            args = f"{args}, noncentered: {self.noncentered}" if args else f"noncentered: {self.noncentered}"
        return f"{self.name}({args})"

    def __repr__(self):
        return self.__str__()


def format_arg(value, decimals):
    try:
        outcome = np.round(value, decimals)
    except:  # pylint: disable = bare-except
        try:
            outcome = value.name
        except:  # pylint: disable = bare-except
            outcome = value
    return outcome
