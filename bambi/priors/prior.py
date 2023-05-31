import numpy as np
import pytensor


class Prior:
    """Abstract specification of a term prior.

    Parameters
    ----------
    name : str
        Name of prior distribution. Must be the name of a PyMC distribution
        (e.g., ``"Normal"``, ``"Bernoulli"``, etc.)
    auto_scale: bool
        Whether to adjust the parameters of the prior or use them as passed. Default to ``True``.
    kwargs : dict
        Optional keywords specifying the parameters of the named distribution.
    dist : pymc.distributions.distribution.DistributionMeta or callable
        A callable that returns a valid PyMC distribution. The signature must contain ``name``,
        ``dims``, and ``shape``, as well as its own keyworded arguments.
    """

    def __init__(self, name, auto_scale=True, dist=None, **kwargs):
        self.name = name
        self.auto_scale = auto_scale
        self.args = {}
        self.update(**kwargs)
        self.dist = dist

    def update(self, **kwargs):
        """Update the arguments of the prior with additional arguments.

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
        return f"{self.name}({args})"

    def __repr__(self):
        return self.__str__()


def format_arg(value, decimals):
    try:
        outcome = np.round_(value, decimals)
    except:  # pylint: disable = bare-except
        try:
            outcome = value.name
        except:  # pylint: disable = bare-except
            outcome = value
    return outcome
