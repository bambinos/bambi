import numpy as np


class Prior:
    """Abstract specification of a term prior.

    Parameters
    ----------
    name : str
        Name of prior distribution. Must be the name of a PyMC3 distribution
        (e.g., ``'Normal'``, ``'Bernoulli'``, etc.)
    auto_scale: bool
        Whether to adjust the parameters of the prior or use them as passed. Default to ``True``.
    scale: num or str
    kwargs : dict
        Optional keywords specifying the parameters of the named distribution.
    """

    def __init__(self, name, auto_scale=True, scale=None, **kwargs):
        self.name = name
        self.auto_scale = auto_scale
        self.scale = scale
        self.args = {}
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update the model arguments with additional arguments.

        Parameters
        ----------
        kwargs : dict
            Optional keyword arguments to add to prior args.
        """
        # Backends expect numpy arrays, so make sure all numeric values are represented as such.
        kwargs_ = {}
        for key, val in kwargs.items():
            if isinstance(val, (int, float)):
                val = np.array(val)
            elif isinstance(val, np.ndarray):
                val = val.squeeze()
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
                f"{k}: {np.round_(v, 4)}" if not isinstance(v, Prior) else f"{k}: {v}"
                for k, v in self.args.items()
            ]
        )
        return f"{self.name}({args})"

    def __repr__(self):
        return self.__str__()
