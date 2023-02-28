import numpy as np

import formulae.terms

from bambi.terms.base import BaseTerm, VALID_PRIORS

GP_VALID_PRIORS = tuple(value for value in VALID_PRIORS if value is not None)

# pylint: disable = invalid-name
class HSGPTerm(BaseTerm):
    def __init__(self, term, prior, prefix=None):
        """Create a term for a HSGP model component

        Parameters
        ----------
        term : formulae.terms.terms.Term
            A term that was created with ``hsgp(...)``. The caller is an instance of ``HSGP()``.
        prior : dict
            The keys are the names of the parameters of the covariance function and the values are
            instances of ``bambi.Prior`` or other values that are accepted by the covariance
            function.
        prefix : str
            It is used to indicate the term belongs to the component of a non-parent parameter.
            Defaults to ``None``.
        """
        self.term = term
        self.prior = prior
        self.prefix = prefix
        self.hsgp_attributes = get_hsgp_attributes(term)
        self.hsgp = None

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Term)
        self._term = value

    @property
    def data(self):
        return self.term.data

    @property
    def data_centered(self):
        if self.by is None:
            output = self.data - self.mean
        else:
            _, levels_idx = np.unique(self.by, return_inverse=True)
            output = self.data - self.mean[levels_idx]
        return output

    @property
    def m(self):
        return np.atleast_1d(np.squeeze(self.hsgp_attributes["m"]))

    @property
    def L(self):
        """Get the value of L
        It's of shape (term.groups_n, term.variables_n). It's computed by variable and group.
        """
        if self.c is not None:
            if self.by is None:
                S = np.max(np.abs(self.data - self.mean), axis=0)
            else:
                S = np.zeros_like(self.c, dtype="float")
                levels = np.unique(self.by)
                for i, level in enumerate(levels):
                    S[i] = np.max(np.abs(self.data[self.by == level] - self.mean[i]), axis=0)
        return S * self.c

    @property
    def c(self):
        return self.hsgp_attributes["c"]

    @property
    def by(self):
        return self.hsgp_attributes["by"]

    @property
    def cov(self):
        return self.hsgp_attributes["cov"]

    @property
    def centered(self):
        return self.hsgp_attributes["centered"]

    @property
    def drop_first(self):
        return self.hsgp_attributes["drop_first"]

    @property
    def variables_n(self):
        return self.hsgp_attributes["variables_n"]

    @property
    def groups_n(self):
        return self.hsgp_attributes["groups_n"]

    @property
    def mean(self):
        return self.hsgp_attributes["mean"]

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        message = (
            "The priors for an HSGP term must be passed within a dictionary. "
            "Keys must the names of the parameters of the covariance function "
            "and values are instances of `bambi.Prior` or numeric constants."
        )
        if value is None:
            self._prior = value
        else:
            if not isinstance(value, dict):
                raise ValueError(message)
            for prior in value.values():
                assert isinstance(prior, GP_VALID_PRIORS), f"Prior must be one of {GP_VALID_PRIORS}"
            self._prior = value

    @property
    def coords(self):
        # NOTE: This has to depend on the 'by' argument.
        return {}

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def shape(self):
        return self.data.shape

    @property
    def categorical(self):
        return False

    @property
    def levels(self):
        return None


def get_hsgp_attributes(term):
    """Extract HSGP attributes from a model matrix term

    Parameters
    ----------
    term : formulae.terms.terms.Term
        The formulae term that creates the HSGP term.

    Returns
    -------
    dict
        The attributes that will be passed to pm.gp.HSGP
    """
    names = (
        "m",
        "L",
        "c",
        "by",
        "cov",
        "share_cov",
        "drop_first",
        "centered",
        "mean",
        "variables_n",
        "groups_n",
    )
    attrs_original = term.components[0].call.stateful_transform.__dict__
    attrs = {}
    for name in names:
        attrs[name] = attrs_original[name]
    return attrs
