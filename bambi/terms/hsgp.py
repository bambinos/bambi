# pylint: disable=no-member
from functools import partial

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
        properties_names = (
            "c",
            "by_levels",
            "cov",
            "share_cov",
            "scale",
            "iso",
            "centered",
            "drop_first",
            "variables_n",
            "groups_n",
            "mean",
            "maximum_distance",
        )
        self.__init_properties(properties_names)
        # When prior is none at initialization, then automatic priors are used
        self.automatic_priors = self.prior is None

    def __init_properties(self, names):
        """Initialize attributes as properties

        The properties are taken from the `self.hsgp_attributes` dictionary. This is to avoid
        writing many @property calls in the class definition.

        Parameters
        ----------
        names : Sequence[str]
            The names of the attributes taken from `self.hsgp_attributes`
        """

        def get(self, name):
            return self.hsgp_attributes[name]

        for name in names:
            get_partial = partial(get, name=name)
            setattr(self.__class__, name, property(get_partial))

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Term)
        self._term = value

    @property
    def data(self):
        if self.by_levels is None:
            data = self.term.data
        else:
            data = self.term.data[:, :-1]
        return data

    @property
    def shape(self):
        if self.by_levels is None:
            return self.term.data.shape
        else:
            return self.term.data[:, :-1].shape

    @property
    def data_centered(self):
        if self.by_levels is None:
            output = self.data - self.mean
        else:
            output = self.data - self.mean[self.by]
        return output

    @property
    def m(self):
        """Get the value of 'm', the number of basis vectors
        It's of shape (term.variables_n, ). It's computed by variable.
        """
        return np.atleast_1d(np.squeeze(self.hsgp_attributes["m"]))

    @property
    def L(self):
        """Get the value of L
        It's of shape (term.groups_n, term.variables_n). It's computed by variable and group.
        """
        if self.c is not None:
            if self.by_levels is None:
                S = np.max(np.abs(self.data - self.mean), axis=0)
            else:
                S = np.zeros_like(self.c, dtype="float")
                for i in range(len(self.by_levels)):
                    S[i] = np.max(np.abs(self.data_centered[self.by == i]), axis=0)
            return S * self.c
        return self.hsgp_attributes["L"]

    @property
    def by(self):
        if self.by_levels is not None:
            return self.term.data[:, -1].astype(int)
        return None

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
    def scale_predictors(self):
        # If scale is None, look if it uses automatic priors.
        #  If automatic priors are used, it will scale the data
        #  If automatic priors are not used, it won't scale the data
        if self.scale is None:
            return self.automatic_priors
        return self.scale

    @property
    def coords(self):
        # This handles univariate and multivariate cases
        coords = {f"{self.name}_weights_dim": np.arange(np.prod(self.m))}
        if self.by_levels is not None:
            coords[f"{self.name}_by"] = self.by_levels
        return coords

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

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
        "by_levels",
        "cov",
        "share_cov",
        "scale",
        "iso",
        "drop_first",
        "centered",
        "mean",
        "variables_n",
        "groups_n",
        "maximum_distance",
    )
    attrs_original = term.components[0].call.stateful_transform.__dict__
    attrs = {}
    for name in names:
        attrs[name] = attrs_original[name]
    return attrs
