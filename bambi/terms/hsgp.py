import formulae.terms

from bambi.terms.base import BaseTerm

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
    def m(self):
        return self.hsgp_attributes["m"]

    @property
    def L(self):
        return self.hsgp_attributes["L"]

    @property
    def c(self):
        return self.hsgp_attributes["c"]

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
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        # assert isinstance(value, VALID_PRIORS), f"Prior must be one of {VALID_PRIORS}"
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
    names = ("m", "L", "c", "cov", "drop_first", "centered", "by")
    attrs_original = term.components[0].call.stateful_transform.__dict__
    attrs = {}
    for name in names:
        attrs[name] = attrs_original[name]
    return attrs


# NOTE: Regarding optimizations of the basis
# When we have multiple GPs, one per level of a covariate, we cannot safely assume that 'X'
# is the same in all cases.
# However, we could test this. And if we find out that the observed values are the same
# for all groups, then we can apply some optimizations
