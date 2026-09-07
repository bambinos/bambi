import formulae.terms

from bambi.terms.base import BaseTerm, VALID_PRIORS
from bambi.priors.prior import Prior


class GroupSpecificTerm(BaseTerm):  # pylint: disable=too-many-instance-attributes
    def __init__(self, term, prior, prefix=None, noncentered=True):
        self._hyperprior_alias = {}
        self.term = term
        self.prior = prior
        self.data = term.data
        self.group_index = self.invert_dummies(self.grouper)
        self.prefix = prefix
        self.noncentered = noncentered

    def invert_dummies(self, dummies):
        """Invert dummies
        For the sake of computational efficiency (i.e., to avoid lots of large matrix
        multiplications in the backend), invert the dummy-coding process and represent full-rank
        dummies as a vector of indices into the coefficients.

        Only used when `bmb.config.SPARSE_DOT` is `False`.
        """
        # NOTE: This asummes there's a single '1' per row, which is true.
        return dummies.argmax(1)

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.GroupSpecificTerm)
        self._term = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def kind(self):
        return self.term.kind

    @property
    def expr(self):
        return self.term.expr

    @property
    def factor(self):
        return self.term.factor

    @property
    def expr_name(self):
        return self.expr.name

    @property
    def factor_name(self):
        return self.factor.name

    @property
    def expr_kind(self):
        return self.expr.kind

    @property
    def is_intercept(self):
        return self.expr_kind == "intercept"

    @property
    def shape(self):
        return self.data.shape

    @property
    def categorical(self):
        # Determine if the expression is categorical
        if self.kind == "interaction":
            return any(component.kind == "categoric" for component in self.expr.components)
        return self.kind == "categoric"

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        # This does not check which argument has hyperprior (must be dispersion?)
        assert isinstance(value, VALID_PRIORS), f"Prior must be one of {VALID_PRIORS}"
        if isinstance(value, Prior):
            any_hyperprior = any(isinstance(x, Prior) for x in value.args.values())
            if not any_hyperprior:
                raise ValueError("Prior for group-specific terms must have hyperpriors")
        self._prior = value

    @property
    def groups(self):
        return self.term.groups

    @property
    def levels(self):
        return self.term.labels

    @property
    def predictor(self):
        return self.expr.data

    @property
    def grouper(self):
        return self.factor.data

    @property
    def hyperprior_alias(self):
        return self._hyperprior_alias

    @hyperprior_alias.setter
    def hyperprior_alias(self, values):
        assert all(isinstance(x, str) for x in values.keys())
        assert all(isinstance(x, str) for x in values.values())
        self._hyperprior_alias.update(values)

    def __str__(self):
        args = [f"groups: {self.groups}"]
        return self.make_str(args)
