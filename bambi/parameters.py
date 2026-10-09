from abc import ABC, abstractmethod

from bambi.defaults import get_default_prior
from bambi.priors.prior import Prior
from bambi.terms import CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm, SmoothTerm
from bambi.utils import is_hsgp_term, is_smooth_term


class Marginal(ABC):
    """A modeled quantity with a direct prior and no covariate formula."""

    def __init__(self, name, prior, spec):
        self.alias = None
        self.name = name
        self.prior = prior
        self.spec = spec

    @property
    def label(self):
        return self.alias or self.name

    def update_priors(self, value):
        self.prior = value

    @property
    @abstractmethod
    def default_prior(self):
        """Prior used when no explicit prior is supplied."""

    def build_priors(self):
        if isinstance(self.prior, Prior):
            self.prior.auto_scale = False
        elif isinstance(self.prior, (int, float)):
            return
        elif self.prior is not None:
            raise ValueError(f"'{self.prior}' is not a valid prior.")
        else:
            self.prior = self.default_prior
            if self.prior is None:
                raise ValueError(f"The parameter '{self.name}' needs a prior.")


class MarginalParameter(Marginal):
    """An observational-model parameter with a direct prior."""

    @property
    def default_prior(self):
        return self.spec.family.default_priors.get(self.name)


class MarginalCoefficient(Marginal):
    """A scalar nonlinear coefficient with a direct prior."""

    @property
    def default_prior(self):
        kind = "common" if self.spec.auto_scale else "common_flat"
        return get_default_prior(kind)


class Conditional(ABC):
    """A quantity defined by an additive design or a nonlinear expression.

    Predictor terms, priors, and coefficient ownership are shared by observational-model
    parameters and nonlinear coefficients. Only parameters can be the likelihood parent.

    Parameters
    ----------
    name : str
        Original quantity name.
    design : formulae.matrices.DesignMatrices or None
        Additive design matrices, or ``None`` for an expression-defined quantity.
    priors : dict
        Priors for terms in an additive design.
    spec : Model
        Model specification that owns the quantity.
    expression : NonlinearExpression or None, optional
        Expression defining the quantity, or ``None`` for an additive design.
    data_names : Collection of str, optional
        Observed data columns referenced directly by the expression.
    nonlinear_coefficients : dict, optional
        Shared coefficient objects referenced directly by the expression.
    """

    @property
    @abstractmethod
    def prefix(self):
        """Prefix for the quantity's term names."""

    def __init__(
        self,
        name,
        design,
        priors,
        spec,
        expression=None,
        data_names=(),
        nonlinear_coefficients=None,
    ):
        self.terms = {}
        self.alias = None
        self.name = name
        self.design = design
        self.spec = spec
        self.expression = expression
        self.data_names = tuple(data_names)
        self.nonlinear_coefficients = nonlinear_coefficients or {}

        if (design is None) == (expression is None):
            raise ValueError(
                "A conditional quantity must have either an additive design or a nonlinear "
                "expression."
            )

        if self.design is not None and self.design.common:
            self.add_common_terms(priors)
            self.add_hsgp_terms(priors)
            self.add_smooth_terms(priors)

        if self.design is not None and self.design.group:
            self.add_group_specific_terms(priors)

    @classmethod
    def from_design(cls, name, design, priors, spec, **kwargs):
        """Create a conditional quantity backed by additive design matrices."""
        return cls(name, design, priors, spec, **kwargs)

    @classmethod
    def from_expression(cls, name, expression, data_names, spec, **kwargs):
        """Create a conditional quantity backed by a nonlinear expression."""
        return cls(name, None, {}, spec, expression=expression, data_names=data_names, **kwargs)

    @property
    def is_nonlinear(self):
        """Whether this parameter is defined by a nonlinear expression.

        Returns
        -------
        bool
            ``True`` for expression-defined parameters and ``False`` for additive parameters.

        Examples
        --------
        ``model.conditional_parameters["mu"].is_nonlinear`` distinguishes a nonlinear parent
        from one constructed with an ordinary additive formula.
        """
        return self.expression is not None

    @property
    def label(self):
        return self.alias or self.name

    @property
    def center_predictors(self):
        return self.spec.center_predictors

    def add_common_terms(self, priors):
        for name, term in self.design.common.terms.items():
            if is_hsgp_term(term):
                continue

            if is_smooth_term(term):
                continue

            prior = priors.get(name, priors.get("common", None))
            if isinstance(prior, Prior):
                if any(isinstance(x, Prior) for x in prior.args.values()):
                    raise ValueError(
                        f"Trying to set hyperprior on '{name}'. "
                        "Can't set a hyperprior on common effects."
                    )

            if term.kind == "offset":
                self.terms[name] = OffsetTerm(term, self.prefix)
            else:
                self.terms[name] = CommonTerm(term, prior, self.prefix)

    def add_group_specific_terms(self, priors):
        if isinstance(self.spec.noncentered, dict):
            noncentered = self.spec.noncentered.get(self.name, True)
        else:
            noncentered = self.spec.noncentered

        for name, term in self.design.group.terms.items():
            if is_smooth_term(term.expr):
                raise NotImplementedError("Group-specific smooths are not supported yet.")
            prior = priors.get(name, priors.get("group_specific", None))
            self.terms[name] = GroupSpecificTerm(term, prior, self.prefix, noncentered)

    def add_hsgp_terms(self, priors):
        for name, term in self.design.common.terms.items():
            if is_hsgp_term(term):
                prior = priors.get(name, None)
                self.terms[name] = HSGPTerm(term, prior, self.prefix)

    def add_smooth_terms(self, priors):
        for name, term in self.design.common.terms.items():
            if is_smooth_term(term):
                prior = priors.get(name, None)
                term = SmoothTerm(term, prior, self.prefix)
                if term.has_intercept and "Intercept" in self.design.common.terms:
                    raise ValueError("Use center=True for a smooth in a model with an intercept.")
                self.terms[name] = term

    def build_priors(self):
        for term in self.terms.values():
            if isinstance(term, OffsetTerm):
                continue

            if isinstance(term, SmoothTerm):
                if term.prior is None:
                    term.prior = get_default_prior(
                        "smooth", term=term, auto_scale=self.spec.auto_scale
                    )
                else:
                    for prior in term.prior.values():
                        if isinstance(prior, Prior):
                            prior.auto_scale = False
                continue

            if isinstance(term, HSGPTerm):
                if term.prior is None:
                    term.prior = get_default_prior("hsgp", cov_func=term.cov)
                else:
                    for prior in term.prior.values():
                        if isinstance(prior, Prior):
                            prior.auto_scale = False
                continue

            if isinstance(term, GroupSpecificTerm):
                kind = "group_specific"
            elif isinstance(term, CommonTerm) and term.kind == "intercept":
                kind = "intercept"
            else:
                kind = "common"

            if term.prior is None:
                kind += "" if self.spec.auto_scale else "_flat"
                term.prior = get_default_prior(kind)
            elif isinstance(term.prior, Prior):
                term.prior.auto_scale = False
            else:
                raise ValueError("'prior' must be instance of Prior or `None`.")

    def update_priors(self, priors):
        common = priors.get("common")
        group_specific = priors.get("group_specific")

        for name, term in self.terms.items():
            if name in priors:
                term.prior = priors[name]
            elif isinstance(term, GroupSpecificTerm):
                if group_specific is not None:
                    term.prior = group_specific
            elif isinstance(term, CommonTerm) and term.kind != "offset":
                if common is not None:
                    term.prior = common

    @property
    def intercept_term(self):
        for term in self.terms.values():
            if isinstance(term, CommonTerm) and term.kind == "intercept":
                return term
        return None

    @property
    def common_terms(self):
        return {
            name: term
            for name, term in self.terms.items()
            if isinstance(term, CommonTerm)
            and not isinstance(term, OffsetTerm)
            and not isinstance(term, SmoothTerm)
            and term.kind != "intercept"
        }

    @property
    def group_specific_terms(self):
        return {
            name: term for name, term in self.terms.items() if isinstance(term, GroupSpecificTerm)
        }

    @property
    def offset_terms(self):
        return {name: term for name, term in self.terms.items() if isinstance(term, OffsetTerm)}

    @property
    def hsgp_terms(self):
        return {name: term for name, term in self.terms.items() if isinstance(term, HSGPTerm)}

    @property
    def smooth_terms(self):
        return {name: term for name, term in self.terms.items() if isinstance(term, SmoothTerm)}


class ConditionalParameter(Conditional):
    """An observational-model parameter defined by a formula."""

    def __init__(self, name, design, priors, spec, is_parent, **kwargs):
        self.is_parent = is_parent
        super().__init__(name, design, priors, spec, **kwargs)

    @property
    def prefix(self):
        return "" if self.is_parent else self.name


class ConditionalCoefficient(Conditional):
    """A nonlinear coefficient defined by covariates or other modeled quantities."""

    @property
    def prefix(self):
        return self.name
