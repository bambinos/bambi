from bambi.defaults import get_default_prior
from bambi.priors.prior import Prior
from bambi.terms import CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm, SmoothTerm
from bambi.utils import is_hsgp_term, is_smooth_term


class MarginalParameter:
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


class ConditionalParameter:
    """Description of a quantity conditional on covariates or other modeled parameters.

    A conditional parameter has either an additive design and terms, or a nonlinear expression
    with the coefficient descriptions used directly by that expression.

    Parameters
    ----------
    name : str
        Original parameter name.
    design : formulae.matrices.DesignMatrices or None
        Additive design matrices. Must be ``None`` for an expression-defined parameter.
    priors : dict
        Priors for terms in an additive design.
    spec : Model
        Model specification that owns the parameter.
    is_parent : bool
        Whether this is the likelihood's parent parameter.
    expression : NonlinearExpression or None, optional
        Expression that defines a nonlinear parameter. Must be ``None`` for an additive parameter.
    data_names : Collection of str, optional
        Observed data columns referenced directly by ``expression``.
    nonlinear_coefficients : dict of str to ConditionalParameter, optional
        Additive coefficients referenced directly by ``expression``.

    Examples
    --------
    A model exposes additive and expression-defined parameters through the same interface.

    >>> import bambi as bmb
    >>> import pandas as pd
    >>> data = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]})
    >>> model = bmb.Model(bmb.Formula("y ~ a * x", nlpars=("a",)), data)
    >>> model.conditional_parameters["mu"].is_nonlinear
    True
    """

    def __init__(
        self,
        name,
        design,
        priors,
        spec,
        is_parent,
        expression=None,
        data_names=(),
        nonlinear_coefficients=None,
    ):
        self.terms = {}
        self.alias = None
        self.name = name
        self.design = design
        self.spec = spec
        self.is_parent = is_parent
        self.prefix = "" if is_parent else name
        self.expression = expression
        self.data_names = tuple(data_names)
        self.nonlinear_coefficients = nonlinear_coefficients or {}

        if (design is None) == (expression is None):
            raise ValueError(
                "A conditional parameter must have either an additive design or a nonlinear "
                "expression."
            )

        if self.design is not None and self.design.common:
            self.add_common_terms(priors)
            self.add_hsgp_terms(priors)
            self.add_smooth_terms(priors)

        if self.design is not None and self.design.group:
            self.add_group_specific_terms(priors)

    @classmethod
    def from_design(cls, name, design, priors, spec, is_parent):
        """Create a parameter backed by additive design matrices.

        Parameters
        ----------
        name : str
            Original parameter name.
        design : formulae.matrices.DesignMatrices
            Additive design matrices.
        priors : dict
            Priors for terms in the design.
        spec : Model
            Model specification that owns the parameter.
        is_parent : bool
            Whether this is the likelihood's parent parameter.

        Returns
        -------
        ConditionalParameter
            Parameter populated with terms from ``design``.

        Examples
        --------
        ``Model`` uses this constructor for ordinary conditional parameters such as ``mu`` in
        ``Model("y ~ x", data)``.
        """
        return cls(name, design, priors, spec, is_parent)

    @classmethod
    def from_expression(cls, name, expression, data_names, spec, is_parent):
        """Create a parameter backed by a nonlinear expression.

        Parameters
        ----------
        name : str
            Original parameter name.
        expression : NonlinearExpression
            Expression that defines the parameter.
        data_names : Collection of str
            Observed data columns referenced directly by ``expression``.
        spec : Model
            Model specification that owns the parameter.
        is_parent : bool
            Whether this is the likelihood's parent parameter.

        Returns
        -------
        ConditionalParameter
            Parameter populated with ``expression`` and its observed inputs.

        Examples
        --------
        ``Model`` uses this constructor for ``mu`` in
        ``Formula("y ~ a * x", nlpars=("a",))``.
        """
        return cls(
            name,
            None,
            {},
            spec,
            is_parent,
            expression=expression,
            data_names=data_names,
        )

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
