from bambi.defaults import get_default_prior
from bambi.priors.prior import Prior
from bambi.terms import CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm
from bambi.utils import is_hsgp_term


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
    def __init__(self, name, design, priors, spec, is_parent):
        self.terms = {}
        self.alias = None
        self.name = name
        self.design = design
        self.spec = spec
        self.is_parent = is_parent
        self.prefix = "" if is_parent else name

        if self.design.common:
            self.add_common_terms(priors)
            self.add_hsgp_terms(priors)

        if self.design.group:
            self.add_group_specific_terms(priors)

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

            prior = priors.pop(name, priors.get("common", None))
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
            prior = priors.pop(name, priors.get("group_specific", None))
            self.terms[name] = GroupSpecificTerm(term, prior, self.prefix, noncentered)

    def add_hsgp_terms(self, priors):
        for name, term in self.design.common.terms.items():
            if is_hsgp_term(term):
                prior = priors.pop(name, None)
                self.terms[name] = HSGPTerm(term, prior, self.prefix)

    def build_priors(self):
        for term in self.terms.values():
            if isinstance(term, GroupSpecificTerm):
                kind = "group_specific"
            elif isinstance(term, CommonTerm) and term.kind == "intercept":
                kind = "intercept"
            elif isinstance(term, OffsetTerm):
                continue
            elif isinstance(term, HSGPTerm):
                if term.prior is None:
                    term.prior = get_default_prior("hsgp", cov_func=term.cov)
                continue
            else:
                kind = "common"

            term.prior = prepare_prior(term.prior, kind, self.spec.auto_scale)

    def update_priors(self, priors):
        for name, value in priors.items():
            self.terms[name].prior = value

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


def prepare_prior(prior, kind, auto_scale):
    if prior is None:
        if auto_scale:
            prior = get_default_prior(kind)
        else:
            prior = get_default_prior(kind + "_flat")
    elif isinstance(prior, Prior):
        prior.auto_scale = False
    else:
        raise ValueError("'prior' must be instance of Prior or `None`.")
    return prior
