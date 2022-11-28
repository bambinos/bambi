import numpy as np

from bambi.families import univariate, multivariate
from bambi.priors import Prior
from bambi.terms import CommonTerm, GroupSpecificTerm, OffsetTerm, ResponseTerm


class ConstantComponent:
    def __init__(self, name, prior, spec):
        self.name = name
        self.prior = prior
        self.spec = spec


class DistributionalComponent:
    """_summary_
    response_kind = ["data", "parameter"]
    """

    def __init__(self, design, priors, response_name, response_kind, spec):
        self.terms = {}
        self.design = design
        self.response_name = response_name
        self.response_kind = response_kind
        self.response_term = None
        self.spec = spec

        if self.response_kind == "data":
            self.suffix = ""
        else:
            self.suffix = response_name

        if self.design.common:
            self.add_common_terms(priors)

        if self.design.group:
            self.add_group_specific_terms(priors)

        if self.design.response:
            self.add_response_term(priors)

    def add_common_terms(self, priors):
        for name, term in self.design.common.terms.items():
            name_with_suffix = with_suffix(name, self.suffix)
            term.name = name_with_suffix  # Update the name of the term
            data = self.design.common[name]
            prior = priors.pop(name, priors.get("common", None))
            if isinstance(prior, Prior):
                any_hyperprior = any(isinstance(x, Prior) for x in prior.args.values())
                if any_hyperprior:
                    raise ValueError(
                        f"Trying to set hyperprior on '{name_with_suffix}'. "
                        "Can't set a hyperprior on common effects."
                    )
            if term.kind == "offset":
                self.terms[name] = OffsetTerm(term, data)
            else:
                self.terms[name] = CommonTerm(term, prior)

    def add_group_specific_terms(self, priors):
        for name, term in self.design.group.terms.items():
            term.name = with_suffix(name, self.suffix)
            prior = priors.pop(name, priors.get("group_specific", None))
            self.terms[name] = GroupSpecificTerm(term, prior)

    def add_response_term(self):
        """Add a response (or outcome/dependent) variable to the model."""
        response = self.design.response

        if hasattr(response.term.term.components[0], "reference"):
            reference = response.term.term.components[0].reference
        else:
            reference = None

        # NOTE: This is a historical feature.
        # I'm not sure how many family specific checks we should add in this type of places now
        if reference is not None and not isinstance(self.family, univariate.Bernoulli):
            raise ValueError("Index notation for response is only available for 'bernoulli' family")

        if isinstance(self.spec.family, univariate.Bernoulli):
            if response.kind == "categoric" and response.levels is None and reference is None:
                raise ValueError("Categoric response must be binary for 'bernoulli' family.")
            if response.kind == "numeric" and not all(np.isin(response.design_matrix, [0, 1])):
                raise ValueError("Numeric response must be all 0 and 1 for 'bernoulli' family.")

        self.response_term = ResponseTerm(response, self.spec.family)

    def build_priors(self):
        ...

    def set_alias(self):
        ...

    @property
    def common_terms(self):
        ...

    @property
    def group_specific_terms(self):
        ...

    @property
    def suffix(self):
        return self._suffix

    @suffix.setter
    def suffix(self, value):
        assert isinstance(value, str), "'.suffix' must be a string"
        self._suffix = value


def with_suffix(value, suffix):
    if suffix:
        return f"{value}_{suffix}"
    return value
