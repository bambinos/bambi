import numpy as np

from bambi.families.multivariate import Categorical, Multinomial
from bambi.families.univariate import Bernoulli
from bambi.utils import extract_argument_names, extra_namespace


class ResponseTerm:
    """Representation of a single response model term.

    Parameters
    ----------
    term : formulae.ResponseMatrix
        An object describing the response of the model,
        as returned by ``formulae.design_matrices().response``
    spec : bambi.Model
        The model where this response term is used.
    """

    def __init__(self, term, spec):
        self.name = term.name
        self.categorical = term.kind == "categoric"
        self.reference = None
        self.levels = None  # Not None for categorical variables
        self.binary = None  # Not None for categorical variables (either True or False)
        self.success = None  # Not None for binary variables (either True or False)
        self.alias = None
        self.data = None

        if self.categorical:
            if term.levels is None:
                self.binary = True
            else:
                self.levels = term.levels
                self.binary = len(term.levels) == 2

            if self.binary:
                self.success = get_success_level(term.term.term)
                if term.design_matrix.ndim == 1:
                    self.data = term.design_matrix
                else:
                    idx = self.levels.index(self.success)
                    self.data = term.design_matrix[:, idx]
            # Applies to the categorical family
            else:
                self.reference = get_reference_level(term.term.term)
                self.data = np.nonzero(term.design_matrix)[1]
        elif isinstance(spec.family, Bernoulli):
            # We've already checked the values are all 0 and 1
            self.success = 1
            self.data = term.design_matrix
        else:
            self.data = term.design_matrix

        # We use pymc coords when the response is multi-categorical.
        # These help to give the appropriate shape to coefficients and make the resulting
        # InferenceData object much cleaner
        self.coords = {}
        if isinstance(spec.family, Categorical):
            name = self.name + "_dim"
            self.coords[name] = [level for level in term.levels if level != self.reference]
        elif isinstance(spec.family, Multinomial):
            name = self.name + "_dim"
            labels = extract_argument_names(self.name, list(extra_namespace))
            if labels:
                self.levels = labels
            else:
                self.levels = [str(level) for level in range(self.data.shape[1])]
            labels = self.levels[1:]
            self.coords[name] = labels
        # TBD: Continue here when we add general multivariate responses.

    def set_alias(self, value):
        self.alias = value

    def __str__(self):
        args = [
            f"name: {self.name}",
            f"shape: {self.data.squeeze().shape}",
        ]

        if self.alias:
            args[0] = f"{args[0]} (alias: {self.alias})"

        if self.categorical:
            args += [f"levels: {self.levels}"]
            if self.binary:
                args += [f"success: {self.success}"]
            else:
                args += [f"reference: {self.reference}"]

        return f"{self.__class__.__name__}({', '.join(args)})"

    def __repr__(self):
        return self.__str__()


class Term:
    """Representation of a single (common) model term.

    Parameters
    ----------
    name: str
        Name of the term.
    term_dict: dict
        A dictionary describing the components of the term. These can be found in
        ``formulae.design_matrices().common.terms_info``
    data: ndarray
        The term values.
    prior: Prior
        A specification of the prior(s) to use. An instance of class ``priors.Prior``.
    """

    group_specific = False

    def __init__(self, name, term, data, prior=None):
        self.name = name
        self.data = data
        self.prior = prior
        self.kind = term.kind
        self.levels = term.labels
        self.categorical = False
        self.term = term
        self.alias = None

        # If the term has one component, it's categorical if the component is categorical.
        # If the term has more than one component (i.e. it is an interaction), it's categorical if
        # at least one of the components is categorical.
        if self.kind == "interaction":
            if any(component.kind == "categoric" for component in term.components):
                self.categorical = True
        else:
            self.categorical = self.kind == "categoric"

        # Flag constant terms
        if self.categorical and len(term.levels) == 1 and (data == data[0]).all():
            raise ValueError(f"The term '{name}' has only 1 category!")

        if not self.categorical and self.kind != "intercept" and np.all(data == data[0]):
            raise ValueError(f"The term '{name}' is constant!")

        # Flag cell-means terms (i.e., full-rank coding), which receive special priors
        # To flag intercepts we use `self.kind`
        self.is_cell_means = self.categorical and (self.data.sum(1) == 1).all()

        # Obtain pymc coordinates, only for categorical components of a term.
        # A categorical component can have up to two coordinates if it is including with both
        # reduced and full rank encodings.
        self.coords = {}
        if self.categorical:
            name = self.name + "_dim"
            self.coords[name] = term.levels
        elif self.data.ndim > 1 and self.data.shape[1] > 1:
            name = self.name + "_dim"
            self.coords[name] = np.arange(self.data.shape[1])

    def set_alias(self, value):
        self.alias = value

    def __str__(self):
        args = [
            f"name: {self.name}",
            f"prior: {self.prior}",
            f"kind: {self.kind}",
            f"shape: {self.data.squeeze().shape}",
            f"categorical: {self.categorical}",
        ]

        if self.alias:
            args[0] = f"{args[0]} (alias: {self.alias})"

        if self.categorical:
            args += [f"levels: {self.levels}"]

        return f"{self.__class__.__name__}({', '.join(args)})"

    def __repr__(self):
        return self.__str__()


class GroupSpecificTerm:
    # pylint: disable=too-many-instance-attributes
    """Representation of a single (group specific) model term.

    Parameters
    ----------
    name: str
        Name of the term.
    term: dict
        A dictionary describing the components of the term. These can be found in
        ``formulae.design_matrices().group.terms_info``
    data: (DataFrame, Series, ndarray)
        The term values.
    prior: Prior
        A specification of the prior(s) to use. An instance of class ``priors.Prior``.
    """

    group_specific = True

    def __init__(self, name, term, data, prior=None):
        self.categorical = False
        self.alias = None
        self.hyperprior_alias = {}

        self.name = name
        self.data = data
        self.prior = prior
        self.kind = term.kind
        self.groups = term.groups
        self.levels = term.labels
        self.grouper = term.factor.data
        self.predictor = term.expr.data
        self.group_index = self.invert_dummies(self.grouper)
        self.term = term

        # Determine if the expression is categorical
        if self.kind == "interaction":
            if any(component.kind == "categoric" for component in term.expr.components):
                self.categorical = True
        else:
            self.categorical = self.kind == "categoric"

        # Determine if the term represents cell-means encoding.
        self.is_cell_means = self.categorical and (self.data.sum(1) == 1).all()

        # Used in pymc model coords to label coordinates appropriately
        self.coords = {}

        # Group is always a coordinate added to the model.
        expr, factor = self.name.split("|")
        self.coords[factor + "__factor_dim"] = self.groups

        if self.categorical:
            self.coords[expr + "__expr_dim"] = term.expr.levels
        elif self.predictor.ndim == 2 and self.predictor.shape[1] > 1:
            self.coords[expr + "__expr_dim"] = [str(i) for i in range(self.predictor.shape[1])]

    def invert_dummies(self, dummies):
        """
        For the sake of computational efficiency (i.e., to avoid lots of large matrix
        multiplications in the backend), invert the dummy-coding process and represent full-rank
        dummies as a vector of indices into the coefficients.
        """
        vec = np.zeros(len(dummies), dtype=int)
        for i in range(1, dummies.shape[1]):
            vec[dummies[:, i] == 1] = i
        return vec

    def set_alias(self, value):
        self.alias = value

    def set_hyperprior_alias(self, name, value):
        self.hyperprior_alias.update({name: value})

    def __str__(self):
        args = [
            f"name: {self.name}",
            f"prior: {self.prior}",
            f"groups: {self.groups}",
            f"type: {self.kind}",
            f"shape: {self.data.squeeze().shape}",
            f"categorical: {self.categorical}",
        ]

        if self.alias:
            args[0] = f"{args[0]} (alias: {self.alias})"

        if self.categorical:
            args += [f"levels: {self.levels}"]

        return f"{self.__class__.__name__}({', '.join(args)})"

    def __repr__(self):
        return self.__str__()


# pylint: disable = protected-access
def get_reference_level(term):
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return None

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]


# pylint: disable = protected-access
def get_success_level(term):
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return term.components[0].reference

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]
