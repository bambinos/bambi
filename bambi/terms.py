import numpy as np


class ResponseTerm:
    """Representation of a single response model term.

    Parameters
    ----------
    term: formulae.ResponseVector
        An object describing the response of the model,
        as returned by ``formulae.design_matrices().response``
    family : Family
        An ``Family`` object.
    """

    def __init__(self, term, family):
        self.name = term.name
        self.data = term.design_vector
        self.categorical = term.type == "categoric"
        self.success_event = term.success if term.success is not None else 1
        self.constant = np.var(self.data) == 0
        self.family = family

        if family.name == "bernoulli":
            if not all(np.isin(self.data, ([0, 1]))):
                raise ValueError("Numeric response must be all 0 and 1 for 'bernoulli' family.")

    def __str__(self):
        args = [
            f"name: {self.name}",
            f"family: {self.family.name}",
            f"shape: {self.data.squeeze().shape}",
        ]
        if self.family.name == "bernoulli":
            args += [f"succes: {self.success_event}"]

        return f"{self.__class__.__name__}({', '.join(args)})"

    def __repr__(self):
        return self.__str__()


class Term:
    """Representation of a single (common) model term.

    Parameters
    ----------
    name : str
        Name of the term.
    term_dict: dict
        A dictionary describing the components of the term. These can be found in
        ``formulae.design_matrices().common.terms_info``
    data : ndarray
        The term values.
    prior : Prior
        A specification of the prior(s) to use. An instance of class ``priors.Prior``.
    """

    group_specific = False

    def __init__(self, name, term_dict, data, prior=None):
        self.name = name
        self.data = data
        self.prior = prior
        self.type = term_dict["type"]
        self.levels = term_dict["full_names"]
        self.categorical = False

        # If the term has one component, it's categorical if the component is categorical.
        # If the term has more than one component (i.e. it is an interaction), it's categorical if
        # at least one of the components is categorical.
        if self.type == "interaction":
            if any((term["type"] == "categoric" for term in term_dict["terms"].values())):
                self.categorical = True
        else:
            self.categorical = self.type == "categoric"

        # Flag constant terms
        if self.categorical and len(term_dict["levels"]) == 1 and (data == data[0]).all():
            raise ValueError(f"The term '{name}' has only 1 category!")

        if not self.categorical and self.type != "intercept" and np.all(data == data[0]):
            raise ValueError(f"The term '{name}' is constant!")

        # Flag cell-means terms (i.e., full-rank coding), which receive special priors
        # To flag intercepts we use `self.type`
        self.is_cell_means = self.categorical and (self.data.sum(1) == 1).all()

        # Obtain pymc coordinates, only for categorical components of a term.
        # A categorical component can have up to two coordinates if it is including with both
        # reduced and full rank encodings.
        self.pymc_coords = {}
        if self.categorical:
            name = self.name + "_coord"
            if self.type == "interaction":
                self.pymc_coords[name] = term_dict["levels"]
            elif term_dict["encoding"] == "full":
                self.pymc_coords[name] = term_dict["levels"]
            else:
                self.pymc_coords[name] = term_dict["levels"][1:]

    def __str__(self):
        args = [
            f"name: {self.name}",
            f"prior: {self.prior}",
            f"type: {self.type}",
            f"shape: {self.data.squeeze().shape}",
            f"categorical: {self.categorical}",
        ]
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
    name : str
        Name of the term.
    term: dict
        A dictionary describing the components of the term. These can be found in
        ``formulae.design_matrices().group.terms_info``
    data : (DataFrame, Series, ndarray)
        The term values.
    prior : Prior
        A specification of the prior(s) to use. An instance of class ``priors.Prior``.
    """

    group_specific = True

    def __init__(self, name, term, data, prior=None):
        self.name = name
        self.data = data
        self.prior = prior
        self.type = term["type"]
        self.groups = term["groups"]
        self.levels = term["full_names"]
        self.grouper = term["Ji"]
        self.predictor = term["Xi"]
        self.group_index = self.invert_dummies(self.grouper)
        self.categorical = False

        # Determine if the expression is categorical
        if self.type == "interaction":
            if any((t["type"] == "categoric" for t in term["terms"].values())):
                self.categorical = True
        else:
            self.categorical = self.type == "categoric"

        # Determine if the term represents cell-means encoding.
        self.is_cell_means = self.categorical and (self.data.sum(1) == 1).all()

        # Used in pymc3 model coords to label coordinates appropiately
        self.pymc_coords = {}
        # Group is always a coordinate added to the model.
        expr, factor = self.name.split("|")
        self.pymc_coords[factor + "_coord_group_factor"] = self.groups

        if self.categorical:
            name = expr + "_coord_group_expr"
            if self.type == "interaction":
                self.pymc_coords[name] = term["levels"]
            elif term["encoding"] == "full":
                self.pymc_coords[name] = term["levels"]
            else:
                self.pymc_coords[name] = term["levels"][1:]

    def invert_dummies(self, dummies):
        """
        For the sake of computational efficiency (i.e., to avoid lots of large matrix
        multiplications in the backends), invert the dummy-coding process and represent full-rank
        dummies as a vector of indices into the coefficients.
        """
        vec = np.zeros(len(dummies), dtype=int)
        for i in range(1, dummies.shape[1]):
            vec[dummies[:, i] == 1] = i
        return vec

    def __str__(self):
        args = [
            f"name: {self.name}",
            f"prior: {self.prior}",
            f"groups: {self.groups}",
            f"type: {self.type}",
            f"shape: {self.data.squeeze().shape}",
            f"categorical: {self.categorical}",
        ]
        if self.categorical:
            args += [f"levels: {self.levels}"]

        return f"{self.__class__.__name__}({', '.join(args)})"

    def __repr__(self):
        return self.__str__()
