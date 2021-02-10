import numpy as np

class ResponseTerm:
    """Representation of a single response model term.

    Parameters
    ----------
    term: formulae.ResponseVector
        An object describing the response of the model,
        as returned by formulae.design_matrices().response
    prior : Prior
        A specification of the prior(s) to use. An instance of class priors.Prior.
    """

    def __init__(self, term, prior=None):
        self.name = term.name
        self.data = term.design_vector
        self.categorical = term.type == "categoric"
        self.success_event = term.refclass
        self.prior = prior
        self.constant = np.var(self.data) == 0

class Term:
    """Representation of a single (common) model term.

    Parameters
    ----------
    name : str
        Name of the term.
    term: dict
        A dictionary describing the components of the term. These can be found in
        formulae.design_matrices().common.terms_info
    data : ndarray
        The term values.
    prior : Prior
        A specification of the prior(s) to use. An instance of class priors.Prior.
    constant : bool
        indicates whether the term levels collectively act as a constant, in which case the term is
        treated as an intercept for prior distribution purposes.
    """

    group_specific = False

    def __init__(self, name, term_dict, data, prior=None, constant=None):
        self.name = name
        self.data = data
        self.prior = prior
        self.categorical = term_dict["type"] == "categoric"
        self.levels = term_dict["full_names"]

        # identify and flag intercept and cell-means terms (i.e., full-rankdummy codes),
        # which receive special priors
        if constant is None:
            self.constant = np.atleast_2d(self.data.T).T.sum(1).var() == 0
        else:
            self.constant = constant

        # Any interaction with 1 categorical is considered categorical (at least for now)
        if term_dict["type"] == "interaction":
            if any([v["type"] == "categoric" for v in term_dict["terms"].values()]):
                self.categorical = True

        if self.categorical:
            if "levels" in term_dict.keys():
                self.cleaned_levels = term_dict["levels"]
            else:
                self.cleaned_levels = term_dict["reference"]
        else:
            self.cleaned_levels = None


class GroupSpecificTerm:
    """Representation of a single (group specific) model term.

    Parameters
    ----------
    name : str
        Name of the term.
    data : (DataFrame, Series, ndarray)
        The term values.
    predictor: (DataFrame, Series, ndarray)
        Data of the predictor variable in the group specific term.
    grouper: (DataFrame, Series, ndarray)
        Data of the grouping variable in the group specific term.
    categorical : bool
        If True, the source variable is interpreted as nominal/categorical. If False, the source
        variable is treated as continuous.
    prior : Prior
        A specification of the prior(s) to use. An instance of class priors.Prior.
    constant : bool
        indicates whether the term levels collectively act as a constant, in which case the term is
        treated as an intercept for prior distribution purposes.
    """

    group_specific = True

    def __init__(self, name, term_dict, data, prior=None, constant=None):
        self.name = name
        self.data = data
        self.prior = prior
        self.categorical = term_dict["type"] == "categoric"
        self.cleaned_levels = term_dict["groups"]
        self.levels = term_dict["full_names"]

        self.grouper = term_dict["Ji"]
        self.predictor = term_dict["Xi"]
        self.group_index = self.invert_dummies(self.grouper)

        if constant is None:
            self.constant = np.atleast_2d(self.data.T).T.sum(1).var() == 0
        else:
            self.constant = constant

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