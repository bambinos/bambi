import itertools
import numpy as np


class ResponseTerm:
    """Representation of a single response model term.

    Parameters
    ----------
    term: formulae.ResponseVector
        An object describing the response of the model,
        as returned by ``formulae.design_matrices().response``
    prior : Prior
        A specification of the prior(s) to use. An instance of class ``priors.Prior``.
    family : str
        The name of the model family.
    """

    def __init__(self, term, prior=None, family=None):
        self.name = term.name
        self.data = term.design_vector
        self.categorical = term.type == "categoric"
        self.success_event = term.refclass if term.refclass is not None else 1
        self.prior = prior
        self.constant = np.var(self.data) == 0

        if family == "bernoulli":
            if not all(np.isin(self.data, ([0, 1]))):
                raise ValueError("Numeric response must be all 0 and 1 for 'bernoulli' family.")


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
    constant : bool
        Indicates whether the term levels collectively act as a constant, in which case the term is
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

        if self.categorical:
            if "levels" in term_dict.keys():
                if term_dict["encoding"] == "full":
                    self.cleaned_levels = term_dict["levels"]
                else:
                    self.cleaned_levels = term_dict["levels"][1:]
            else:
                self.cleaned_levels = term_dict["reference"]
        else:
            self.cleaned_levels = None

        # Any interaction with 1 categorical is considered categorical (at least for now)
        if term_dict["type"] == "interaction":
            if any((v["type"] == "categoric" for v in term_dict["terms"].values())):
                self.categorical = True
                self.cleaned_levels = _interaction_labels(term_dict)


class GroupSpecificTerm:
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
    constant : bool
        Indicates whether the term levels collectively act as a constant, in which case the term is
        treated as an intercept for prior distribution purposes.
    """

    group_specific = True

    def __init__(self, name, term, data, prior=None, constant=None):
        self.name = name
        self.data = data
        self.prior = prior
        self.categorical = term["type"] == "categoric"
        self.cleaned_levels = term["groups"]
        self.levels = term["full_names"]

        self.grouper = term["Ji"]
        self.predictor = term["Xi"]
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


def _interaction_labels(x):
    # taken from formulae
    terms = x["terms"]
    colnames = []

    for k, val in terms.items():
        if val["type"] in ["numeric"]:
            colnames.append([k])
        if val["type"] == "categoric":
            if "levels" in val.keys():
                if val["encoding"] == "full":
                    colnames.append([f"{k}[{level}]" for level in val["levels"]])
                else:
                    colnames.append([f"{k}[{level}]" for level in val["levels"][1:]])
            else:
                colnames.append([f"{k}[{val['reference']}]"])

    return [":".join(str_tuple) for str_tuple in list(itertools.product(*colnames))]
