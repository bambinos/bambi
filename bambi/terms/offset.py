class OffsetTerm:
    """Representation of a single offset  term.

    Parameters
    ----------
    name : str
        Name of the term.
    term : formulae.terms.terms.Term
        A model term created in formulae.
    data : (DataFrame, Series, ndarray)
        The term values.
    """

    group_specific = False

    def __init__(self, name, term, data):
        self.name = name
        self.data = data.squeeze()
        self.kind = "offset"
        self.term = term
        self.alias = None
        self.coords = {}

    def set_alias(self, value):
        self.alias = value

    def __str__(self):
        args = [f"name: {self.name}", f"shape: {self.data.shape}"]
        if self.alias:
            args[0] = f"{args[0]} (alias: {self.alias})"
        return f"{self.__class__.__name__}({', '.join(args)})"

    def __repr__(self):
        return self.__str__()
