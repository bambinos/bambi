from bambi.terms.common import CommonTerm


class OffsetTerm(CommonTerm):
    """Representation of a single offset term.

    Parameters
    ----------
    name : str
        Name of the term.
    term : formulae.terms.terms.Term
        A model term created in formulae.
    """

    def __init__(self, term, prefix):
        super().__init__(term, 1, prefix)
        self.prior = 1

    @property
    def kind(self):
        return "offset"

    @property
    def categorical(self):
        return False

    @property
    def levels(self):
        return []

    @property
    def coords(self):
        return {}

    def __str__(self):
        return self.make_str()
