import numpy as np

from .priors import Prior


class PriorScaler2:
    """Scale prior distributions parameters."""

    # Standard deviation multiplier.
    STD = 2.5

    def __init__(self, model):
        self.model = model
        self.has_intercept = any(term.type == "intercept" for term in self.model.terms.values())

        # Compute mean and std of the response
        if self.model.family.name == "gaussian":
            self.response_mean = np.mean(model.response.data)
            self.response_std = np.std(self.model.response.data)
        else:
            self.response_mean = 0
            self.response_std = 1

    def scale_response(self):
        if self.model.response.prior.auto_scale:
            if self.model.family.name == "gaussian":
                lam = 1 / self.response_std
                self.model.response.prior.update(sigma=Prior("Exponential", lam=lam))
            # Add cases for other families

    def scale_intercept(self, term):
        if term.prior.name != "Normal":
            return
        mu = self.response_mean
        sigma = self.STD * self.response_std
        term.prior.update(mu=mu, sigma=sigma)

    def scale_common(self, term):
        if term.prior.name != "Normal":
            return

        # As many zeros as columns in the data. It can be greater than 1 for categorical variables
        mu = np.zeros(term.data.shape[1])
        sigma = np.zeros(term.data.shape[1])

        # Iterate over columns in the data
        for i, x in enumerate(term.data.T):
            sigma[i] = self.STD * (self.response_std * np.std(x))

        # Set prior
        term.prior.update(mu=mu, sigma=sigma)

    def scale_group_specific(self, term):
        pass

    def scale(self):
        # Scale response
        self.scale_response()

        # Scale intercept
        if self.has_intercept:
            term = [t for t in self.model.common_terms.values() if t.type == "intercept"][0]
            if term.prior.auto_scale:
                self.scale_intercept(term)

        # Scale common terms
        for term in self.model.common_terms.values():
            # maybe intercept shouldn't go in common terms?
            if term.type == "intercept":
                continue
            if term.prior.auto_scale:
                self.scale_common(term)

        # Scale group-specific terms
        for term in self.model.group_specific_terms.values():
            if term.prior.auto_scale:
                self.scale_group_specific(term)
