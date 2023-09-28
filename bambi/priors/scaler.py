import numpy as np
import pymc as pm

from bambi.families.univariate import Cumulative, Gaussian, StoppingRatio, StudentT, VonMises
from bambi.model_components import ConstantComponent
from bambi.priors.prior import Prior


class PriorScaler:
    """Scale prior distributions parameters."""

    # Standard deviation multipliefr.
    STD = 2.5

    def __init__(self, model):
        self.model = model
        self.response_component = model.response_component
        self.has_intercept = self.response_component.intercept_term is not None
        self.priors = {}

        # Compute mean and std of the response
        if isinstance(self.model.family, (Gaussian, StudentT)):
            self.response_mean = np.mean(self.response_component.response_term.data)
            self.response_std = np.std(self.response_component.response_term.data)
        else:
            self.response_mean = 0
            self.response_std = 1

    def get_intercept_stats(self):
        mu = self.response_mean
        sigma = self.STD * self.response_std

        # Only adjust mu and sigma if there is at least one Normal prior for a common term.
        if self.priors:
            sigmas = np.hstack([prior["sigma"] for prior in self.priors.values()])
            x_mean = np.hstack(
                [self.response_component.terms[term].data.mean(axis=0) for term in self.priors]
            )
            sigma = (sigma**2 + np.dot(sigmas**2, x_mean**2)) ** 0.5

        return mu, sigma

    def get_slope_sigma(self, x):
        return self.STD * (self.response_std / np.std(x))

    def scale_response(self):
        # Here we would add cases for other families if we wanted
        if isinstance(self.model.family, (Gaussian, StudentT)):
            sigma = self.model.components["sigma"]
            if isinstance(sigma, ConstantComponent) and sigma.prior.auto_scale:
                sigma.prior = Prior("HalfStudentT", nu=4, sigma=self.response_std)
        elif isinstance(self.model.family, VonMises):
            kappa = self.model.components["kappa"]
            if isinstance(kappa, ConstantComponent) and kappa.prior.auto_scale:
                kappa.prior = Prior("HalfStudentT", nu=4, sigma=self.response_std)

    def scale_intercept(self, term):
        if term.prior.name != "Normal":
            return
        mu, sigma = self.get_intercept_stats()
        term.prior.update(mu=mu, sigma=sigma)

    def scale_common(self, term):
        if term.prior.name != "Normal":
            return

        # It can be greater than 1 for categorical variables
        if term.data.ndim == 1:
            mu = 0
            sigma = self.get_slope_sigma(term.data)
        else:
            mu = np.zeros(term.data.shape[1])
            sigma = np.zeros(term.data.shape[1])
            # Iterate over columns in the data
            for i, value in enumerate(term.data.T):
                sigma[i] = self.get_slope_sigma(value)

        # Save and set prior
        self.priors.update({term.name: {"mu": mu, "sigma": sigma}})
        term.prior.update(mu=mu, sigma=sigma)

    def scale_group_specific(self, term):
        if term.prior.args["sigma"].name != "HalfNormal":
            return

        # Handle intercepts
        if term.kind == "intercept":
            _, sigma = self.get_intercept_stats()
        # Handle slopes
        else:
            # Recreate the corresponding common effect data
            if len(term.predictor.shape) == 2:
                data_as_common = term.predictor
            else:
                data_as_common = term.predictor[:, None]
            sigma = np.zeros(data_as_common.shape[1])
            for i, value in enumerate(data_as_common.T):
                sigma[i] = self.get_slope_sigma(value)
        term.prior.args["sigma"].update(sigma=np.squeeze(np.atleast_1d(sigma)))

    def scale_threshold(self):
        if isinstance(self.model.family, Cumulative):
            threshold = self.model.components["threshold"]
            if isinstance(threshold, ConstantComponent) and threshold.prior.auto_scale:
                response_level_n = len(np.unique(self.response_component.response_term.data))
                mu = np.round(np.linspace(-2, 2, num=response_level_n - 1), 2)
                threshold.prior = Prior(
                    "Normal",
                    mu=mu,
                    sigma=1,
                    transform=pm.distributions.transforms.ordered,
                )
        elif isinstance(self.model.family, StoppingRatio):
            threshold = self.model.components["threshold"]
            if isinstance(threshold, ConstantComponent) and threshold.prior.auto_scale:
                response_level_n = len(np.unique(self.response_component.response_term.data))
                mu = np.zeros(response_level_n - 1)
                threshold.prior = Prior("Normal", mu=mu, sigma=1)

    def scale(self):
        # Scale response
        self.scale_response()

        # Scale common terms
        for term in self.response_component.common_terms.values():
            if hasattr(term.prior, "auto_scale") and term.prior.auto_scale:
                self.scale_common(term)

        # Scale intercept
        if self.has_intercept:
            term = self.response_component.intercept_term
            if term.prior.auto_scale:
                self.scale_intercept(term)

        # Scale group-specific terms
        for term in self.response_component.group_specific_terms.values():
            if term.prior.auto_scale:
                self.scale_group_specific(term)

        # Scale threshold parameters in ordinal families
        self.scale_threshold()
