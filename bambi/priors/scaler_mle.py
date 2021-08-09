# pylint: disable=no-name-in-module
import sys

from os.path import dirname, join

import numpy as np
import pandas as pd

from scipy.special import hyp2f1
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .prior import Prior


class PriorScalerMLE:
    """Scale prior distributions parameters.

    Used internally. Based on https://arxiv.org/abs/1702.01201
    """

    # Default is 'wide'. The wide prior sigma is sqrt(1/3) = .577 on the partial
    # corr scale, which is the sigma of a flat prior over [-1,1].
    names = {"narrow": 0.2, "medium": 0.4, "wide": 3 ** -0.5, "superwide": 0.8}

    def __init__(self, model, taylor):
        self.model = model

        # Equal to the design matrix for the common terms. Categorical are like "var[level]".
        # Q: What if the model does not have common effects? Not even intercepts. It doesn't work
        # right now. Should we flag it? Attempt to fix it?
        if model._design.common:
            self.dm = model._design.common.as_dataframe()
        else:
            self.dm = pd.DataFrame()

        self.has_intercept = model.intercept_term is not None

        self.priors = {}
        self.mle = None
        self.taylor = taylor
        with open(join(dirname(__file__), "config", "derivs.txt"), "r") as file:
            self.deriv = [next(file).strip("\n") for x in range(taylor + 1)]

    def get_intercept_stats(self, add_slopes=True):
        # Start with mean and variance of Y on the link scale
        mod = GLM(
            endog=self.model.response.data,
            exog=np.repeat(1, len(self.model.response.data)),
            family=self.model.family.smfamily(self.model.family.smlink),
            missing="drop" if self.model.dropna else "none",
        ).fit()
        mu = mod.params

        # Multiply SE by sqrt(N) to turn it into (approx.) sigma(Y) on link scale
        sigma = (mod.cov_params()[0] * len(mod.mu)) ** 0.5

        # Modify mu and sigma based on means and sigmas of slope priors.
        if self.model.common_terms and add_slopes:
            # prior["mu"] and prior["sigma"] have more than one value when the term is categoric.
            means = np.hstack([prior["mu"] for prior in self.priors.values()])
            sigmas = np.hstack([prior["sigma"] for prior in self.priors.values()])
            x_mean = np.hstack([self.model.terms[term].data.mean(axis=0) for term in self.priors])

            mu -= np.dot(means, x_mean)
            sigma = (sigma ** 2 + np.dot(sigmas ** 2, x_mean ** 2)) ** 0.5
        return mu, sigma

    def get_slope_stats(self, exog, name, values, sigma_corr, points=4, full_model=None):
        """
        Parameters
        ----------
        name: str
            Name of the term
        values: np.array
            Values of the term
        full_model: statsmodels.genmod.generalized_linear_model.GLM
            Statsmodels GLM to replace MLE model. For when ``'predictor'`` is not in the common
            part of the model.
        points : int
            Number of points to use for LL approximation.
        """

        # Make sure 'name' is in 'exog' columns

        if full_model is None:
            full_model = self.mle

        # Get log-likelihood values from beta=0 to beta=MLE
        beta_mle = full_model.params[name].item()
        beta_seq = np.linspace(0, beta_mle, points)

        log_likelihood = get_llh(self.model, exog, full_model, name, values, beta_seq)
        coef_a, coef_b = get_llh_coeffs(log_likelihood, beta_mle, beta_seq)
        p, q = shape_params(sigma_corr)

        # Evaluate the derivatives of beta = f(correlation).
        # dict 'point' gives points about which to Taylor expand.
        # We want to expand about the mean (generally 0), but some of the derivatives
        # do not exist at 0. Evaluating at a point very close to 0 (e.g., .001)
        # generally gives good results, but the higher order the expansion, the
        # further from 0 we need to evaluate the derivatives, or they blow up.
        point = dict(zip(range(1, 14), 2 ** np.linspace(-1, 5, 13) / 100))
        vals = dict(a=coef_a, b=coef_b, n=len(self.model.response.data), r=point[self.taylor])
        deriv = [eval(x, globals(), vals) for x in self.deriv]  # pylint: disable=eval-used

        terms = [
            compute_sigma(deriv, p, q, i, j)
            for i in range(1, self.taylor + 1)
            for j in range(1, self.taylor + 1)
        ]
        return np.array(terms).sum() ** 0.5

    def scale_response(self):
        # Add cases for other families
        priors = self.model.response.family.likelihood.priors
        if self.model.family.name == "gaussian":
            if priors["sigma"].auto_scale:
                sigma = np.std(self.model.response.data)
                priors["sigma"] = Prior("HalfStudentT", nu=4, sigma=sigma)

    def scale_common(self, term):
        """Scale common terms, excluding intercepts."""
        # Defaults are only defined for Normal priors
        if term.prior.name != "Normal":
            return
        mu = []
        sigma = []
        sigma_corr = term.prior.scale
        for name, values in zip(term.levels, term.data.T):
            mu += [0]
            sigma += [
                self.get_slope_stats(exog=self.dm, name=name, values=values, sigma_corr=sigma_corr)
            ]
        # Save and set prior
        self.priors.update({term.name: {"mu": mu, "sigma": sigma}})
        term.prior.update(mu=np.array(mu), sigma=np.array(sigma))

    def scale_intercept(self, term):
        # Default priors are only defined for Normal priors
        if term.prior.name != "Normal":
            return

        # Get prior mean and sigma for common intercept
        mu, sigma = self.get_intercept_stats()

        # Save and set prior
        term.prior.update(mu=mu, sigma=sigma)

    def scale_group_specific(self, term):
        # these default priors are only defined for HalfNormal priors
        if term.prior.args["sigma"].name != "HalfNormal":
            return

        sigma_corr = term.prior.scale

        # recreate the corresponding common effect data
        data_as_common = term.predictor

        # Handle intercepts
        if term.type == "intercept":
            _, sigma = self.get_intercept_stats()
            sigma *= sigma_corr
        # Handle slopes
        else:
            # Check whether the expr is also included as common term in the model.
            expr = term.name.split("|")[0]
            term_levels_len = term.predictor.shape[1]

            # Handle case where there IS a corresponding common effect with same encoding
            if expr in self.priors and term_levels_len == len(self.priors[expr]["mu"]):
                sigma = self.priors[expr]["sigma"]
            # Handle case where there IS NOT a corresponding common effect
            else:
                if expr in self.priors and not term_levels_len == len(self.priors[expr]["mu"]):
                    # Common effect is present, but with different encoding
                    # Replace columns from the common term with those from the group specific term.
                    exog = self.dm.drop(self.model.terms[expr].levels, axis=1)
                else:
                    # Common effect is not present
                    exog = self.dm

                # Append columns from 'data_as_common'
                df_to_append = pd.DataFrame(data_as_common)
                df_to_append.columns = [f"_name_{i}" for i in df_to_append.columns]
                exog = exog.join(df_to_append)

                # If there's intercept and the term is cell means, drop intercept to avoid
                # linear dependence in design matrix columns.
                if term.is_cell_means and self.has_intercept:
                    exog = exog.drop("Intercept", axis=1)

                sigma = []
                for name, values in zip(df_to_append.columns, data_as_common.T):
                    full_model = GLM(
                        endog=self.model.response.data,
                        exog=exog,
                        family=self.model.family.smfamily(self.model.family.smlink),
                        missing="drop" if self.model.dropna else "none",
                    ).fit()
                    sigma += [
                        self.get_slope_stats(
                            exog=exog,
                            name=name,
                            values=values,
                            full_model=full_model,
                            sigma_corr=sigma_corr,
                        )
                    ]
                sigma = np.array(sigma)
        # Set the prior sigma.
        term.prior.args["sigma"].update(sigma=np.squeeze(np.atleast_1d(sigma)))

    def scale(self):
        # Classify all terms
        common = list(self.model.common_terms.values())
        group_specific = list(self.model.group_specific_terms.values())

        if self.has_intercept:
            intercept = [self.model.intercept_term]
        else:
            intercept = []

        # Arrange them in the order in which they should be initialized
        terms = common + intercept + group_specific
        term_types = (
            ["common"] * len(common)
            + ["intercept"] * len(intercept)
            + ["group_specific"] * len(group_specific)
        )

        # Scale response
        self.scale_response()

        # Initialize terms in order
        for term, term_type in zip(terms, term_types):
            # Only scale priors if term or model is set to be auto scaled.
            # By default, use "wide".
            if not term.prior.auto_scale:
                continue

            if term.prior.scale is None:
                term.prior.scale = "wide"

            # Convert scale names to floats
            if isinstance(term.prior.scale, str):
                term.prior.scale = self.names[term.prior.scale]

            if self.mle is None:
                self.fit_mle()

            # Scale term with the appropiate method
            getattr(self, f"scale_{term_type}")(term)

    def fit_mle(self):
        """Fits MLE of the common part of the model.

        This used to be called in the class instantiation, but there is no need to fit the GLM when
        there are no automatic priors. So this method is only called when needed.
        """
        missing = "drop" if self.model.dropna else "none"
        try:
            self.mle = GLM(
                endog=self.model.response.data,
                exog=self.dm,
                family=self.model.family.smfamily(self.model.family.smlink),
                missing=missing,
            ).fit()
        except PerfectSeparationError as error:
            msg = "Perfect separation detected, automatic priors are not available. "
            msg += "Please indicate priors manually."
            raise PerfectSeparationError(msg) from error
        except:
            msg = "Unexpected error when trying to compute automatic priors."
            msg += "Please indicate priors manually."
            print(msg, sys.exc_info()[0])
            raise


def get_llh(model, exog, full_model, name, values, beta_seq):
    """
    Parameters
    ---------
    model: bambi.Model
    exog: pandas.DataFrame
    name: str
        Name of the term for which we want to compute the llh
    values: np.array
        Values of the term for which we want to compute the llh
    beta_seq: np.array
        Sequence of values from to beta_mle.
    """
    # True if there are other predictors appart from `predictor_name`
    if name not in exog.columns:
        raise ValueError("get_llh failed. Term name not in exog.")

    multiple_predictors = exog.shape[1] > 1
    sm_family = model.family.smfamily(model.family.smlink)

    if multiple_predictors:
        # Use statsmodels to _optimize_ the LL. Model is fitted 'points' times.
        glm_model = GLM(endog=model.response.data, exog=exog, family=sm_family)
        null_models = [glm_model.fit_constrained(f"{name}={beta}") for beta in beta_seq[:-1]]
        null_models = np.append(null_models, full_model)
        log_likelihood = np.array([x.llf for x in null_models])
    else:
        # Use statsmodels to _evaluate_ the LL. Model is fitted 'points' times.
        log_likelihood = [
            sm_family.loglike(np.squeeze(model.response.data), beta * values)
            for beta in beta_seq[:-1]
        ]
        log_likelihood = np.append(log_likelihood, full_model.llf)
    return log_likelihood


def moment(p, q, k):
    """Return central moments of rescaled beta distribution"""
    return (2 * p / (p + q)) ** k * hyp2f1(p, -k, p + q, (p + q) / p)


def compute_sigma(deriv, p, q, i, j):
    """Compute and return the approximate sigma"""
    return (
        1
        / np.math.factorial(i)
        * 1
        / np.math.factorial(j)
        * deriv[i]
        * deriv[j]
        * (moment(p, q, i + j) - moment(p, q, i) * moment(p, q, j))
    )


def get_llh_coeffs(llh, beta_mle, beta_seq):
    # compute params of quartic approximation to log-likelihood
    # c: intercept, d: shift parameter
    # a: quartic coefficient, b: quadratic coefficient
    # beta_mle: beta obtained via MLE
    # beta_seq: sequence from 0 to beta_mle
    intercept, shift_parameter = llh[-1], -beta_mle
    X = np.array([(beta_seq + shift_parameter) ** 4, (beta_seq + shift_parameter) ** 2]).T
    a, b = np.squeeze(  # pylint: disable=invalid-name
        np.linalg.multi_dot([np.linalg.inv(np.dot(X.T, X)), X.T, (llh[:, None] - intercept)])
    )
    return a, b


def shape_params(sigma_corr, mean=0.5):
    # m, v: mean and variance of beta distribution of correlations
    # p, q: corresponding shape parameters of beta distribution
    mean = 0.5
    variance = sigma_corr ** 2 / 4
    p = mean * (mean * (1 - mean) / variance - 1)
    q = (1 - mean) * (mean * (1 - mean) / variance - 1)
    return p, q
