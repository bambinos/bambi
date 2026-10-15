import pathlib

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from scipy import stats
from scipy.special import expit, ndtr, gamma

import bambi as bmb
from bambi.parameters import MarginalCoefficient
from bambi.terms import GroupSpecificTerm

from helpers import (
    VECTOR_NORMAL_COV,
    assert_ip_dlogp,
    censored_nonlinear_data,
    censored_nonlinear_model,
    competing_risks_nonlinear_data,
    exponential_formula,
    exponential_priors,
    join_vector_response,
    linear_data,
    normal_prior,
    vector_normal_data,
    vector_normal_family,
    vector_student_t_family,
    weighted_nonlinear_data,
    weighted_nonlinear_model,
)


@pytest.fixture
def survival_data():
    return pd.DataFrame(
        {
            "time": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "x": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
            "status": ["none", "right", "left", "none", "right", "none"],
        }
    )


@pytest.fixture(scope="module")
def data_n100():
    size = 100
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "b1": rng.binomial(n=1, p=0.5, size=size),
            "n1": rng.poisson(lam=2, size=size),
            "n2": rng.poisson(lam=2, size=size),
            "y1": rng.normal(size=size),
            "y2": rng.normal(size=size),
            "y3": rng.normal(size=size),
            "cat2": rng.choice(["a", "b"], size=size),
            "cat4": rng.choice(list("MNOP"), size=size),
            "cat5": rng.choice(list("FGHIJK"), size=size),
        }
    )
    return data


@pytest.fixture(scope="module")
def cat_response_cat_preds_data():
    data = pd.read_csv(
        pathlib.Path(__file__).parent / "data" / "categorical_family_categorical_predictor.csv"
    )
    return data


@pytest.fixture(scope="module")
def zi_count_data():
    rng = np.random.default_rng(1234)
    n1, n2 = 30, 70
    y = np.concatenate([np.zeros(n1), rng.poisson(3, size=n2)])
    x = np.concatenate(
        [rng.normal(loc=-1, scale=0.25, size=n1), rng.normal(loc=0.5, scale=0.5, size=n2)]
    )
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def zi_bounded_count_data():
    rng = np.random.default_rng(1234)
    n1, n2 = 40, 60
    y = np.concatenate([np.zeros(n1), rng.binomial(n=30, p=0.6, size=n2)])
    x = np.concatenate(
        [rng.normal(loc=-1, scale=0.25, size=n1), rng.normal(loc=0.5, scale=0.5, size=n2)]
    )
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def zi_continuous_data():
    rng = np.random.default_rng(1234)
    n1, n2 = 40, 60
    y = np.concatenate([np.zeros(n1), rng.gamma(shape=2, scale=3, size=n2)])
    x = np.concatenate(
        [rng.normal(loc=-1, scale=0.25, size=n1), rng.normal(loc=0.5, scale=0.5, size=n2)]
    )
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def truncated_data():
    rng = np.random.default_rng(12345)
    slope, intercept, sigma, N = 1, 0, 2, 200
    x = rng.uniform(-10, 10, N)
    y = rng.normal(loc=slope * x + intercept, scale=sigma)
    bounds = [-5, 5]
    keep = (y >= bounds[0]) & (y <= bounds[1])
    xt = x[keep]
    yt = y[keep]

    return pd.DataFrame({"x": xt, "y": yt})


class FitPredictParent:
    def fit(self, model, **kwargs):
        self.assert_ip_dlogp(model)
        return model.fit(tune=200, draws=200, chains=2, **kwargs)

    def predict_oos(self, model, idata, data=None):
        # Reuse the original data
        if data is None:
            data = model.data.head()
        return model.predict(idata, kind="response", data=data, inplace=False)

    def assert_ip_dlogp(self, model):
        model.build()
        assert_ip_dlogp(model)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestGaussian(FitPredictParent):
    def test_intercept_only_model(self, data_crossed):
        model = bmb.Model("Y ~ 1", data_crossed)
        idata = self.fit(model)
        result = self.predict_oos(model, idata)
        assert result.predictions["Y"].shape[-1] == 5

    def test_slope_only_model(self, data_crossed):
        model = bmb.Model("Y ~ 0 + continuous", data_crossed)
        idata = self.fit(model)
        self.predict_oos(model, idata)

    def test_cell_means_parameterization(self, data_crossed):
        model = bmb.Model("Y ~ 0 + threecats", data_crossed)
        idata = self.fit(model)
        assert list(idata.posterior.coords["threecats_dim"]) == ["a", "b", "c"]
        self.predict_oos(model, idata)

    def test_2_factors_saturated(self, data_crossed):
        model = bmb.Model("Y ~ threecats*fourcats", data_crossed)
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {
            "Intercept",
            "Intercept_centered",
            "threecats",
            "fourcats",
            "threecats:fourcats",
            "sigma",
        }
        assert list(idata.posterior.coords["threecats_dim_reduced"]) == ["b", "c"]
        assert list(idata.posterior.coords["fourcats_dim_reduced"]) == ["b", "c", "d"]
        assert idata.posterior["threecats:fourcats"].dims == (
            "chain",
            "draw",
            "threecats_dim_reduced",
            "fourcats_dim_reduced",
        )
        self.predict_oos(model, idata)

    def test_2_factors_no_intercept(self, data_crossed):
        model = bmb.Model("Y ~ 0 + threecats*fourcats", data_crossed)
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {
            "threecats",
            "fourcats",
            "threecats:fourcats",
            "sigma",
        }
        assert list(idata.posterior.coords["threecats_dim"]) == ["a", "b", "c"]
        assert list(idata.posterior.coords["fourcats_dim_reduced"]) == ["b", "c", "d"]
        assert idata.posterior["threecats:fourcats"].dims == (
            "chain",
            "draw",
            "threecats_dim_reduced",
            "fourcats_dim_reduced",
        )
        self.predict_oos(model, idata)

    def test_2_factors_cell_means(self, data_crossed):
        model = bmb.Model("Y ~ 0 + threecats:fourcats", data_crossed)
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"threecats:fourcats", "sigma"}
        assert idata.posterior["threecats:fourcats"].dims == (
            "chain",
            "draw",
            "threecats_dim",
            "fourcats_dim",
        )
        assert list(idata.posterior.coords["threecats_dim"]) == ["a", "b", "c"]
        assert list(idata.posterior.coords["fourcats_dim"]) == ["a", "b", "c", "d"]
        self.predict_oos(model, idata)

    def test_cell_means_with_covariate(self, data_crossed):
        model = bmb.Model("Y ~ 0 + threecats + continuous", data_crossed)
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"threecats", "continuous", "sigma"}
        assert list(idata.posterior.coords["threecats_dim"]) == ["a", "b", "c"]
        self.predict_oos(model, idata)

    def test_many_common_many_group_specific(self, data_crossed):
        # Comparing implicit/explicit intercepts for group specific effects work the same way.
        terms0 = [
            "continuous",
            "dummy",
            "threecats",
            "(threecats|subj)",
            "(1|item)",
            "(0 + continuous|item)",
            "(dummy|item)",
            "(threecats|site)",
        ]
        terms1 = [
            "continuous",
            "dummy",
            "threecats",
            "(threecats|subj)",
            "(continuous|item)",
            "(dummy|item)",
            "(threecats|site)",
        ]

        model0 = bmb.Model("Y ~ " + " + ".join(terms0), data_crossed)
        idata0 = self.fit(model0)
        self.predict_oos(model0, idata0)

        model1 = bmb.Model("Y ~ " + " + ".join(terms1), data_crossed)
        idata1 = self.fit(model1)
        self.predict_oos(model1, idata1)

        # Check that the group specific effects design matrices have the same shape
        X0 = pd.concat(
            [
                pd.DataFrame(term.data.toarray())
                for term in model0.parameters["mu"].group_specific_terms.values()
            ],
            axis=1,
        )
        X1 = pd.concat(
            [
                pd.DataFrame(term.data.toarray())
                for term in model1.parameters["mu"].group_specific_terms.values()
            ],
            axis=1,
        )
        assert X0.shape == X1.shape

        # check that the group specific effect design matrix contain the same columns,
        # even if term names / columns names / order of columns is different
        X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
        X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
        assert X0_set == X1_set

        # check that common effect design matrices are the same,
        # even if term names / level names / order of columns is different
        X0_list = []
        X1_list = []
        for term in model0.parameters["mu"].common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X0_list.append(tuple(term.data[:, level_idx]))
            else:
                X0_list.append(tuple(term.data))

        for term in model1.parameters["mu"].common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X1_list.append(tuple(term.data[:, level_idx]))
            else:
                X1_list.append(tuple(term.data))

        assert set(X0_list) == set(X1_list)

        # check that models have same priors for common effects
        priors0 = {
            x.name: x.prior.args
            for x in model0.parameters["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args
            for x in model1.parameters["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }

        # check dictionary keys
        assert set(priors0) == set(priors1)

        # check dictionary values
        def dicts_close(a, b):
            if set(a) != set(b):
                return False
            else:
                return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

        assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])

        # check that fit and add models have same priors for group specific effects
        priors0 = {
            x.name: x.prior.args["sigma"].args
            for x in model0.parameters["mu"].group_specific_terms.values()
        }
        priors1 = {
            x.name: x.prior.args["sigma"].args
            for x in model1.parameters["mu"].group_specific_terms.values()
        }

        # check dictionary keys
        assert set(priors0) == set(priors1)

        # check dictionary values
        def dicts_close(a, b):
            if set(a) != set(b):
                return False
            else:
                return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

        assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])

    def test_cell_means_with_many_group_specific_effects(self, data_crossed):
        # Group specific intercepts are added in different way, but the final result should be the same.
        terms0 = [
            "0",
            "threecats",
            "(threecats|subj)",
            "(1|subj)",
            "(0 + continuous|item)",
            "(dummy|item)",
            "(0 + threecats|site)",
            "(1|site)",
        ]

        terms1 = [
            "0",
            "threecats",
            "(threecats|subj)",
            "(continuous|item)",
            "(dummy|item)",
            "(threecats|site)",
        ]
        model0 = bmb.Model("Y ~ " + " + ".join(terms0), data_crossed)
        idata0 = self.fit(model0)
        self.predict_oos(model0, idata0)

        model1 = bmb.Model("Y ~ " + " + ".join(terms1), data_crossed)
        idata1 = self.fit(model1)
        self.predict_oos(model1, idata1)

        # check that the group specific effects design matrices have the same shape
        X0 = pd.concat(
            [
                pd.DataFrame(term.data.toarray())
                for term in model0.parameters["mu"].group_specific_terms.values()
            ],
            axis=1,
        )
        X1 = pd.concat(
            [
                pd.DataFrame(term.data.toarray())
                for term in model0.parameters["mu"].group_specific_terms.values()
            ],
            axis=1,
        )
        assert X0.shape == X1.shape

        # check that the group specific effect design matrix contain the same columns,
        # even if term names / columns names / order of columns is different
        X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
        X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
        assert X0_set == X1_set

        # check that common effect design matrices are the same,
        # even if term names / level names / order of columns is different
        X0 = set(
            [
                tuple(t.data[:, lev])
                for t in model0.parameters["mu"].common_terms.values()
                for lev in range(len(t.levels))
            ]
        )
        X1 = set(
            [
                tuple(t.data[:, lev])
                for t in model1.parameters["mu"].common_terms.values()
                for lev in range(len(t.levels))
            ]
        )
        assert X0 == X1

        # check that fit and add models have same priors for common effects
        priors0 = {
            x.name: x.prior.args
            for x in model0.parameters["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args
            for x in model1.parameters["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        assert set(priors0) == set(priors1)

        # check that fit and add models have same priors for group specific effects
        priors0 = {
            x.name: x.prior.args["sigma"].args
            for x in model0.parameters["mu"].terms.values()
            if isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args["sigma"].args
            for x in model1.parameters["mu"].terms.values()
            if isinstance(x, GroupSpecificTerm)
        }
        assert set(priors0) == set(priors1)

    def test_group_specific_categorical_interaction(self, data_crossed):
        model = bmb.Model("Y ~ continuous + (threecats:fourcats|site)", data_crossed)
        idata = self.fit(model)
        self.predict_oos(model, idata)

        assert set(idata.posterior.data_vars) == {
            "Intercept",
            "Intercept_centered",
            "continuous",
            "sigma",
            "1|site_sigma",
            "threecats:fourcats|site_sigma",
            "1|site",
            "threecats:fourcats|site",
        }
        assert set(idata.posterior["threecats:fourcats|site"].coords) == {
            "chain",
            "draw",
            "site_dim",
            "threecats_dim_reduced",
            "fourcats_dim_reduced",
        }
        assert set(idata.posterior["1|site"].coords) == {"chain", "draw", "site_dim"}
        assert set(idata.posterior["1|site_sigma"].coords) == {"chain", "draw"}
        assert set(idata.posterior["threecats:fourcats|site_sigma"].coords) == {
            "chain",
            "draw",
            "threecats_dim_reduced",
            "fourcats_dim_reduced",
        }

        assert set(idata.posterior["threecats_dim_reduced"].values) == {"b", "c"}
        assert set(idata.posterior["fourcats_dim_reduced"].values) == {"b", "c", "d"}
        assert set(idata.posterior["site_dim"].values) == {"0", "1", "2", "3", "4"}

    def test_fit_include_mean(self, data_crossed):
        draws = 100
        model = bmb.Model("Y ~ continuous * threecats", data_crossed)
        idata = model.fit(tune=draws, draws=draws, chains=2, include_response_params=True)
        assert idata.posterior["mu"].shape[1:] == (draws, 120)

        # Compare with the mean obtained with `model.predict()`
        mean = idata.posterior["mu"].stack(sample=("chain", "draw")).values.mean(1)

        model.predict(idata)
        predicted_mean = idata.posterior["mu"].stack(sample=("chain", "draw")).values.mean(1)

        assert np.array_equal(mean, predicted_mean)

    def test_group_specific_splines(self):
        x_check = pd.DataFrame(
            {
                "x": [
                    82.0,
                    143.0,
                    426.0,
                    641.0,
                    1156.0,
                    986.0,
                    365.0,
                    187.0,
                    254.0,
                    550.0,
                    101.0,
                    661.0,
                    327.0,
                    119.0,
                ],
                "day": ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] * 2,
                "y": [
                    571.0,
                    684.0,
                    1652.0,
                    2130.0,
                    2455.0,
                    1874.0,
                    1288.0,
                    1011.0,
                    1004.0,
                    1993.0,
                    593.0,
                    1986.0,
                    1503.0,
                    711.0,
                ],
            }
        )
        knots = np.array([191.0, 297.0, 512.5])

        model = bmb.Model("y ~ (bs(x, knots=knots, intercept=False, degree=1)|day)", data=x_check)
        idata = self.fit(model)
        self.predict_oos(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestBernoulli(FitPredictParent):
    def assert_response_range(self, model, idata, group):
        y_name = model.response_term.label
        response = idata[group][y_name].to_numpy()
        assert set(np.unique(response)) == {0, 1}

    def assert_mean_range(self, idata, group):
        y_mean_name = "p"
        y_mean = idata[group][y_mean_name].to_numpy()
        assert ((0 < y_mean) & (y_mean < 1)).all()

    def test_bernoulli_empty_index(self, data_n100):
        model = bmb.Model("b1 ~ 1 + y1", data_n100, family="bernoulli")
        idata = self.fit(model)
        model.predict(idata, kind="response")
        self.assert_mean_range(idata, "posterior")
        self.assert_response_range(model, idata, "posterior_predictive")

        # out of sample prediction
        idata = self.predict_oos(model, idata)
        self.assert_mean_range(idata, "predictions")
        self.assert_response_range(model, idata, "predictions")

    def test_bernoulli_good_numeric(self, data_n100):
        model = bmb.Model("b1 ~ y1", data_n100, family="bernoulli")
        idata = self.fit(model)
        model.predict(idata, kind="response")
        self.assert_mean_range(idata, "posterior")
        self.assert_response_range(model, idata, "posterior_predictive")

    def test_bernoulli_bad_numeric(self, data_n100):
        error_msg = "Numeric response must be all 0 and 1 for 'bernoulli' family"
        with pytest.raises(ValueError, match=error_msg):
            model = bmb.Model("y1 ~ y2", data_n100, family="bernoulli")
            self.fit(model)

    def test_categorical_group_specific(self, data_n100):
        # see https://github.com/bambinos/bambi/issues/447

        formula = "b1 ~ 0 + cat2 + y2 + (0 + cat2|cat5) + (0 + y2| cat4 + cat5)"
        model = bmb.Model(formula, data_n100, family="bernoulli")
        idata = self.fit(model)
        idata = self.predict_oos(model, idata)

        self.assert_mean_range(idata, "predictions")
        self.assert_response_range(model, idata, "predictions")


@pytest.mark.usefixtures("mock_pymc_sample")
class TestBinomial(FitPredictParent):
    def assert_mean_range(self, idata, group):
        y_mean_name = "p"
        y_mean = idata[group][y_mean_name].to_numpy()
        assert ((0 < y_mean) & (y_mean < 1)).all()

    def test_binomial_regression(self, data_beetle):
        model = bmb.Model("prop(y, n) ~ x", data_beetle, family="binomial")
        idata = self.fit(model)
        model.predict(idata, kind="response")
        self.assert_mean_range(idata, "posterior")
        y_reshaped = data_beetle["n"].to_numpy()[None, None, :]

        assert (idata.posterior_predictive["y"].to_numpy() <= y_reshaped).all()
        assert (0 <= idata.posterior_predictive["y"].to_numpy()).all()

        y_reshaped = data_beetle["n"].to_numpy()[None, None, :3]
        idata = self.predict_oos(model, idata, data=model.data.head(3))
        self.assert_mean_range(idata, "predictions")
        assert (idata.predictions["y"].to_numpy() <= y_reshaped).all()

        # Test log-likelihood computation
        model.compute_log_likelihood(idata)
        idata_2 = model.compute_log_likelihood(idata, data=data_beetle, inplace=False)
        assert (idata_2.log_likelihood["y"] == idata.log_likelihood["y"]).all().item()

    def test_binomial_regression_constant(self, data_beetle):
        # Uses a constant instead of variable in data frame
        model = bmb.Model("p(y, 62) ~ x", data_beetle, family="binomial")
        idata = self.fit(model)
        model.predict(idata, kind="response")
        self.assert_mean_range(idata, "posterior")
        assert (idata.posterior_predictive["y"].to_numpy() <= 62).all()
        assert (0 <= idata.posterior_predictive["y"].to_numpy()).all()

        # Out of sample prediction
        idata = self.predict_oos(model, idata)
        self.assert_mean_range(idata, "predictions")
        assert (idata.predictions["y"].to_numpy() <= 62).all()

        # Test log-likelihood computation
        model.compute_log_likelihood(idata)
        idata_2 = model.compute_log_likelihood(idata, data=data_beetle, inplace=False)
        assert (idata_2.log_likelihood["y"] == idata.log_likelihood["y"]).all().item()

    def test_response_params_default_to_one_trial(self, data_beetle):
        model = bmb.Model("p(y, n) ~ x", data_beetle, family="binomial")
        idata = self.fit(model)
        new_data = data_beetle[["x"]].head(3)

        with pytest.warns(UserWarning, match="Using n=1"):
            result = model.predict(idata, data=new_data, kind="response_params", inplace=False)

        assert result.predictions["p"].shape[-1] == len(new_data)

        with pytest.raises(ValueError, match=r"Required variables: \['n'\]"):
            model.predict(idata, data=new_data, kind="response", inplace=False)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestPoisson(FitPredictParent):
    def assert_mean_range(self, idata, group):
        y_mean_name = "mu"
        y_mean = idata[group][y_mean_name].to_numpy()
        assert (y_mean > 0).all()

    def test_poisson_regression(self, data_crossed):
        data_crossed["count"] = (data_crossed["Y"] - data_crossed["Y"].min()).round()
        model0 = bmb.Model("count ~ dummy + continuous + threecats", data_crossed, family="poisson")
        idata0 = self.fit(model0)
        idata0 = self.predict_oos(model0, idata0)
        self.assert_mean_range(idata0, "predictions")

        # build model using add
        model1 = bmb.Model("count ~ threecats + continuous + dummy", data_crossed, family="poisson")
        idata1 = self.fit(model1)
        idata1 = self.predict_oos(model1, idata1)
        self.assert_mean_range(idata1, "predictions")

        # check that term names agree
        assert set(model0.parameters["mu"].terms) == set(model1.parameters["mu"].terms)

        # check that common effect design matrices are the same,
        # even if term names / level names / order of columns is different

        X0_list = []
        X1_list = []
        for term in model0.parameters["mu"].common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X0_list.append(tuple(term.data[:, level_idx]))
            else:
                X0_list.append(tuple(term.data))

        for term in model1.parameters["mu"].common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X1_list.append(tuple(term.data[:, level_idx]))
            else:
                X1_list.append(tuple(term.data))

        assert set(X0_list) == set(X1_list)

        # check that models have same priors for common effects
        priors0 = {
            x.name: x.prior.args
            for x in model0.parameters["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args
            for x in model1.parameters["mu"].terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        # check dictionary keys
        assert set(priors0) == set(priors1)

        # check dictionary values
        def dicts_close(a, b):
            if set(a) != set(b):
                return False
            else:
                return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

        assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])

        # Now test prior predictive
        pps = model1.prior_predictive(draws=200, random_seed=1234)

        keys = ["Intercept", "threecats", "continuous", "dummy"]
        shapes = [(1, 200), (1, 200, 2), (1, 200), (1, 200)]

        for key, shape in zip(keys, shapes):
            assert pps.prior[key].shape == shape

        assert pps.prior_predictive["count"].shape == (1, 200, 120)
        assert pps.observed_data["count"].shape == (120,)

        pps = model1.prior_predictive(draws=200, var_names=["count"], random_seed=1234)
        assert set(pps.children) == {"prior_predictive", "observed_data", "prior", "constant_data"}

        pps = model1.prior_predictive(draws=200, var_names=["Intercept"], random_seed=1234)
        assert set(pps.children) == {"prior_predictive", "observed_data", "prior", "constant_data"}

        # Now test posterior predictive
        # Fit again to make sure we fix the number of chains
        idata = model1.fit(tune=50, draws=50, chains=2)
        pps = model1.predict(idata, kind="response", inplace=False)
        assert pps.posterior_predictive["count"].shape == (2, 50, 120)

        pps = model1.predict(idata, kind="response", inplace=True)
        assert pps is None
        assert idata.posterior_predictive["count"].shape == (2, 50, 120)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestNegativeBinomial(FitPredictParent):
    # To Do: Could be modified to follow the same format than the others
    def test_predict_negativebinomial(self, data_n100):

        model = bmb.Model("n1 ~ y1", data_n100, family="negativebinomial")
        idata = self.fit(model)

        model.predict(idata, kind="response_params")
        assert (0 < idata.posterior["mu"]).all()

        model.predict(idata, kind="response")
        assert (np.equal(np.mod(idata.posterior_predictive["n1"].values, 1), 0)).all()

        model.predict(idata, kind="response_params", data=data_n100.iloc[:20, :])
        assert (0 < idata.predictions["mu"]).all()

        model.predict(idata, kind="response", data=data_n100.iloc[:20, :])
        assert (np.equal(np.mod(idata.predictions["n1"].values, 1), 0)).all()


@pytest.mark.usefixtures("mock_pymc_sample")
class TestLaplace(FitPredictParent):
    def test_laplace_regression(self, data_n100):
        model = bmb.Model("y1 ~ y2", data_n100, family="laplace")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "Intercept_centered", "y2", "b"}
        assert (idata.posterior["b"] > 0).all().item()

        idata = self.predict_oos(model, idata)
        assert "mu" in idata.predictions


@pytest.mark.usefixtures("mock_pymc_sample")
class TestGamma(FitPredictParent):
    def test_gamma_regression(self, data_n100):
        # Construct a positive variable
        data_n100["o"] = np.exp(data_n100["y1"])
        model = bmb.Model("o ~ y2 + y3 + n1 + cat4", data_n100, family="gamma", link="log")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {
            "Intercept",
            "Intercept_centered",
            "y2",
            "y3",
            "n1",
            "cat4",
            "alpha",
        }
        idata = self.predict_oos(model, idata)
        assert (idata.predictions["o"] >= 0).all().item()

        # Compute log likelihood
        model.compute_log_likelihood(idata)
        idata_2 = model.compute_log_likelihood(idata, data=data_n100, inplace=False)
        assert (idata.log_likelihood["o"] == idata_2.log_likelihood["o"]).all().item()

    def test_gamma_regression_categoric(self, data_n100):
        data_n100["o"] = np.exp(data_n100["y1"])
        model = bmb.Model("o ~ 0 + cat2:cat4", data_n100, family="gamma", link="log")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"cat2:cat4", "alpha"}
        idata = self.predict_oos(model, idata)
        assert (idata.predictions["o"] >= 0).all().item()


@pytest.mark.usefixtures("mock_pymc_sample")
class TestBeta(FitPredictParent):
    def test_beta_regression(self, data_gasoline):
        model = bmb.Model(
            "yield ~  temp + batch", data_gasoline, family="beta", categorical="batch"
        )
        idata = self.fit(model, target_accept=0.9, random_seed=1234)

        # To Do: Could be adjusted but this is what we had before
        model.predict(idata, kind="response_params")
        model.predict(idata, kind="response")

        assert (0 < idata.posterior["mu"]).all() & (idata.posterior["mu"] < 1).all()
        assert (0 <= idata.posterior_predictive["yield"]).all() & (
            idata.posterior_predictive["yield"] <= 1
        ).all()

        model.predict(idata, kind="response_params", data=data_gasoline.iloc[:20, :])
        model.predict(idata, kind="response", data=data_gasoline.iloc[:20, :])

        assert (0 < idata.predictions["mu"]).all() & (idata.predictions["mu"] < 1).all()
        assert (0 <= idata.predictions["yield"]).all() & (idata.predictions["yield"] <= 1).all()


@pytest.mark.usefixtures("mock_pymc_sample")
class TestStudentT(FitPredictParent):
    def test_t_regression(self, data_n100):
        model = bmb.Model("y1 ~ y2", data_n100, family="t")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {
            "Intercept",
            "Intercept_centered",
            "y2",
            "nu",
            "sigma",
        }
        self.predict_oos(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestVonMises(FitPredictParent):
    def test_vonmises_regression(self):
        rng = np.random.default_rng(1234)
        data = pd.DataFrame({"y": rng.vonmises(0, 1, size=100), "x": rng.normal(size=100)})
        model = bmb.Model("y ~ x", data, family="vonmises")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "Intercept_centered", "x", "kappa"}
        idata = self.predict_oos(model, idata)
        assert (idata.predictions["y"].min() >= -np.pi).item() and (
            idata.predictions["y"].max() <= np.pi
        ).item()


@pytest.mark.usefixtures("mock_pymc_sample")
class TestExGaussian(FitPredictParent):
    def test_exgaussian_regression(self):
        rng = np.random.default_rng(1234)
        # Reaction time like data: Gaussian component plus exponential component
        x = rng.normal(size=100)
        y = 0.4 + 0.2 * x + rng.normal(0, 0.05, size=100) + rng.exponential(0.15, size=100)
        data = pd.DataFrame({"y": y, "x": x})
        model = bmb.Model("y ~ x", data, family="exgaussian")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {
            "Intercept",
            "Intercept_centered",
            "x",
            "sigma",
            "nu",
        }
        idata = self.predict_oos(model, idata)
        assert "y" in idata["predictions"]


@pytest.mark.usefixtures("mock_pymc_sample")
class TestLogNormal(FitPredictParent):
    def test_lognormal_regression_rng(self):
        rng = np.random.default_rng(1234)
        # Positive, right-skewed outcome: exp of a linear predictor plus noise
        x = rng.normal(size=100)
        y = np.exp(0.4 + 0.2 * x + rng.normal(0, 0.2, size=100))
        data = pd.DataFrame({"y": y, "x": x})
        model = bmb.Model("y ~ x", data, family="lognormal")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "Intercept_centered", "x", "sigma"}
        idata = self.predict_oos(model, idata)
        assert "y" in idata["predictions"]


# NOTE: We don't use mock_pymc_sample because the assertion needs actual posterior samples to work.
class TestAsymmetricLaplace(FitPredictParent):
    # This test doesn't follow the previous pattern but it works...
    def test_quantile_regression(self):
        rng = np.random.default_rng(1234)
        x = rng.uniform(2, 10, 100)
        y = 2 * x + rng.normal(0, 0.6 * x**0.75)
        data = pd.DataFrame({"x": x, "y": y})
        bmb_model0 = bmb.Model("y ~ x", data, family="asymmetriclaplace", priors={"kappa": 9})
        idata0 = bmb_model0.fit(chains=2)
        bmb_model0.predict(idata0)

        bmb_model1 = bmb.Model("y ~ x", data, family="asymmetriclaplace", priors={"kappa": 0.1})
        idata1 = bmb_model1.fit(chains=2)
        bmb_model1.predict(idata1)

        assert np.all(
            idata0.posterior["mu"].mean(("chain", "draw"))
            > idata1.posterior["mu"].mean(("chain", "draw"))
        )


@pytest.mark.usefixtures("mock_pymc_sample")
class TestCategorical(FitPredictParent):
    # assert pps.shape[-1] == inhaler.shape[0]
    def assert_mean_sum(self, model, idata):
        y_mean_name = "p"
        y_mean_predictions = idata.predictions[y_mean_name]
        assert np.allclose(y_mean_predictions.sum(y_mean_predictions.dims[-1]).to_numpy(), 1)

    def assert_mean_range(self, model, idata):
        y_mean_name = "p"
        y_mean_predictions = idata.predictions[y_mean_name].to_numpy()
        assert ((0 < y_mean_predictions) & (y_mean_predictions < 1)).all()

    def assert_predictions_range(self, model, idata, n):
        y_name = model.response_term.label
        response = idata.predictions[y_name].to_numpy()
        assert set(np.unique(response)).issubset(set(range(n)))

    def test_basic(self, data_inhaler):
        model = bmb.Model("rating ~ period + carry + treat", data_inhaler, family="categorical")
        idata = self.fit(model)

        for name in ["Intercept", "period", "carry", "treat"]:
            assert list(idata.posterior[name].coords) == [
                "chain",
                "draw",
                "rating_dim_reduced",
            ]

        assert list(idata.posterior.coords["rating_dim_reduced"].values) == ["2", "3", "4"]

        idata = self.predict_oos(model, idata)
        assert list(idata.predictions["p"].coords) == [
            "chain",
            "draw",
            "__obs__",
            "rating_dim",
        ]
        assert list(idata.predictions.coords["rating_dim"].values) == ["1", "2", "3", "4"]
        self.assert_mean_range(model, idata)
        self.assert_mean_sum(model, idata)
        self.assert_predictions_range(model, idata, len(np.unique(data_inhaler["rating"])))

    def test_varying_intercept(self, data_inhaler):
        formula = "rating ~ period + carry + treat + (1|subject)"
        model = bmb.Model(formula, data_inhaler, family="categorical")
        idata = self.fit(model)

        for name in ["Intercept", "period", "carry", "treat"]:
            assert set(idata.posterior[name].coords) == {
                "chain",
                "draw",
                "rating_dim_reduced",
            }

        assert set(idata.posterior["1|subject"].coords) == {
            "chain",
            "draw",
            "rating_dim_reduced",
            "subject_dim",
        }

        assert (
            idata.posterior["subject_dim"].values == np.unique(data_inhaler["subject"]).astype(str)
        ).all()

        assert list(idata.posterior.coords["rating_dim_reduced"].values) == ["2", "3", "4"]

        idata = self.predict_oos(model, idata)
        assert set(idata.predictions["p"].coords) == {
            "chain",
            "draw",
            "__obs__",
            "rating_dim",
        }
        assert list(idata.predictions.coords["rating_dim"].values) == ["1", "2", "3", "4"]
        self.assert_mean_range(model, idata)
        self.assert_mean_sum(model, idata)
        self.assert_predictions_range(model, idata, len(np.unique(data_inhaler["rating"])))

    def test_categorical_predictors(self, cat_response_cat_preds_data):
        formula = "response ~ group + city"
        model = bmb.Model(formula, cat_response_cat_preds_data, family="categorical")
        idata = self.fit(model)

        assert set(idata.posterior["group"].coords) == {
            "chain",
            "draw",
            "response_dim_reduced",
            "group_dim_reduced",
        }
        assert set(idata.posterior["city"].coords) == {
            "chain",
            "draw",
            "response_dim_reduced",
            "city_dim_reduced",
        }
        assert list(idata.posterior["group_dim_reduced"].values) == ["group 2", "group 3"]
        assert list(idata.posterior["city_dim_reduced"].values) == ["Rosario", "San Luis"]
        assert list(idata.posterior["response_dim_reduced"].values) == ["B", "C", "D"]

        idata = self.predict_oos(model, idata)
        assert list(idata.predictions["response_dim"].values) == ["A", "B", "C", "D"]
        self.assert_mean_range(model, idata)
        self.assert_mean_sum(model, idata)
        self.assert_predictions_range(model, idata, 4)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestZeroInflatedFamilies(FitPredictParent):
    @pytest.mark.parametrize(
        "formula, data_name, family, priors",
        [  # Zero Inflated Poisson
            (bmb.Formula("y ~ x"), "zi_count_data", "zero_inflated_poisson", None),
            (bmb.Formula("y ~ x", "psi ~ x"), "zi_count_data", "zero_inflated_poisson", None),
            # Zero Inflated Negative Binomial
            (
                bmb.Formula("y ~ x"),
                "zi_count_data",
                "zero_inflated_negativebinomial",
                {"alpha": bmb.Prior("HalfNormal", sigma=20)},
            ),
            (
                bmb.Formula("y ~ x", "psi ~ x"),
                "zi_count_data",
                "zero_inflated_negativebinomial",
                {"alpha": bmb.Prior("HalfNormal", sigma=20)},
            ),
            # Zero Inflated Binomial
            (bmb.Formula("p(y, 30) ~ 1"), "zi_bounded_count_data", "zero_inflated_binomial", None),
            (
                bmb.Formula("p(y, 30) ~ 1", "psi ~ x"),
                "zi_bounded_count_data",
                "zero_inflated_binomial",
                None,
            ),
        ],
    )
    def test_family(self, formula, data_name, family, priors, request):
        data = request.getfixturevalue(data_name)
        model = bmb.Model(formula, data, priors=priors, family=family)
        idata = self.fit(model)
        self.predict_oos(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestHurdle(FitPredictParent):
    @pytest.mark.parametrize(
        "data_name, family",
        [
            ("zi_count_data", "hurdle_poisson"),
            ("zi_count_data", "hurdle_negativebinomial"),
            ("zi_continuous_data", "hurdle_gamma"),
            ("zi_continuous_data", "hurdle_lognormal"),
        ],
    )
    def test_hurlde_families(self, data_name, family, request):
        # To access 'data' which is a fixture
        data = request.getfixturevalue(data_name)
        model = bmb.Model("y ~ 1", data, family=family)
        idata = self.fit(model, random_seed=1234)
        self.predict_oos(model, idata)


# NOTE: We don't use mock_pymc_sample because the assertion needs actual posterior samples to work.
class TestOrdinal(FitPredictParent):
    @pytest.mark.parametrize(
        "family, link",
        [
            ("cumulative", "logit"),
            ("cumulative", "probit"),
            ("cumulative", "cloglog"),
            ("sratio", "logit"),
            ("sratio", "probit"),
            ("sratio", "cloglog"),
            ("cratio", "logit"),
            ("cratio", "probit"),
            ("cratio", "cloglog"),
            ("acat", "logit"),
            ("acat", "probit"),
            ("acat", "cloglog"),
            ("osm", "logit"),
        ],
    )
    def test_ordinal_families(self, data_inhaler, family, link):
        # To have both numeric and categoric predictors
        data_inhaler["carry"] = pd.Categorical(data_inhaler["carry"])
        model = bmb.Model("rating ~ period + carry + treat", data_inhaler, family=family, link=link)
        idata = self.fit(model, random_seed=1234)
        idata = self.predict_oos(model, idata)

        assert np.allclose(idata.predictions["p"].sum("rating_dim").to_numpy(), 1)
        assert set(np.unique(idata.predictions["rating"])).issubset({0, 1, 2, 3})

    def test_cumulative_family_priors(self, data_inhaler):
        priors = {
            "threshold": bmb.Prior(
                "Normal",
                mu=[-0.5, 0, 0.5],
                sigma=1.5,
                transform=pm.distributions.transforms.ordered,
            )
        }
        model = bmb.Model(
            "rating ~ 0 + period + carry + treat", data_inhaler, family="cumulative", priors=priors
        )
        idata = self.fit(model, random_seed=1234)
        self.predict_oos(model, idata)

    @pytest.mark.parametrize("family", ["cumulative", "sratio", "cratio", "acat"])
    def test_ordinal_cutpoint_dimensions(self, data_inhaler, family):
        model = bmb.Model("rating ~ period", data_inhaler, family=family)
        model.build()

        pymc_model = model.backend.model
        assert pymc_model.named_vars_to_dims["threshold"] == ("threshold_dim",)
        assert list(pymc_model.coords["threshold_dim"]) == ["1->2", "2->3", "3->4"]
        assert pymc_model.eval_rv_shapes()["threshold"] == (3,)

    def test_ordered_stereotype_prior(self, data_inhaler):
        prior = bmb.Prior("Dirichlet", dist=pm.Dirichlet, a=[2, 2, 2])
        model = bmb.Model("rating ~ period", data_inhaler, family="osm", priors={"delta": prior})
        idata = self.fit(model, random_seed=1234)
        self.predict_oos(model, idata)

    def test_ordered_stereotype_binary(self, data_inhaler):
        data = data_inhaler[data_inhaler["rating"].isin([1, 2])].copy()
        data["rating"] = data["rating"].cat.remove_unused_categories()
        model = bmb.Model("rating ~ period", data, family="osm")
        idata = self.fit(model, random_seed=1234)
        prediction = self.predict_oos(model, idata)

        np.testing.assert_allclose(idata.posterior["delta"], 1)
        assert set(np.unique(prediction.predictions["rating"])).issubset({0, 1})

    def test_cumulative_accepts_pymc_ordered_transform(self, data_inhaler):
        prior = bmb.Prior(
            "Normal",
            mu=[-0.5, 0, 0.5],
            sigma=1.5,
            transform=pm.distributions.transforms.ordered,
        )
        model = bmb.Model(
            "rating ~ period", data_inhaler, family="cumulative", priors={"threshold": prior}
        )
        model.build()

        assert model.backend.model.eval_rv_shapes()["threshold"] == (3,)


@pytest.mark.parametrize("is_censored", [False, True])
@pytest.mark.parametrize("model_shape", [False, True])
def test_weibull_ph_predict_and_log_likelihood(
    survival_data, mock_pymc_sample, is_censored, model_shape
):
    response = "censored(time, status)" if is_censored else "time"
    formula = f"{response} ~ x"
    if model_shape:
        formula = bmb.Formula(formula, "alpha ~ x")
    model = bmb.Model(formula, survival_data, family="weibull_ph")
    idata = model.fit(draws=4, chains=2, random_seed=1234)
    assert np.isfinite(
        model.backend.model.compile_dlogp()(model.backend.model.initial_point())
    ).all()
    new_data = survival_data.head(3)
    result = model.predict(idata, data=new_data, kind="response", inplace=False, random_seed=1234)
    assert result.predictions["lam"].shape == (2, 4, 3)
    assert result.predictions["time"].shape == (2, 4, 3)
    assert (result.predictions["time"] > 0).all()
    lam = result.predictions["lam"].to_numpy()
    alpha = (
        result.predictions["alpha"].to_numpy()
        if model_shape
        else idata.posterior["alpha"].to_numpy()[..., None]
    )
    time = new_data["time"].to_numpy()
    log_survival = -lam * time**alpha
    expected = np.log(alpha) + np.log(lam) + (alpha - 1) * np.log(time) + log_survival
    if is_censored:
        expected[..., 1] = log_survival[..., 1]
        expected[..., 2] = np.log(-np.expm1(log_survival[..., 2]))
    likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
    np.testing.assert_allclose(likelihood.log_likelihood["time"], expected, atol=1e-12)


@pytest.fixture
def cure_weibull_data():
    rng = np.random.default_rng(1234)
    x = rng.normal(size=100)
    cure = 1 / (1 + np.exp(-(-0.5 + 0.3 * x)))
    event_time = np.exp(0.3 * x) * rng.weibull(1.5, size=100)
    event_time[rng.uniform(size=100) < cure] = np.inf
    follow_up = rng.uniform(2, 8, size=100)
    time = np.minimum(event_time, follow_up)
    status = np.where(event_time > follow_up, "right", "none")
    status[time < 0.3] = "left"
    time = np.maximum(time, 0.3)
    data = pd.DataFrame({"time": time, "status": status, "x": x})
    # Keep the boundary cases used by conditional predictions and the survival plateau.
    data.loc[:3, ["time", "status", "x"]] = [
        [0.2, "none", -1.0],
        [1.0, "right", 0.0],
        [2.0, "left", 1.0],
        [1000.0, "right", 2.0],
    ]
    return data


@pytest.mark.parametrize("model_cure", [False, True])
@pytest.mark.parametrize("model_shape", [False, True])
def test_cure_weibull_predict_and_log_likelihood(
    cure_weibull_data, mock_pymc_sample, model_cure, model_shape
):
    additionals = []
    if model_cure:
        additionals.append("cure ~ x")

    if model_shape:
        additionals.append("alpha ~ x")

    priors = {}
    if model_shape:
        priors["alpha"] = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=0.2),
            "x": bmb.Prior("Normal", mu=0, sigma=0.2),
        }
    model = bmb.Model(
        bmb.Formula("censored(time, status) ~ x", *additionals),
        cure_weibull_data,
        family="cure_weibull",
        priors=priors,
    )
    assert model.family.link["cure"].name == "logit"
    idata = model.fit(draws=11, chains=2, random_seed=1234)
    assert np.isfinite(
        model.backend.model.compile_dlogp()(model.backend.model.initial_point())
    ).all()
    prediction = model.predict(
        idata, data=cure_weibull_data, kind="response", inplace=False, random_seed=1234
    )
    mu = prediction.predictions["mu"].to_numpy()
    cure = (
        prediction.predictions["cure"].to_numpy()
        if model_cure
        else idata.posterior["cure"].to_numpy()[..., None]
    )
    alpha = (
        prediction.predictions["alpha"].to_numpy()
        if model_shape
        else idata.posterior["alpha"].to_numpy()[..., None]
    )
    beta = mu / gamma(1 + 1 / alpha)
    time = cure_weibull_data["time"].to_numpy()
    log_susceptible_survival = -((time / beta) ** alpha)
    expected = (
        np.log1p(-cure)
        + np.log(alpha)
        - np.log(beta)
        + (alpha - 1) * np.log(time / beta)
        + log_susceptible_survival
    )
    log_survival = np.logaddexp(np.log(cure), np.log1p(-cure) + log_susceptible_survival)
    right = cure_weibull_data["status"].to_numpy() == "right"
    left = cure_weibull_data["status"].to_numpy() == "left"
    expected[..., right] = log_survival[..., right]
    expected[..., left] = (np.log1p(-cure) + np.log(-np.expm1(log_susceptible_survival)))[..., left]
    likelihood = model.compute_log_likelihood(idata, inplace=False)
    np.testing.assert_allclose(likelihood.log_likelihood["time"], expected, atol=1e-12)
    new_data = cure_weibull_data.head(2).drop(columns=["time", "status"])
    result = model.predict(idata, data=new_data, kind="response", inplace=False, random_seed=1234)
    assert result.predictions["time"].shape == (2, 11, 2)
    assert (result.predictions["time"] > 0).all()
    if model_cure:
        assert result.predictions["cure"].shape == (2, 11, 2)
        assert ((result.predictions["cure"] > 0) & (result.predictions["cure"] < 1)).all()


def test_cure_weibull_conditional_prediction(cure_weibull_data, mock_pymc_sample):
    model = bmb.Model(
        "censored(time, status) ~ 1",
        cure_weibull_data,
        family="cure_weibull",
        priors={
            "Intercept": bmb.Prior("Normal", mu=0, sigma=0.1),
            "alpha": bmb.Prior("Gamma", alpha=100, beta=100),
        },
    )
    idata = model.fit(draws=10, chains=2, random_seed=1234)
    result = model.predict(
        idata,
        data=cure_weibull_data.head(3),
        kind="response_conditional",
        inplace=False,
        random_seed=1234,
    )
    time = result.predictions["time"].to_numpy()
    assert (time[..., 1] >= 1.0).all()
    assert (time[..., 2] <= 2.0).all()
    assert np.isfinite(time[..., 2]).all()


@pytest.mark.parametrize("model_cure", [False, True])
@pytest.mark.parametrize("model_shape", [False, True])
@pytest.mark.parametrize(
    "family, parent, shape",
    [
        ("cure_exponential", "mu", None),
        ("cure_gamma", "mu", "alpha"),
        ("cure_lognormal", "mu", "sigma"),
        ("cure_loglogistic", "mu", "alpha"),
        ("cure_weibull_ph", "lam", "alpha"),
    ],
)
def test_cure_models_predict_and_log_likelihood(
    survival_data, mock_pymc_sample, family, parent, shape, model_cure, model_shape
):
    additionals = ["cure ~ x"] if model_cure else []
    priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=0.1),
        "x": bmb.Prior("Normal", mu=0, sigma=0.1),
    }
    if shape:
        if model_shape:
            additionals.append(f"{shape} ~ x")
            priors[shape] = {
                "Intercept": bmb.Prior("Normal", mu=0, sigma=0.1),
                "x": bmb.Prior("Normal", mu=0, sigma=0.1),
            }
        else:
            priors[shape] = bmb.Prior("Gamma", alpha=100, beta=100)
    model = bmb.Model(
        bmb.Formula("censored(time, status) ~ x", *additionals),
        survival_data,
        family=family,
        priors=priors,
    )
    idata = model.fit(draws=15, chains=2, random_seed=1234)
    prediction = model.predict(
        idata, data=survival_data, kind="response", inplace=False, random_seed=1234
    )
    mu = prediction.predictions[parent].to_numpy()
    cure = (
        prediction.predictions["cure"].to_numpy()
        if model_cure
        else idata.posterior["cure"].to_numpy()[..., None]
    )
    if shape:
        aux = (
            prediction.predictions[shape].to_numpy()
            if model_shape
            else idata.posterior[shape].to_numpy()[..., None]
        )
    if family == "cure_exponential":
        reference = stats.expon(scale=mu)
    elif family == "cure_gamma":
        reference = stats.gamma(a=aux, scale=mu / aux)
    elif family == "cure_lognormal":
        reference = stats.lognorm(s=aux, scale=np.exp(mu))
    elif family == "cure_loglogistic":
        reference = stats.fisk(c=1 / aux, scale=np.exp(mu))
    else:
        reference = stats.weibull_min(c=aux, scale=mu ** (-1 / aux))
    time = survival_data["time"].to_numpy()
    expected = np.log1p(-cure) + reference.logpdf(time)
    right = survival_data["status"].to_numpy() == "right"
    left = survival_data["status"].to_numpy() == "left"
    expected[..., right] = np.logaddexp(np.log(cure), np.log1p(-cure) + reference.logsf(time))[
        ..., right
    ]
    expected[..., left] = (np.log1p(-cure) + reference.logcdf(time))[..., left]
    likelihood = model.compute_log_likelihood(idata, inplace=False)
    np.testing.assert_allclose(likelihood.log_likelihood["time"], expected, atol=1e-10)

    new_data = survival_data.head(3)
    prediction = model.predict(
        idata,
        data=new_data.drop(columns=["time", "status"]),
        kind="response",
        inplace=False,
        random_seed=1234,
    )
    draws = prediction.predictions["time"].to_numpy()
    assert draws.shape == (2, 15, 3)
    assert (draws > 0).all()
    assert np.isinf(draws).any()
    assert np.isfinite(draws).any()

    conditional = (
        model.predict(
            idata, data=new_data, kind="response_conditional", inplace=False, random_seed=1234
        )
        .predictions["time"]
        .to_numpy()
    )
    assert (conditional[..., 1] >= 1.0).all()
    assert (conditional[..., 2] <= 1.5).all()
    assert np.isfinite(conditional[..., 2]).all()


@pytest.mark.usefixtures("mock_pymc_sample")
class TestCensoredResponses(FitPredictParent):
    @pytest.mark.parametrize("family", ["weibull", "weibull_ph"])
    def test_model_with_intercept(self, data_kidney, family):
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
            "sex": bmb.Prior("Normal", mu=0, sigma=2),
            "age": bmb.Prior("Normal", mu=0, sigma=1),
            "alpha": bmb.Prior("Gamma", alpha=3, beta=5),
        }
        model = bmb.Model(
            "censored(time, status) ~ 1 + sex + age",
            data_kidney,
            family=family,
            link="log",
            priors=priors,
        )
        idata = self.fit(model, random_seed=121195)
        self.predict_oos(model, idata)
        # Assert response is censored
        assert isinstance(model.backend.model.observed_RVs[0].owner.op, pm.Censored.rv_type)

    @pytest.mark.parametrize("family", ["weibull", "weibull_ph"])
    def test_model_without_intercept(self, data_kidney, family):
        priors = {
            "sex": bmb.Prior("Normal", mu=0, sigma=2),
            "age": bmb.Prior("Normal", mu=0, sigma=1),
            "alpha": bmb.Prior("Gamma", alpha=3, beta=5),
        }
        model = bmb.Model(
            "censored(time, status) ~ 0 + sex + age",
            data_kidney,
            family=family,
            link="log",
            priors=priors,
        )
        idata = self.fit(model, random_seed=121195)
        self.predict_oos(model, idata)
        # Assert response is censored
        assert isinstance(model.backend.model.observed_RVs[0].owner.op, pm.Censored.rv_type)

    @pytest.mark.parametrize("family", ["weibull", "weibull_ph"])
    def test_model_with_group_specific_effects(self, data_kidney, family):
        # Model 3, with group-specific effects
        priors = {
            "alpha": bmb.Prior("Gamma", alpha=3, beta=5),
            "sex": bmb.Prior("Normal", mu=0, sigma=1),
            "age": bmb.Prior("Normal", mu=0, sigma=1),
            "1|patient": bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("InverseGamma", alpha=5, beta=10)
            ),
        }
        model = bmb.Model(
            "censored(time, status) ~ 1 + sex + age + (1|patient)",
            data_kidney,
            family=family,
            link="log",
            priors=priors,
        )
        idata = self.fit(model, random_seed=121195)
        self.predict_oos(model, idata)
        # Assert response is censored
        assert isinstance(model.backend.model.observed_RVs[0].owner.op, pm.Censored.rv_type)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestTruncatedResponse(FitPredictParent):
    def test_truncated_response(self, truncated_data):
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
            "x": bmb.Prior("Normal", mu=0, sigma=1),
            "sigma": bmb.Prior("HalfNormal", sigma=1),
        }
        model = bmb.Model("truncated(y, -5, 5) ~ x", truncated_data, priors=priors)
        idata = self.fit(model, random_seed=121195)
        self.predict_oos(model, idata)
        # PyMC seems to automatically dispatch to TruncatedNormal
        assert isinstance(model.backend.model.observed_RVs[0].owner.op, pm.TruncatedNormal.rv_type)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestConstrainedResponse(FitPredictParent):
    def test_constrained_response(self, truncated_data):
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
            "x": bmb.Prior("Normal", mu=0, sigma=1),
            "sigma": bmb.Prior("HalfNormal", sigma=1),
        }
        model = bmb.Model("constrained(y, -5) ~ x", truncated_data, priors=priors)
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata)
        assert idata.predictions["y"].to_numpy().min() > -5

        model = bmb.Model("constrained(y, ub=5) ~ x", truncated_data, priors=priors)
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata)
        assert idata.predictions["y"].to_numpy().max() < 5

        model = bmb.Model("constrained(y, -5, 5) ~ x", truncated_data, priors=priors)
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata)
        assert idata.predictions["y"].to_numpy().min() > -5
        assert idata.predictions["y"].to_numpy().max() < 5


@pytest.mark.usefixtures("mock_pymc_sample")
class TestMultinomial(FitPredictParent):
    def assert_predictions(self, model, idata):
        y_name = model.response_term.label
        response = idata.predictions[y_name].to_numpy()
        assert (response.sum(-1).var((0, 1)) == 0).all()

    def test_intercept_only(self, data_multinomial):
        model = bmb.Model("counts(y1, y2, y3, y4) ~ 1", data_multinomial, family="multinomial")
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata, data=model.data)
        self.assert_predictions(model, idata)

        # Out of sample with different number of rows, see issue #845
        idata = self.predict_oos(model, idata, data=model.data.sample(frac=0.8, random_state=1211))
        self.assert_predictions(model, idata)

    def test_numerical_predictors(self, data_multinomial):
        model = bmb.Model(
            "counts(y1, y2, y3, y4) ~ treat + carry",
            data_multinomial,
            family="multinomial",
        )
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata, data=model.data)
        self.assert_predictions(model, idata)

        # Log likelihood computation
        model.compute_log_likelihood(idata)
        idata_2 = model.compute_log_likelihood(idata, data=data_multinomial, inplace=False)
        name = "y1_y2_y3_y4"
        assert (idata.log_likelihood[name] == idata_2.log_likelihood[name]).all().item()

    def test_categorical_predictors(self, data_multinomial):
        data_multinomial["treat"] = data_multinomial["treat"].replace({-0.5: "A", 0.5: "B"})
        data_multinomial["carry"] = data_multinomial["carry"].replace({-1: "a", 0: "b", 1: "c"})

        model = bmb.Model(
            "counts(y1, y2, y3, y4) ~ treat + carry",
            data_multinomial,
            family="multinomial",
        )
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata, data=model.data)
        self.assert_predictions(model, idata)

    def test_group_specific_effects(self):
        data = pd.DataFrame(
            {
                "state": ["A", "B", "C"],
                "y1": [35298, 1885, 5775],
                "y2": [167328, 20731, 21564],
                "y3": [212682, 37716, 20222],
                "y4": [37966, 5196, 3277],
            }
        )

        model = bmb.Model(
            "counts(y1, y2, y3, y4) ~ 1 + (1 | state)",
            data,
            family="multinomial",
            noncentered=False,
        )
        idata = self.fit(model, random_seed=121195)
        idata = self.predict_oos(model, idata, data=model.data)
        self.assert_predictions(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestDirichletMultinomial(FitPredictParent):
    def assert_predictions(self, model, idata):
        y_name = model.response_term.label
        response = idata.predictions[y_name].to_numpy()
        assert (response.sum(-1).var((0, 1)) == 0).all()

    def test_intercept_only(self, data_multinomial):
        model = bmb.Model(
            "counts(y1, y2, y3, y4) ~ 1", data_multinomial, family="dirichlet_multinomial"
        )
        idata = self.fit(model)
        idata = self.predict_oos(model, idata, model.data)
        self.assert_predictions(model, idata)

        # Out of sample with different number of rows, see issue #845
        idata = self.predict_oos(model, idata, data=model.data.sample(frac=0.8, random_state=1211))
        self.assert_predictions(model, idata)

    def test_predictor(self, data_multinomial):
        model = bmb.Model(
            "counts(y1, y2, y3, y4) ~ 0 + treat",
            data_multinomial,
            family="dirichlet_multinomial",
        )
        idata = self.fit(model)
        idata = self.predict_oos(model, idata, model.data)
        self.assert_predictions(model, idata)


@pytest.mark.usefixtures("mock_pymc_sample")
class TestBetaBinomial(FitPredictParent):
    def test_basic(self, data_beetle):
        model = bmb.Model("prop(y, n) ~ x", data_beetle, family="beta_binomial")
        idata = self.fit(model)
        idata = self.predict_oos(model, idata, model.data)
        n = data_beetle["n"].to_numpy()
        assert np.all(idata.predictions["y"].values <= n[np.newaxis, np.newaxis, :])


def test_wald_family(data_n100, mock_pymc_sample):
    data_n100["y"] = np.exp(data_n100["y1"])
    priors = {"common": bmb.Prior("Normal", mu=0, sigma=1)}
    model = bmb.Model("y ~ y2", data_n100, family="wald", link="log", priors=priors)
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)

    model.predict(idata, kind="response_params")
    model.predict(idata, kind="response")

    assert (0 < idata.posterior["mu"]).all()
    assert (0 < idata.posterior_predictive["y"]).all()

    model.predict(idata, kind="response_params", data=data_n100.iloc[:20, :])
    model.predict(idata, kind="response", data=data_n100.iloc[:20, :])

    assert (0 < idata.predictions["mu"]).all()
    assert (0 < idata.predictions["y"]).all()


def test_competing_risks_predictions():
    data = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "status": ["right", "event", "event", "right"],
            "cause": ["none", "cause_b", "cause_a", "none"],
            "x": [0.0, 1.0, 2.0, 3.0],
        }
    )
    model = bmb.Model("risks(time, status, cause) ~ x", data, family="exponential")
    idata = model.fit(draws=200, tune=200, chains=2, random_seed=1234)
    response = model.response_term.label

    in_sample = model.predict(idata, kind="response", inplace=False, random_seed=1234)
    assert in_sample.posterior_predictive[response].shape == (2, 200, len(data))

    in_sample_conditional = model.predict(
        idata, kind="response_conditional", inplace=False, random_seed=1234
    )
    conditional_times = in_sample_conditional.posterior_predictive[response]
    assert conditional_times.shape == (2, 200, len(data))
    assert (conditional_times[..., [0, 3]] >= data["time"].to_numpy()[[0, 3]] - 1e-3).all()

    in_sample_time_and_cause = model.predict(
        idata, kind="time_and_cause", inplace=False, random_seed=1234
    )
    event_times = in_sample_time_and_cause.posterior_predictive[f"{response}_time"]
    assert event_times.shape == (2, 200, len(data))
    assert (event_times > 0).all()
    causes = in_sample_time_and_cause.posterior_predictive[f"{response}_cause"]
    assert causes.shape == (2, 200, len(data))
    assert np.isin(causes, [1, 2]).all()

    new_data = pd.DataFrame({"x": [4.0, 5.0]})
    out_of_sample = model.predict(
        idata, data=new_data, kind="response", inplace=False, random_seed=1234
    )
    assert out_of_sample.predictions[response].shape == (2, 200, len(new_data))

    with pytest.raises(ValueError, match="Conditional competing-risks predictions require"):
        model.predict(idata, data=new_data, kind="response_conditional", inplace=False)

    out_of_sample_conditional = model.predict(
        idata, data=data, kind="response_conditional", inplace=False, random_seed=1234
    )
    conditional_times = out_of_sample_conditional.predictions[response]
    assert conditional_times.shape == (2, 200, len(data))
    assert (conditional_times[..., [0, 3]] >= data["time"].to_numpy()[[0, 3]] - 1e-3).all()

    out_of_sample_time_and_cause = model.predict(
        idata, data=new_data, kind="time_and_cause", inplace=False, random_seed=1234
    )
    assert out_of_sample_time_and_cause.predictions[f"{response}_time"].shape == (
        2,
        200,
        len(new_data),
    )
    causes = out_of_sample_time_and_cause.predictions[f"{response}_cause"]
    assert causes.shape == (2, 200, len(new_data))
    assert np.isin(causes, [1, 2]).all()

    log_likelihood = model.compute_log_likelihood(idata, data=data, inplace=False)
    assert log_likelihood.log_likelihood[response].shape == (2, 200, len(data))

    with pytest.raises(ValueError, match="requires variables 'time', 'status', 'cause'"):
        model.compute_log_likelihood(idata, data=new_data, inplace=False)

    non_cr_model = bmb.Model("y ~ 1", pd.DataFrame({"y": [0.0]}))

    with pytest.raises(ValueError, match="only available for competing-risks"):
        non_cr_model.predict(None, kind="time_and_cause")

    with pytest.raises(ValueError, match="only available for censored, truncated"):
        non_cr_model.predict(None, kind="response_conditional")


def test_competing_risks_time_and_cause_predictions_match_exponential_distribution():
    data = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "status": ["right", "event", "event", "right"],
            "cause": ["none", "cause_b", "cause_a", "none"],
        }
    )
    model = bmb.Model("risks(time, status, cause) ~ 1", data, family="exponential")
    model.build()

    response = model.response_term.label
    response_dim = f"{response}_dim"
    # The exponential family models the mean, so these intercepts correspond to
    # cause-specific rates 2 and 3. The first event is Exp(5), and cause 1 wins
    # with probability 2 / 5.
    posterior = xr.Dataset(
        {
            "Intercept": (
                ("chain", "draw", response_dim),
                np.broadcast_to(np.log([1 / 2, 1 / 3]), (1, 1000, 2)),
            )
        }
    )
    idata = xr.DataTree.from_dict({"posterior": posterior})

    result = model.predict(idata, kind="time_and_cause", inplace=False, random_seed=1234)
    times = result.posterior_predictive[f"{response}_time"]
    causes = result.posterior_predictive[f"{response}_cause"]

    np.testing.assert_allclose(times.mean(), 1 / 5, atol=0.015)
    np.testing.assert_allclose((causes == 1).mean(), 2 / 5, atol=0.03)


def test_predict_include_group_specific(data_random_n100):
    model = bmb.Model("binary_num ~ 1 + (1|continuous1)", data_random_n100, family="bernoulli")
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    idata_1 = model.predict(
        idata, data=data_random_n100, inplace=False, include_group_specific=True
    )
    idata_2 = model.predict(
        idata, data=data_random_n100, inplace=False, include_group_specific=False
    )

    assert not np.isclose(idata_1.predictions["p"].values, idata_2.predictions["p"].values).all()

    # Since it's an intercept-only model, predictions are the same for all observations if
    # we drop group-specific terms.
    assert (idata_2.predictions["p"] == idata_2.predictions["p"][:, :, 0]).all()

    # When we include group-specific terms, these predictions are different
    assert not (idata_1.predictions["p"] == idata_1.predictions["p"][:, :, 0]).all()


def test_predict_offset(mock_pymc_sample):
    # Simple case
    data = bmb.load_data("carclaims")
    model = bmb.Model("numclaims ~ offset(np.log(exposure))", data, family="poisson", link="log")
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata)
    model.predict(idata, kind="response")

    # More complex case
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.poisson(20, size=100),
            "x": rng.normal(size=100),
            "group": np.tile(np.arange(10), 10),
        }
    )
    data["time"] = data["y"] - rng.normal(loc=1, size=100)
    model = bmb.Model("y ~ offset(np.log(time)) + x + (1 | group)", data, family="poisson")
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata)
    assert "1|group_offset" not in idata.posterior
    model.predict(idata, kind="response")
    assert "1|group_offset" not in idata.posterior


def test_offset_contributes_to_linear_predictor():
    data = pd.DataFrame({"events": [1, 1], "exposure": [1.0, 100.0]})
    model = bmb.Model("events ~ offset(np.log(exposure))", data, family="poisson")
    model.build()

    with model.backend.model:
        mu = pm.draw(model.backend.model["mu"], draws=1, random_seed=121195)

    np.testing.assert_allclose(mu[1] / mu[0], 100)

    new_data = pd.DataFrame({"events": [1, 1], "exposure": [10.0, 1000.0]})
    prediction_data, _, _ = model.backend._build_new_data(
        new_data, purpose="prediction", kind="response_params"
    )
    np.testing.assert_allclose(
        prediction_data["offset(np.log(exposure))_data"], np.log(new_data["exposure"])
    )


def test_group_specific_offsets_predictions_and_log_likelihood(mock_pymc_sample):
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "y": rng.poisson(20, size=100),
            "x": rng.normal(size=100),
            "group": np.tile(np.arange(10), 10),
        }
    )
    data["time"] = data["y"] - rng.normal(loc=1, size=100)

    model = bmb.Model("y ~ offset(np.log(time)) + x + (1 | group)", data, family="poisson")
    idata = model.fit(chains=2)

    # In-sample prediction
    model.predict(idata)
    assert "mu" in idata.posterior

    # Out-of-sample prediction with the same group levels must use the fitted group effects.
    idata_oos = model.predict(idata, data=data, inplace=False)
    np.testing.assert_allclose(idata.posterior["mu"], idata_oos.predictions["mu"])

    # In-sample and out-of-sample log-likelihood
    model.compute_log_likelihood(idata)
    assert idata.log_likelihood["y"].shape == (2, 1000, len(data))

    idata_oos = model.compute_log_likelihood(idata, data=data, inplace=False)
    assert idata_oos.log_likelihood["y"].shape == (2, 1000, len(data))
    assert (idata.log_likelihood["y"] == idata_oos.log_likelihood["y"]).all()


def test_out_of_sample_prediction_accepts_fractional_integer_predictor(mock_pymc_sample):
    data = pd.DataFrame({"y": [1.2, 1.8, 3.1, 3.9], "x": [1, 2, 3, 4]})
    model = bmb.Model("y ~ x", data)
    idata = model.fit(draws=4, chains=2)

    prediction_data = pd.DataFrame({"x": [1.5, 2.5]})
    result = model.predict(idata, data=prediction_data, inplace=False)

    assert result.predictions["mu"].shape == (2, 4, 2)
    assert model.backend.model["x_data"].dtype == "float64"


def test_categorical_prediction_without_response_column(mock_pymc_sample):
    data = pd.DataFrame(
        {
            "choice": ["fish", "other", "invertebrate", "fish"],
            "length": [10, 12, 9, 11],
        }
    )
    model = bmb.Model("choice ~ length", data, family="categorical")
    idata = model.fit(draws=4, chains=2)

    result = model.predict(idata, data=pd.DataFrame({"length": [10.5, 11.5]}), inplace=False)

    assert result.predictions["p"].shape == (2, 4, 2, 3)


def test_binary_categorical_prediction_without_response_column(mock_pymc_sample):
    data = pd.DataFrame(
        {
            "vote": ["clinton", "trump", "clinton", "trump"],
            "age": [21, 35, 47, 61],
        }
    )
    model = bmb.Model("vote[clinton] ~ age", data, family="bernoulli")
    idata = model.fit(draws=4, chains=2)

    result = model.predict(idata, data=pd.DataFrame({"age": [30, 50]}), inplace=False)

    assert model.response_term.levels is None
    assert model.response_term.reference == "clinton"
    assert result.predictions["p"].shape == (2, 4, 2)


@pytest.mark.parametrize("sparse_dot", [False, True])
def test_predict_new_groups_is_automatic(
    data_sleepstudy, mock_pymc_sample, monkeypatch, sparse_dot
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = bmb.Model("Reaction ~ 1 + Days + (1 + Days | Subject)", data_sleepstudy)
    idata = model.fit(chains=2)
    model.build()
    assert_ip_dlogp(model)

    df_new = data_sleepstudy.head(10).reset_index(drop=True)
    df_new["Subject"] = "xxx"
    result = model.predict(idata, data=df_new, inplace=False)
    assert result.predictions["mu"].shape == (
        idata.posterior.sizes["chain"],
        idata.posterior.sizes["draw"],
        len(df_new),
    )


@pytest.mark.parametrize(
    "data,formula,family,df_new",
    [
        (
            "data_sleepstudy",
            "Reaction ~ 1 + Days + (1 + Days | Subject)",
            "gaussian",
            pd.DataFrame({"Days": [1, 2, 3], "Subject": ["x", "y", "z"]}),
        ),
        (
            "data_inhaler",
            "rating ~ 1 + period + treat + (1 + treat|subject)",
            "categorical",
            pd.DataFrame(
                {
                    "subject": [1, 999],
                    "rating": [1, 1],
                    "treat": [0.5, 0.5],
                    "period": [0.5, 0.5],
                    "carry": [0, 0],
                }
            ),
        ),
        (
            "data_crossed",
            "Y ~ 0 + threecats + (0 + threecats | subj)",
            "gaussian",
            pd.DataFrame({"threecats": ["a", "a"], "subj": ["0", "11"]}),
        ),
    ],
)
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_predict_new_groups(
    data, formula, family, df_new, request, mock_pymc_sample, monkeypatch, sparse_dot
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = request.getfixturevalue(data)
    model = bmb.Model(formula, data, family=family)
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata, data=df_new)


@pytest.mark.parametrize(
    "data,formula,family,df_new",
    [
        (
            "data_sleepstudy",
            "Reaction ~ 1 + Days + (1 + Days | Subject)",
            "gaussian",
            pd.DataFrame({"Days": [1, 2, 3], "Subject": ["x", "y", "z"]}),
        ),
        (
            "data_inhaler",
            "rating ~ 1 + period + treat + (1 + treat|subject)",
            "categorical",
            pd.DataFrame(
                {
                    "subject": [1, 999],
                    "rating": [1, 1],
                    "treat": [0.5, 0.5],
                    "period": [0.5, 0.5],
                    "carry": [0, 0],
                }
            ),
        ),
        (
            "data_crossed",
            "Y ~ 0 + threecats + (0 + threecats | subj)",
            "gaussian",
            pd.DataFrame({"threecats": ["a", "a"], "subj": ["0", "11"]}),
        ),
    ],
)
def test_predict_new_groups_deterministic(data, formula, family, df_new, request, mock_pymc_sample):
    data = request.getfixturevalue(data)
    model = bmb.Model(formula, data, family=family)
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)

    pred1 = model.predict(idata, data=df_new, random_seed=42, inplace=False)
    pred2 = model.predict(idata, data=df_new, random_seed=42, inplace=False)
    pred3 = model.predict(idata, data=df_new, random_seed=123, inplace=False)

    param_name, parameter = next(iter(model.conditional_parameters.items()))
    if parameter.alias:
        param_name = parameter.alias

    # Extract the parameter values
    param1 = pred1.predictions[param_name].values
    param2 = pred2.predictions[param_name].values
    param3 = pred3.predictions[param_name].values

    assert np.allclose(param1, param2), "Predictions with same random_seed should be identical"
    assert not np.allclose(
        param1, param3
    ), "Predictions with different random_seed should be different"


@pytest.mark.parametrize("sparse_dot", [False, True])
def test_predict_new_groups_mixed_known_and_unseen_levels(
    data_sleepstudy, mock_pymc_sample, monkeypatch, sparse_dot
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = bmb.Model("Reaction ~ 1 + Days + (1 + Days | Subject)", data_sleepstudy)
    idata = model.fit(draws=4, chains=2)

    known_data = data_sleepstudy.head(2).reset_index(drop=True)
    mixed_data = pd.concat([known_data, known_data.assign(Subject="unseen")], ignore_index=True)

    known_predictions = model.predict(idata, data=known_data, inplace=False)
    mixed_predictions = model.predict(idata, data=mixed_data, random_seed=42, inplace=False)

    np.testing.assert_allclose(
        known_predictions.predictions["mu"],
        mixed_predictions.predictions["mu"].isel(__obs__=[0, 1]),
    )
    assert mixed_predictions.predictions["mu"].shape == (2, 4, 4)

    # Group effects are omitted entirely, so unseen levels need not be sampled or rejected.
    no_group_predictions = model.predict(
        idata, data=mixed_data, include_group_specific=False, inplace=False
    )
    assert no_group_predictions.predictions["mu"].shape == (2, 4, 4)


@pytest.mark.parametrize("sparse_dot", [False, True])
def test_predict_unknown_groups(data_sleepstudy, mock_pymc_sample, monkeypatch, sparse_dot):
    """Missing factor values donate a fitted group without changing known rows."""
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = bmb.Model("Reaction ~ 1 + Days + (1 + Days | Subject)", data_sleepstudy)
    idata = model.fit(draws=4, chains=2)

    known_data = data_sleepstudy.head(1).reset_index(drop=True)
    prediction_data = pd.concat(
        [known_data, known_data.assign(Subject=None), known_data.assign(Subject=None)],
        ignore_index=True,
    )

    known = model.predict(idata, data=known_data, random_seed=42, inplace=False)
    prediction_1 = model.predict(idata, data=prediction_data, random_seed=42, inplace=False)
    prediction_2 = model.predict(idata, data=prediction_data, random_seed=42, inplace=False)

    np.testing.assert_allclose(
        known.predictions["mu"], prediction_1.predictions["mu"].isel(__obs__=[0])
    )
    np.testing.assert_allclose(prediction_1.predictions["mu"], prediction_2.predictions["mu"])
    if not sparse_dot:
        np.testing.assert_array_equal(
            prediction_1.predictions_constant_data["Subject__idx"], [0, -1, -1]
        )


@pytest.mark.parametrize("sparse_dot", [False, True])
def test_predict_identified_groups_share_effects(
    data_sleepstudy, mock_pymc_sample, monkeypatch, sparse_dot
):
    """Repeated new labels use one population draw per term and factor."""
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = bmb.Model("Reaction ~ 1 + Days + (1 + Days | Subject)", data_sleepstudy)
    idata = model.fit(draws=4, chains=2)
    data = pd.DataFrame(
        {
            "Days": [2, 2, 2, 2],
            "Subject": ["new_a", "new_a", "new_b", "new_b"],
        }
    )

    result = model.predict(idata, data=data, random_seed=42, inplace=False)
    mu = result.predictions["mu"]

    np.testing.assert_allclose(mu.isel(__obs__=[0]), mu.isel(__obs__=[1]))
    np.testing.assert_allclose(mu.isel(__obs__=[2]), mu.isel(__obs__=[3]))
    assert not np.allclose(mu.isel(__obs__=[0]), mu.isel(__obs__=[2]))


@pytest.mark.parametrize("sparse_dot", [False, True])
def test_predict_factor_interaction_groups(mock_pymc_sample, monkeypatch, sparse_dot):
    """Missing and new components use the unknown and identified joint-factor rules."""
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = pd.DataFrame(
        {
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "a": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "c": ["one", "one", "two", "two", "one", "one", "two", "two"],
        }
    )
    model = bmb.Model("y ~ 1 + (1|a:c)", data)
    idata = model.fit(draws=4, chains=2)
    prediction_data = pd.DataFrame(
        {
            "a": [None, "a", "a", "new_a", "new_a"],
            "c": ["one", None, "new_c", "one", "one"],
        }
    )

    term = next(iter(model.backend.spec.conditional_parameters["mu"].group_specific_terms.values()))
    group_index, _ = term.term.eval_new_data_group_index(prediction_data)
    assert np.all(group_index[:2] == -1)
    assert group_index[2] != group_index[3]
    assert group_index[3] == group_index[4]

    result = model.predict(idata, data=prediction_data, random_seed=42, inplace=False)

    assert result.predictions["mu"].shape == (2, 4, len(prediction_data))
    np.testing.assert_allclose(
        result.predictions["mu"].isel(__obs__=[3]),
        result.predictions["mu"].isel(__obs__=[4]),
    )


def test_predict_new_group_dense_single_term(mock_pymc_sample, monkeypatch):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", False)
    data = pd.DataFrame({"y": [0.0, 1.0, 2.0, 3.0], "group": ["a", "a", "b", "b"]})
    model = bmb.Model("y ~ 1 + (1 | group)", data)
    idata = model.fit(draws=4, chains=2)

    result = model.predict(idata, data=pd.DataFrame({"group": ["unseen"]}), inplace=False)

    assert result.predictions["mu"].shape == (2, 4, 1)


def test_weighted(mock_pymc_sample):
    weights = 1 + np.random.poisson(lam=3, size=100)
    y = np.random.exponential(scale=3, size=100)
    data = pd.DataFrame({"w": weights, "y": y})
    model = bmb.Model("weighted(y, w) ~ 1", data, family="exponential")
    model.build()
    assert_ip_dlogp(model)
    idata = model.fit(chains=2)
    model.predict(idata, kind="response")
    model.predict(idata, kind="response", data=data)


# Nonlinear auxiliary-parameter behavior


def nonlinear_auxiliary_data():
    x = np.linspace(-1, 1, 12)
    return pd.DataFrame({"y": 1 + x, "x": x, "z": x**2, "group": np.repeat(["a", "b", "c"], 4)})


def nonlinear_auxiliary_formula(groups=False):
    suffix = " + (1 | group)" if groups else ""
    additionals = ["sigma ~ z" + suffix]
    if groups:
        additionals.insert(0, "a ~ 1" + suffix)
    return bmb.Formula("y ~ a * x", *additionals, nlpars=("a",))


@pytest.mark.parametrize("nested", [False, True])
def test_nonlinear_auxiliary_preserves_parent_priors(nested):
    data = nonlinear_auxiliary_data()
    formula = bmb.Formula("y ~ x", "sigma ~ exp(a)", nlpars=("a",))
    model = bmb.Model(formula, data)
    reference = bmb.Model("y ~ x", data)
    for name, term in reference.parameters["mu"].terms.items():
        assert model.parameters["mu"].terms[name].prior == term.prior

    slope_prior = bmb.Prior("Normal", mu=0.5, sigma=0.2)
    coefficient_prior = bmb.Prior("Normal", mu=-1, sigma=0.5)
    priors = {"mu": {"x": slope_prior}} if nested else {"x": slope_prior}
    priors["a"] = coefficient_prior
    model = bmb.Model(formula, data, priors=priors)
    assert model.parameters["mu"].terms["x"].prior.args == slope_prior.args
    coefficient = model.parameters["sigma"].nonlinear_coefficients["a"]
    assert coefficient.prior.args == coefficient_prior.args

    slope_prior = bmb.Prior("Normal", mu=1, sigma=0.3)
    coefficient_prior = bmb.Prior("Normal", mu=-0.5, sigma=0.25)
    priors = {"mu": {"x": slope_prior}} if nested else {"x": slope_prior}
    priors["a"] = coefficient_prior
    model.set_priors(priors)
    assert model.parameters["mu"].terms["x"].prior.args == slope_prior.args
    assert coefficient.prior.args == coefficient_prior.args


def test_auxiliary_graph_matches_direct_pymc():
    data = nonlinear_auxiliary_data()
    model = bmb.Model(nonlinear_auxiliary_formula(), data, center_predictors=False)
    model.build()
    draws = xr.Dataset(
        {
            "a": (("chain", "draw"), [[2.0]]),
            "sigma_Intercept": (("chain", "draw"), [[-0.5]]),
            "sigma_z": (("chain", "draw"), [[0.3]]),
        }
    )
    with model.backend.model:
        actual = pm.compute_deterministics(draws, progressbar=False)
    np.testing.assert_allclose(actual.mu.values, [[2 * data.x]])
    np.testing.assert_allclose(actual.sigma.values, [[np.exp(-0.5 + 0.3 * data.z)]])
    likelihood = model.compute_log_likelihood(
        xr.DataTree.from_dict({"posterior": draws}), inplace=False
    )
    direct = pm.logp(
        pm.Normal.dist(mu=2 * data.x, sigma=np.exp(-0.5 + 0.3 * data.z)), data.y
    ).eval()
    np.testing.assert_allclose(likelihood.log_likelihood.y.values, [[direct]])
    assert set(model.parameters["mu"].nonlinear_coefficients) == {"a"}
    assert set(model.parameters["sigma"].terms) == {"Intercept", "z"}
    assert not model.marginal_parameters
    assert "target = sigma" in str(model)


def test_auxiliary_priors_match_ordinary_model_and_update():
    data = nonlinear_auxiliary_data()
    nonlinear = bmb.Model(nonlinear_auxiliary_formula(), data)
    ordinary = bmb.Model(bmb.Formula("y ~ x", "sigma ~ z"), data)
    for name, term in nonlinear.parameters["sigma"].terms.items():
        assert term.prior == ordinary.parameters["sigma"].terms[name].prior
    nonlinear.set_priors({"sigma": {"z": bmb.Prior("Normal", mu=0.2, sigma=0.4)}})
    assert nonlinear.parameters["sigma"].terms["z"].prior.args["mu"] == 0.2
    nonlinear.build()
    with pytest.warns(UserWarning, match="sigma.unknown"):
        nonlinear.set_priors({"sigma": {"unknown": bmb.Prior("Normal", mu=0, sigma=1)}})
    with pytest.raises(ValueError, match="must be a dictionary"):
        bmb.Model(
            nonlinear_auxiliary_formula(),
            data,
            priors={"sigma": bmb.Prior("HalfNormal", sigma=1)},
        )


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_auxiliary_group_prediction_and_likelihood(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = bmb.Model(
        nonlinear_auxiliary_formula(groups=True),
        nonlinear_auxiliary_data(),
        noncentered={"a": False, "sigma": True},
    )
    idata = model.fit(draws=3, chains=1, include_response_params=True, random_seed=123)
    assert {"mu", "sigma"} <= set(idata.posterior)
    assert "a" not in idata.posterior
    new_data = nonlinear_auxiliary_data().iloc[:3].copy()
    new_data["z"] += 1
    result = model.predict(idata, data=new_data, kind="response", inplace=False)
    assert result.predictions.sigma.shape == (1, 3, 3)
    assert result.predictions.y.shape == (1, 3, 3)
    x = xr.DataArray(new_data.x.to_numpy(), dims="__obs__")
    z = xr.DataArray(new_data.z.to_numpy(), dims="__obs__")
    group = idata.posterior["a_1|group"].sel(group_dim="a")
    expected_mu = (idata.posterior.a_Intercept + group) * x
    expected_sigma = np.exp(
        idata.posterior.sigma_Intercept
        + idata.posterior.sigma_z * z
        + idata.posterior["sigma_1|group"].sel(group_dim="a")
    )
    np.testing.assert_allclose(result.predictions.mu, expected_mu)
    np.testing.assert_allclose(result.predictions.sigma, expected_sigma)
    likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
    assert likelihood.log_likelihood.y.shape == (1, 3, 3)
    y = xr.DataArray(new_data.y.to_numpy(), dims="__obs__")
    expected_log_likelihood = (
        -0.5 * np.log(2 * np.pi)
        - np.log(expected_sigma)
        - 0.5 * ((y - expected_mu) / expected_sigma) ** 2
    )
    np.testing.assert_allclose(likelihood.log_likelihood.y, expected_log_likelihood)
    new_data.loc[new_data.index[0], "z"] = np.nan
    with pytest.raises(ValueError, match="incomplete rows"):
        model.predict(idata, data=new_data, inplace=False)


def test_auxiliary_missing_rows_share_one_mask():
    data = nonlinear_auxiliary_data()
    for index, column in enumerate(["y", "x", "z", "group"]):
        data.loc[index, column] = np.nan
    model = bmb.Model(nonlinear_auxiliary_formula(groups=True), data, dropna=True)
    assert len(model.data) == 8
    np.testing.assert_array_equal(model.response_term.data, data.y.iloc[4:])
    np.testing.assert_array_equal(model.parameters["sigma"].terms["z"].data, data.z.iloc[4:])
    model.build()
    with pytest.raises(ValueError, match="incomplete rows"):
        bmb.Model(nonlinear_auxiliary_formula(groups=True), data)


# Nonlinear aliases and offset behavior


def nonlinear_alias_model(groups=False, auxiliary=True, bare=False, noncentered=True):
    x = np.linspace(-1, 1, 12)
    data = pd.DataFrame({"y": 1 + x, "x": x, "z": x**2, "g": np.repeat(["u", "v", "w"], 4)})
    suffix = " + (1 | g)" if groups else ""
    formulas = ["y ~ a" if bare else "y ~ a * x"]
    if groups:
        formulas.append("a ~ 1" + suffix)
    if auxiliary:
        formulas.append("sigma ~ z" + suffix)
    return bmb.Model(
        bmb.Formula(*formulas, nlpars=("a",)),
        data,
        center_predictors=False,
        noncentered=noncentered,
    )


def nonlinear_aliases(auxiliary=True, groups=False):
    return {
        "mu": {"mu": "mean"},
        "a": {"a": "baseline", "Intercept": "a0"} if groups else "a0",
        "sigma": {"sigma": "noise", "Intercept": "s0", "z": "noise_z"} if auxiliary else "noise",
        "y": "response",
    }


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("bare", [False, True])
def test_alias_predictions_likelihood_and_rebuild(auxiliary, bare):
    model = nonlinear_alias_model(auxiliary=auxiliary, bare=bare)
    model.build()
    values = {"a": 2.0}
    values.update({"sigma_Intercept": -0.5, "sigma_z": 0.3} if auxiliary else {"sigma": 0.8})
    draws = xr.Dataset({name: (("chain", "draw"), [[value]]) for name, value in values.items()})
    original = model.predict(xr.DataTree.from_dict({"posterior": draws}), inplace=False)
    model.set_alias(nonlinear_aliases(auxiliary))
    assert not model.built
    model.build()
    renamed = {"a": "a0"}
    renamed.update(
        {"sigma_Intercept": "s0", "sigma_z": "noise_z"} if auxiliary else {"sigma": "noise"}
    )
    idata = xr.DataTree.from_dict({"posterior": draws.rename(renamed)})
    result = model.predict(idata, inplace=False)
    np.testing.assert_allclose(result.posterior["mean"], original.posterior["mu"])
    for data in (model.data, model.data.iloc[:3].assign(x=0.7, z=0.4)):
        result = model.predict(idata, data=data, inplace=False)
        expected = np.full(len(data), 2.0) if bare else 2 * data.x.to_numpy()
        scale = np.exp(-0.5 + 0.3 * data.z) if auxiliary else 0.8
        np.testing.assert_allclose(result.predictions["mean"], [[expected]])
        likelihood = model.compute_log_likelihood(idata, data=data, inplace=False)
        direct = pm.logp(pm.Normal.dist(mu=expected, sigma=scale), data.y).eval()
        np.testing.assert_allclose(likelihood.log_likelihood.response, [[direct]])
    if not bare:
        assert "mean__x_data" in model.backend.model.named_vars
        assert "mu__x_data" not in model.backend.model.named_vars
    assert "a0" in model.backend.model.named_vars
    assert "a" not in model.backend.model.named_vars
    assert "a0" in str(model)
    model.set_priors({"a": bmb.Prior("Normal", mu=0.4, sigma=0.2)})
    model.build()
    coefficient = model.parameters["mu"].nonlinear_coefficients["a"]
    assert coefficient.prior.args["mu"] == 0.4
    assert "a0" in model.backend.model.named_vars


@pytest.mark.parametrize("auxiliary", [False, True])
def test_alias_prior_filtering(auxiliary):
    model = nonlinear_alias_model(groups=True, auxiliary=auxiliary)
    mapping = nonlinear_aliases(auxiliary, groups=True)
    mapping["a"].update({"1|g": "by_group", "sigma": "group_sd"})
    model.set_alias(mapping)
    model.build()
    prior = model.backend.prior_predictive(
        draws=3, prior_only=True, omit_group_specific=True, random_seed=123
    )
    assert "a0" in prior.prior
    assert "by_group_group_sd" in prior.prior
    assert not {"mean", "baseline", "by_group", "sigma_1|g"} & set(prior.prior)
    assert ("noise" in prior.prior) is not auxiliary


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize("include_response_params", [False, True])
def test_alias_group_sampling_and_unknown_prediction(
    monkeypatch, sparse_dot, include_response_params
):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    model = nonlinear_alias_model(groups=True)
    mapping = nonlinear_aliases(groups=True)
    mapping["a"].update({"1|g": "by_group", "sigma": "group_sd"})
    mapping["sigma"].update({"1|g": "noise_group", "sigma": "noise"})
    model.set_alias(mapping)
    idata = model.fit(draws=3, chains=1, include_response_params=include_response_params)
    assert "baseline" not in idata.posterior
    assert ("mean" in idata.posterior) is include_response_params
    assert ("noise" in idata.posterior) is include_response_params
    assert "by_group_group_sd" in idata.posterior
    for group in ("u", "unseen"):
        data = model.data.iloc[:3].assign(g=group)
        prediction = model.predict(
            idata, data=data, kind="response", inplace=False, random_seed=123
        )
        assert prediction.predictions.response.shape == (1, 3, 3)
        assert prediction.predictions["mean"].shape == (1, 3, 3)
        if group == "unseen":
            with pytest.raises(ValueError, match="Cannot compute log likelihood for new groups"):
                model.compute_log_likelihood(idata, data=data, inplace=False)
        else:
            likelihood = model.compute_log_likelihood(idata, data=data, inplace=False)
            assert likelihood.log_likelihood.response.shape == (1, 3, 3)


def test_alias_unknown_names_types_and_collisions():
    model = nonlinear_alias_model(groups=True)
    with pytest.warns(UserWarning, match="missing, absent"):
        model.set_alias({"a": {"missing": "unused"}, "absent": {"Intercept": "other"}})
    with pytest.raises(AssertionError, match="Alias must be a string"):
        model.set_alias({"mu": {"mu": 5}})
    with pytest.raises(ValueError, match="must be a dictionary"):
        model.set_alias("a")
    model.set_alias({"a": {"a": "mu"}})
    with pytest.raises(ValueError, match="already exists"):
        model.build()


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("omit_offsets", [False, True])
def test_offset_suffix_aliases_remain_in_prior_and_posterior(nonlinear, omit_offsets):
    model = nonlinear_alias_model(groups=True, auxiliary=False)
    if nonlinear:
        model.set_alias(
            {"a": {"Intercept": "baseline_offset", "1|g": "group_offset"}, "sigma": "noise_offset"}
        )
    else:
        model = bmb.Model("y ~ x + (1|g)", model.data)
        model.set_alias(
            {"Intercept": "baseline_offset", "1|g": "group_offset", "sigma": "noise_offset"}
        )
    model.build()
    for prior_only in (False, True):
        prior = model.backend.prior_predictive(
            draws=3, prior_only=prior_only, omit_offsets=omit_offsets, random_seed=123
        )
        assert {"baseline_offset", "noise_offset", "group_offset"} <= set(prior.prior)
        assert ("group_offset_offset" in prior.prior) is not omit_offsets
    posterior = model.fit(draws=3, chains=1, omit_offsets=omit_offsets).posterior
    assert {"baseline_offset", "noise_offset", "group_offset"} <= set(posterior)
    assert ("group_offset_offset" in posterior) is not omit_offsets


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("override", [False, True])
def test_offset_inventory_respects_prior_override(override):
    model = nonlinear_alias_model(groups=True, auxiliary=False, noncentered=not override)
    model.set_priors(
        {
            "a": {
                "1|g": bmb.Prior(
                    "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1), noncentered=override
                )
            }
        }
    )
    model.build()
    assert ("a_1|g_offset" in model.backend.model.__bambi_attrs__["offset_names"]) is override
    prior = model.backend.prior_predictive(draws=3, omit_offsets=True, random_seed=123)
    assert "a_1|g_offset" not in prior.prior
    assert "a_1|g" in prior.prior
    idata = model.fit(draws=3, chains=1)
    prediction = model.predict(idata, data=model.data.iloc[:3], inplace=False)
    assert prediction.predictions.mu.shape == (1, 3, 3)
    likelihood = model.compute_log_likelihood(idata, inplace=False)
    assert likelihood.log_likelihood.y.shape == (1, 3, 12)


def test_offset_inventory_includes_recursive_hyperpriors():
    model = nonlinear_alias_model(groups=True, auxiliary=False)
    model.set_priors(
        {
            "a": {
                "1|g": bmb.Prior(
                    "Normal",
                    mu=0,
                    sigma=bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)),
                )
            }
        }
    )
    model.build()
    assert model.backend.model.__bambi_attrs__["offset_names"] == {
        "a_1|g_offset",
        "a_1|g_sigma_offset",
    }
    prior = model.backend.prior_predictive(draws=3, omit_offsets=True, random_seed=123)
    assert {"a_1|g", "a_1|g_sigma", "a_1|g_sigma_sigma"} <= set(prior.prior)
    assert not model.backend.model.__bambi_attrs__["offset_names"] & set(prior.prior)


# Nonlinear end-to-end integration


@pytest.fixture
def linked_auxiliary_model():
    data = pd.DataFrame(
        {"x": [0.0, 0.5, 1.5, 2.0], "z": [-1.0, 0.0, 0.5, 1.0], "y": [0.2, 0.4, 0.7, 0.6]}
    )
    formula = bmb.Formula("y ~ a + b * exp(-x)", "kappa ~ 1 + z", nlpars=("a", "b"))
    model = bmb.Model(
        formula,
        data,
        family="beta",
        center_predictors=False,
        priors={
            "a": bmb.Prior("Normal", mu=0, sigma=1),
            "b": bmb.Prior("Normal", mu=0, sigma=1),
            "kappa": {term: bmb.Prior("Normal", mu=0, sigma=1) for term in ("Intercept", "z")},
        },
    )
    model.set_alias(
        {
            "y": "outcome",
            "mu": {"mu": "probability"},
            "a": "a0",
            "b": "b0",
            "kappa": {"kappa": "precision", "Intercept": "p0", "z": "pz"},
        }
    )
    model.build()
    return model


@pytest.mark.parametrize("out_of_sample", [False, True])
def test_link_auxiliary_and_aliases_match_direct_pymc(linked_auxiliary_model, out_of_sample):
    model = linked_auxiliary_model
    data = model.data
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), [values])
            for name, values in {
                "a0": [-0.4, 0.2],
                "b0": [0.8, -0.3],
                "p0": [1.0, 1.5],
                "pz": [0.2, -0.1],
            }.items()
        }
    )
    with pm.Model(coords={"__obs__": range(len(data))}) as reference:
        x = pm.Data("x", data.x, dims="__obs__")
        z = pm.Data("z", data.z, dims="__obs__")
        y = pm.Data("y", data.y, dims="__obs__")
        a0 = pm.Normal("a0", mu=0, sigma=1)
        b0 = pm.Normal("b0", mu=0, sigma=1)
        p0 = pm.Normal("p0", mu=0, sigma=1)
        pz = pm.Normal("pz", mu=0, sigma=1)
        probability = pm.Deterministic(
            "probability", pm.math.sigmoid(a0 + b0 * pm.math.exp(-x)), dims="__obs__"
        )
        precision = pm.Deterministic("precision", pm.math.exp(p0 + pz * z), dims="__obs__")
        pm.Beta(
            "outcome",
            alpha=probability * precision,
            beta=(1 - probability) * precision,
            observed=y,
            dims="__obs__",
        )

    actual_logp = model.backend.model.compile_logp()
    expected_logp = reference.compile_logp()
    for draw in range(2):
        point = {name: value.values[0, draw] for name, value in posterior.items()}
        np.testing.assert_allclose(actual_logp(point), expected_logp(point))

    if out_of_sample:
        data = pd.DataFrame({"x": [0.2, 2.5], "z": [0.3, -0.7], "y": [0.4, 0.8]})
        pm.set_data(
            {"x": data.x, "z": data.z, "y": data.y},
            coords={"__obs__": range(len(data))},
            model=reference,
        )
    with reference:
        expected = pm.compute_deterministics(
            posterior, var_names=["probability", "precision"], progressbar=False
        )
        expected_likelihood = pm.compute_log_likelihood(
            xr.DataTree.from_dict({"posterior": posterior}), progressbar=False
        )

    idata = xr.DataTree.from_dict({"posterior": posterior})
    prediction_data = data.drop(columns="y") if out_of_sample else None
    predicted = model.predict(idata, data=prediction_data, inplace=False)
    actual = predicted.predictions if out_of_sample else predicted.posterior
    for name in ["probability", "precision"]:
        xr.testing.assert_allclose(actual[name], expected[name])
    likelihood = model.compute_log_likelihood(
        idata, data=data if out_of_sample else None, inplace=False
    )
    xr.testing.assert_allclose(
        likelihood.log_likelihood["outcome"], expected_likelihood.log_likelihood["outcome"]
    )
    assert {"baseline", "amplitude", "mu", "kappa"}.isdisjoint(actual.data_vars)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse", [False, True])
def test_linked_auxiliary_aliases_with_groups(monkeypatch, sparse):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse)
    data = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.5, 2.0],
            "z": [-1.0, 0.0, 0.5, 1.0],
            "y": [0.2, 0.4, 0.7, 0.6],
            "g": ["a", "a", "b", "b"],
        }
    )
    formula = bmb.Formula("y ~ a * exp(-x)", "a ~ 1 + (1 | g)", "kappa ~ 1 + z", nlpars=("a",))
    model = bmb.Model(formula, data, family="beta")
    model.set_alias(
        {
            "y": "outcome",
            "mu": {"mu": "probability"},
            "a": {"a": "baseline", "1|g": "group_effect"},
            "kappa": {"kappa": "precision"},
        }
    )
    idata = model.fit(draws=3, chains=1, random_seed=123)
    new_data = pd.DataFrame({"x": [0.2, 1.0], "z": [-0.5, 0.5], "g": ["a", "new"]})
    for include_groups in [False, True]:
        predicted = model.predict(
            idata,
            data=new_data,
            kind="response",
            include_group_specific=include_groups,
            random_seed=123,
            inplace=False,
        )
        assert predicted.predictions["outcome"].shape == (1, 3, 2)
        assert np.isfinite(predicted.predictions["probability"]).all()
        assert (
            (predicted.predictions["probability"] > 0) & (predicted.predictions["probability"] < 1)
        ).all()
        assert (predicted.predictions["precision"] > 0).all()
        z = xr.DataArray(new_data.z.to_numpy(), dims="__obs__")
        expected_precision = np.exp(idata.posterior.kappa_Intercept + idata.posterior.kappa_z * z)
        np.testing.assert_allclose(predicted.predictions["precision"], expected_precision)
        baseline = idata.posterior["a_Intercept"]
        if include_groups:
            baseline = baseline + idata.posterior["group_effect"].sel(g_dim="a")
        expected = 1 / (1 + np.exp(-baseline * np.exp(-0.2)))
        np.testing.assert_allclose(predicted.predictions["probability"].isel(__obs__=0), expected)


@pytest.mark.parametrize("link", ["identity", "log"])
def test_mixed_predictor_prediction_and_prior_updates(link):
    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0], "z": [1.0, 0.0, 3.0, 2.0]}
    )
    model = bmb.Model(
        bmb.Formula("y ~ nl(b * x + x) + z + offset(z)", nlpars=("b",)),
        data,
        link=link,
        center_predictors=False,
        priors={"Intercept": bmb.Prior("Normal", mu=0, sigma=2)},
    )
    model.set_priors(
        {"z": bmb.Prior("Normal", mu=0, sigma=3), "b": bmb.Prior("Normal", mu=0, sigma=4)}
    )
    assert model.parameters["mu"].terms["z"].prior.args["sigma"] == 3
    assert model.parameter_graph.nonlinear_coefficients["b"].prior.args["sigma"] == 4
    model.set_alias({"mu": {"mu": "mean", "z": "slope"}, "b": "rate"})
    model.build()
    draws = xr.Dataset(
        {
            name: (("chain", "draw"), [[value]])
            for name, value in {"Intercept": 0.5, "slope": 0.3, "rate": 0.2, "sigma": 1.0}.items()
        }
    )
    idata = xr.DataTree.from_dict({"posterior": draws})
    for new_data in [None, data.iloc[:2].assign(x=[0.3, 0.7], z=[0.5, 0.2])]:
        result = model.predict(idata, data=new_data, inplace=False)
        inputs = data if new_data is None else new_data
        expected = 0.5 + 0.3 * inputs.z + 0.2 * inputs.x + inputs.x + inputs.z
        if link == "log":
            expected = np.exp(expected)
        group = result.posterior if new_data is None else result.predictions
        np.testing.assert_allclose(group["mean"], expected.to_numpy()[None, None, :])
        likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
        expected_logp = -0.5 * np.log(2 * np.pi) - 0.5 * (inputs.y - expected) ** 2
        np.testing.assert_allclose(
            likelihood.log_likelihood.y, expected_logp.to_numpy()[None, None, :]
        )
    with pytest.raises(ValueError, match="incomplete rows"):
        model.predict(idata, data=data.assign(z=np.nan), inplace=False)


def test_data_only_nl_contribution():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.1, 0.2, 0.3]})
    model = bmb.Model("y ~ 0 + nl(exp(x) + 1)", data)
    model.build()
    draws = xr.Dataset({"sigma": (("chain", "draw"), [[1.0]])})
    result = model.predict(xr.DataTree.from_dict({"posterior": draws}), inplace=False)
    np.testing.assert_allclose(result.posterior.mu, (np.exp(data.x) + 1).to_numpy()[None, None, :])


@pytest.mark.parametrize("sparse_dot", [False, True])
def test_mixed_predictor_group_contributions(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0], "group": ["a", "a", "b", "b"]}
    )
    model = bmb.Model(
        bmb.Formula("y ~ 0 + nl(b * x) + (1 | group)", nlpars=("b",)),
        data,
        noncentered=False,
    )
    model.build()
    draws = xr.Dataset(
        {
            "b": (("chain", "draw"), [[0.2]]),
            "sigma": (("chain", "draw"), [[1.0]]),
            "1|group_sigma": (("chain", "draw"), [[0.5]]),
            "1|group": (("chain", "draw", "group_dim"), [[[0.3, -0.4]]]),
        },
        coords={"group_dim": ["a", "b"]},
    )
    idata = xr.DataTree.from_dict({"posterior": draws})
    new_data = pd.DataFrame({"x": [0.5, 1.5], "group": ["b", "a"]})
    for include_groups in [False, True]:
        result = model.predict(
            idata, data=new_data, inplace=False, include_group_specific=include_groups
        )
        expected = 0.2 * new_data.x.to_numpy()
        if include_groups:
            expected += [-0.4, 0.3]
        np.testing.assert_allclose(result.predictions.mu, expected[None, None, :])


def test_mixed_coefficient_and_auxiliary_predictors():
    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0], "z": [1.0, 0.0, 3.0, 2.0]}
    )
    formula = bmb.Formula(
        "y ~ a * x",
        "sigma ~ 0 + nl(exp(b)) + z",
        "a ~ nl(b * x) + z",
        nlpars=("a", "b"),
    )
    model = bmb.Model(formula, data, center_predictors=False)
    model.build()
    draws = xr.Dataset(
        {
            name: (("chain", "draw"), [[value]])
            for name, value in {"a_Intercept": 0.5, "a_z": 0.3, "b": 0.2, "sigma_z": 0.1}.items()
        }
    )
    idata = xr.DataTree.from_dict({"posterior": draws})
    new_data = data.iloc[:2].drop(columns="y")
    result = model.predict(idata, data=new_data, inplace=False)
    expected_mu = (0.5 + 0.3 * new_data.z + 0.2 * new_data.x) * new_data.x
    expected_sigma = np.exp(0.2) + 0.1 * new_data.z
    np.testing.assert_allclose(result.predictions.mu, expected_mu.to_numpy()[None, None, :])
    np.testing.assert_allclose(result.predictions.sigma, expected_sigma.to_numpy()[None, None, :])


def test_constant_parameters_match_linear_regression():
    data = linear_data()
    nonlinear = bmb.Model(
        bmb.Formula("y ~ a + b * x", nlpars=("a", "b")),
        data,
        priors={
            "a": normal_prior(),
            "b": normal_prior(),
        },
        center_predictors=False,
    )
    linear = bmb.Model(
        "y ~ x",
        data,
        priors={"Intercept": normal_prior(), "x": normal_prior()},
        center_predictors=False,
    )
    nonlinear.build()
    linear.build()

    nonlinear_draws = xr.Dataset(
        {
            "a": (("chain", "draw"), [[1.25]]),
            "b": (("chain", "draw"), [[-0.75]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )
    linear_draws = xr.Dataset(
        {
            "Intercept": (("chain", "draw"), [[1.25]]),
            "x": (("chain", "draw"), [[-0.75]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )

    with nonlinear.backend.model:
        nonlinear_mu = pm.compute_deterministics(
            nonlinear_draws, var_names=["mu"], progressbar=False
        )["mu"]
    with linear.backend.model:
        linear_mu = pm.compute_deterministics(linear_draws, var_names=["mu"], progressbar=False)[
            "mu"
        ]

    xr.testing.assert_allclose(nonlinear_mu, linear_mu)


def test_omitted_constant_formulas_match_explicit_intercepts():
    data = linear_data()
    formulas = [
        bmb.Formula("y ~ a + b * x", nlpars=("a", "b")),
        bmb.Formula("y ~ a + b * x", "a ~ 1", "b ~ 1", nlpars=("a", "b")),
    ]
    draws = xr.Dataset(
        {
            "a": (("chain", "draw"), [[1.25]]),
            "b": (("chain", "draw"), [[-0.75]]),
            "sigma": (("chain", "draw"), [[1.0]]),
        }
    )
    predictions = []

    for formula in formulas:
        model = bmb.Model(formula, data, center_predictors=False)
        model.build()
        with model.backend.model:
            predictions.append(
                pm.compute_deterministics(
                    draws,
                    var_names=["mu"],
                    progressbar=False,
                )["mu"]
            )

    xr.testing.assert_allclose(predictions[0], predictions[1])


def test_predictor_dependent_parameter_recovers_simulated_values():
    rng = np.random.default_rng(8)
    size = 120
    x = rng.uniform(0, 3, size)
    z = rng.normal(size=size)
    a = 0.7 + 0.4 * z
    y = a + 1.5 * np.exp(-0.8 * x) + rng.normal(0, 0.15, size)
    data = pd.DataFrame({"x": x, "z": z, "y": y})
    priors = exponential_priors()
    priors["b"] = bmb.Prior("Normal", mu=1.5, sigma=0.5)
    priors["k"] = bmb.Prior("LogNormal", mu=np.log(0.8), sigma=0.35)
    priors["sigma"] = bmb.Prior("HalfNormal", sigma=0.5)
    model = bmb.Model(exponential_formula(), data, priors=priors)

    idata = model.fit(
        draws=200,
        tune=200,
        chains=2,
        cores=1,
        random_seed=11,
        inference_method="pymc",
        target_accept=0.9,
    )

    expected = {
        "a_Intercept": 0.7,
        "a_z": 0.4,
        "b": 1.5,
        "k": 0.8,
    }
    for name, value in expected.items():
        assert float(idata.posterior[name].mean()) == pytest.approx(value, abs=0.15)

    predicted = model.predict(idata, kind="response", random_seed=4, inplace=False)
    assert predicted.posterior_predictive["y"].shape == (2, 200, size)


@pytest.mark.usefixtures("mock_pymc_sample")
def test_posterior_predictive_and_new_data():
    data = linear_data()
    model = bmb.Model(exponential_formula(), data, priors=exponential_priors())
    idata = model.fit(draws=5, chains=2)
    assert {"b", "k"} <= set(idata.posterior.data_vars)
    assert {"a", "mu"}.isdisjoint(idata.posterior.data_vars)

    predicted = model.predict(idata, kind="response", inplace=False)
    assert predicted.posterior["mu"].shape == (2, 5, len(data))
    assert predicted.posterior_predictive["y"].shape == (2, 5, len(data))

    new_data = pd.DataFrame({"x": np.linspace(1, 2, 7), "z": np.linspace(-0.5, 0.5, 7)})
    predicted = model.predict(idata, kind="response", data=new_data, inplace=False)
    assert predicted.predictions["mu"].shape == (2, 5, len(new_data))
    assert predicted.predictions["y"].shape == (2, 5, len(new_data))

    with pytest.raises(ValueError, match="missing nonlinear expression column"):
        model.predict(idata, data=new_data.drop(columns="x"), inplace=False)


@pytest.mark.usefixtures("mock_pymc_sample")
def test_include_response_params_only_keeps_likelihood_parameter():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    idata = model.fit(draws=5, chains=2, include_response_params=True)

    assert "mu" in idata.posterior
    assert {"b", "k"} <= set(idata.posterior.data_vars)
    assert {"a"}.isdisjoint(idata.posterior.data_vars)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_group_specific_parameter_predicts_new_data(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = linear_data(24)
    data["group"] = np.repeat(["a", "b", "c"], 8)
    formula = exponential_formula(group_specific=True)
    model = bmb.Model(formula, data, priors=exponential_priors(group_specific=True))
    idata = model.fit(draws=5, chains=2)

    assert model.backend.model.named_vars_to_dims["a_1|group"] == ("group_dim",)
    assert idata.posterior["a_1|group"].shape == (2, 5, 3)
    new_data = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3],
            "group": ["a", "new", "new"],
        }
    )
    predicted = model.predict(idata, data=new_data, random_seed=123, inplace=False)
    assert predicted.predictions["mu"].shape == (2, 5, 3)


def test_nonlinear_coefficient_alias():
    model = bmb.Model(exponential_formula(), linear_data(), priors=exponential_priors())

    model.set_alias({"a": {"Intercept": "a0"}})
    model.build()
    assert "a0" in model.backend.model.named_vars
    assert "a_Intercept" not in model.backend.model.named_vars


def ordinal_nonlinear_data():
    return pd.DataFrame(
        {
            "y": pd.Categorical(
                ["low", "mid", "high", "low", "mid", "high"],
                categories=["low", "mid", "high"],
                ordered=True,
            ),
            "x": [-1.0, -0.4, 0.0, 0.2, 0.7, 1.3],
            "z": [0.0, 1.0, 0.2, -0.4, 0.3, 1.1],
        }
    )


def ordinal_probabilities(family, eta, threshold, alpha=None, delta=None, link="logit"):
    from scipy.special import softmax

    inverse_link = ndtr if link == "probit" else expit
    eta = np.asarray(eta)[..., None]
    if family == "cumulative":
        cumulative = inverse_link(threshold - eta)
        return np.concatenate(
            [cumulative[..., :1], np.diff(cumulative, axis=-1), 1 - cumulative[..., -1:]],
            axis=-1,
        )
    if family in ("sratio", "cratio"):
        if family == "sratio":
            stop = inverse_link(threshold - eta)
            survive = 1 - stop
        else:
            survive = inverse_link(eta - threshold)
            stop = 1 - survive
        survival = np.cumprod(survive, axis=-1)
        return np.concatenate(
            [stop[..., :1], stop[..., 1:] * survival[..., :-1], survival[..., -1:]],
            axis=-1,
        )
    if family == "acat":
        log_odds = eta - threshold
        log_weights = np.concatenate(
            [np.zeros_like(log_odds[..., :1]), np.cumsum(log_odds, axis=-1)], axis=-1
        )
        return softmax(log_weights, axis=-1)
    phi = np.r_[0, np.cumsum(delta[:-1]), 1]
    intercepts = np.r_[0, alpha]
    return softmax(intercepts + eta * phi, axis=-1)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize(
    ("family", "link"),
    [
        ("cumulative", "logit"),
        ("cumulative", "probit"),
        ("sratio", "logit"),
        ("cratio", "logit"),
        ("acat", "logit"),
        ("osm", "logit"),
    ],
)
def test_nonlinear_ordinal_probabilities_and_predictions(family, link):
    data = ordinal_nonlinear_data()
    model = bmb.Model(
        bmb.Formula("y ~ nl(rate*x)", "rate ~ 1", nlpars=("rate",)),
        data,
        family=family,
        link=link,
    )
    rate_value = 0.4
    threshold_value = np.array([-0.8, 0.7])
    alpha_value = np.array([0.2, -0.35])
    delta_value = np.array([0.3, 0.7])
    values = (
        {"rate": rate_value, "alpha": alpha_value, "delta": delta_value}
        if family == "osm"
        else {"rate": rate_value, "threshold": threshold_value}
    )
    eta = rate_value * data.x.to_numpy()
    expected = ordinal_probabilities(
        family,
        eta,
        threshold_value,
        alpha=alpha_value,
        delta=delta_value,
        link=link,
    )
    idata = model.fit(draws=3, chains=2)
    posterior = idata.posterior.to_dataset().copy(deep=True)
    for name, value in values.items():
        posterior[name] = posterior[name].copy(data=np.broadcast_to(value, posterior[name].shape))
    idata["posterior"] = posterior
    parameter_prediction = model.predict(idata, kind="response_params", inplace=False)
    expected_draws = np.broadcast_to(expected, (2, 3, len(data), 3))
    np.testing.assert_allclose(parameter_prediction.posterior["p"], expected_draws)
    assert list(model.backend.model.coords["y_dim"]) == ["low", "mid", "high"]
    log_likelihood = model.compute_log_likelihood(idata, inplace=False)
    expected_log_likelihood = np.log(expected_draws[:, :, np.arange(len(data)), data.y.cat.codes])
    np.testing.assert_allclose(log_likelihood.log_likelihood["y"], expected_log_likelihood)
    in_sample = model.predict(idata, kind="response", inplace=False, random_seed=114)
    assert in_sample.posterior_predictive["y"].shape == (2, 3, len(data))
    for count in (2, 1):
        new_data = data.iloc[:count]
        params = model.predict(idata, data=new_data[["x"]], kind="response_params", inplace=False)
        prediction = model.predict(
            idata, data=new_data[["x"]], kind="response", inplace=False, random_seed=114
        )
        np.testing.assert_allclose(params.predictions["p"], expected_draws[:, :, :count])
        samples = prediction.predictions["y"]
        assert samples.shape == (2, 3, count)
        assert set(np.unique(samples)).issubset({0, 1, 2})
        out_of_sample_log_likelihood = model.compute_log_likelihood(
            idata, data=new_data, inplace=False
        )
        np.testing.assert_allclose(
            out_of_sample_log_likelihood.log_likelihood["y"], expected_log_likelihood[:, :, :count]
        )

    if family == "cumulative" and link == "logit":
        from bambi.interpret.effects import predictions, slopes

        summary = predictions(model, idata, conditional={"x": [-0.2, 0.3]}).summary
        assert set(summary.y_dim) == {"low", "mid", "high"}
        np.testing.assert_allclose(summary.groupby("x").estimate.sum(), 1)
        slope_summary = slopes(model, idata, wrt={"x": 0.2}).summary
        assert np.isfinite(slope_summary.estimate).all()
        np.testing.assert_allclose(slope_summary.estimate.sum(), 0, atol=1e-8)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("family", ["cumulative", "sratio", "cratio", "acat", "osm"])
def test_nonlinear_ordinal_binary_constant_expression(family):
    data = ordinal_nonlinear_data()
    data = data[data.y != "high"].copy()
    data["y"] = data.y.cat.remove_unused_categories()
    model = bmb.Model(bmb.Formula("y ~ 0 + nl(0.4)"), data, family=family)
    model.build()
    backend = model.backend.model
    values = {"alpha": np.array([0.2])} if family == "osm" else {"threshold": np.array([0.3])}
    actual = backend.compile_fn(backend["p"], inputs=backend.free_RVs)(values)
    expected = ordinal_probabilities(
        family, np.full(len(data), 0.4), np.array([0.3]), alpha=np.array([0.2]), delta=np.ones(1)
    )
    np.testing.assert_allclose(actual, expected)
    idata = model.fit(draws=2, chains=1, random_seed=115)
    for count in (2, 1):
        result = model.predict(
            idata,
            data=pd.DataFrame(index=range(count)),
            kind="response",
            inplace=False,
            random_seed=116,
        )
        assert result.predictions.p.shape == (1, 2, count, 2)
        np.testing.assert_allclose(result.predictions.p.sum(axis=-1), 1)
        assert np.isfinite(result.predictions.p).all()
        assert set(np.unique(result.predictions.y)).issubset({0, 1})


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize("family", ["cumulative", "osm"])
def test_nonlinear_ordinal_mixed_group_prediction(monkeypatch, sparse_dot, family):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = ordinal_nonlinear_data().assign(
        category=pd.Categorical(["a", "b"] * 3), group=["g", "h"] * 3
    )
    formula = bmb.Formula(
        "y ~ 1 + category + nl(rate*x)", "rate ~ 1 + z + (1 | group)", nlpars=("rate",)
    )
    with pytest.warns(UserWarning, match="intercept is omitted"):
        model = bmb.Model(formula, data, family=family, center_predictors=False)
    assert "Intercept" not in model.parameters["p"].terms
    model.set_alias(
        {
            "p": {"p": "probability", "category": "category_effect"},
            "rate": {"Intercept": "rate0", "z": "slope", "1|group": "group_effect"},
        }
    )
    idata = model.fit(draws=3, chains=1, random_seed=117)
    posterior = idata.posterior.to_dataset().copy(deep=True)
    auxiliary = (
        {"alpha": np.array([0.2, -0.35]), "delta": np.array([0.3, 0.7])}
        if family == "osm"
        else {"threshold": np.array([-0.8, 0.7])}
    )
    for name, value in auxiliary.items():
        posterior[name] = posterior[name].copy(data=np.broadcast_to(value, posterior[name].shape))
    idata["posterior"] = posterior
    assert posterior.rate0.dims == ("chain", "draw")
    assert posterior.category_effect.shape == (1, 3, 1)
    assert posterior.group_effect.dims == ("chain", "draw", "group_dim")
    fixed_eta = posterior.rate0.values + 0.2 * posterior.slope.values
    additive_eta = posterior.category_effect.values[..., 0]
    for groups in (["g", "h"], ["g"], ["new", "new"]):
        new_data = pd.DataFrame({"x": 0.3, "z": 0.2, "category": "b", "group": groups})
        result = model.predict(idata, data=new_data, inplace=False, random_seed=118)
        values = result.predictions.probability.values
        assert values.shape == (1, 3, len(groups), 3)
        np.testing.assert_allclose(values.sum(axis=-1), 1)
        assert (values >= 0).all()
        if groups[0] == "new":
            np.testing.assert_allclose(values[:, :, 0], values[:, :, 1])
        else:
            effects = posterior.group_effect.sel(group_dim=groups).values
            eta = additive_eta[..., None] + (fixed_eta[..., None] + effects) * 0.3
            expected = ordinal_probabilities(
                family,
                eta,
                np.array([-0.8, 0.7]),
                alpha=np.array([0.2, -0.35]),
                delta=np.array([0.3, 0.7]),
            )
            np.testing.assert_allclose(values, expected)
        population = model.predict(
            idata, data=new_data, include_group_specific=False, inplace=False
        )
        expected = ordinal_probabilities(
            family,
            additive_eta[..., None] + fixed_eta[..., None] * 0.3,
            np.array([-0.8, 0.7]),
            alpha=np.array([0.2, -0.35]),
            delta=np.array([0.3, 0.7]),
        )
        np.testing.assert_allclose(
            population.predictions.probability, np.broadcast_to(expected, values.shape)
        )


def test_nonlinear_coefficient_can_depend_on_another_coefficient():
    formula = bmb.Formula("y ~ a + b * x", "a ~ 1 + b", nlpars=("a", "b"))
    model = bmb.Model(formula, linear_data())
    model.build()
    draws = xr.Dataset(
        {
            "b": (("chain", "draw"), [[0.25, -0.5]]),
            "sigma": (("chain", "draw"), [[1.0, 1.0]]),
        }
    )

    with model.backend.model:
        actual = pm.compute_deterministics(draws, var_names=["mu"], progressbar=False)

    x = xr.DataArray(linear_data().x.to_numpy(), dims="__obs__")
    expected = 1 + draws.b + draws.b * x
    np.testing.assert_allclose(actual.mu, expected)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("column", ["x", "z"])
def test_prediction_rejects_incomplete_inputs(column):
    model = bmb.Model(
        exponential_formula(), linear_data(), priors=exponential_priors(), dropna=True
    )
    idata = model.fit(draws=2, chains=1)
    data = linear_data(3).drop(columns="y")
    data.loc[1, column] = np.nan
    with pytest.raises(ValueError, match="incomplete rows"):
        model.predict(idata, data=data, inplace=False)


@pytest.mark.parametrize("auto_scale", [False, True])
@pytest.mark.parametrize("prior", [None, bmb.Prior("Normal", mu=0.3, sigma=1.2)])
def test_marginal_coefficient_matches_explicit_intercept_density(auto_scale, prior):
    models = [
        bmb.Model(
            bmb.Formula("y ~ a * x", *additionals, nlpars=("a",)),
            linear_data(),
            priors={"a": prior, "sigma": 1.0},
            auto_scale=auto_scale,
        )
        for additionals, prior in [
            ((), prior),
            (("a ~ 1",), prior),
        ]
    ]
    for model in models:
        model.build()
    for value in (-0.5, 0.0, 1.3):
        points = [{"a": value}, {"a": value}]
        logps = [model.backend.model.compile_logp()(point) for model, point in zip(models, points)]
        gradients = [
            model.backend.model.compile_dlogp()(point) for model, point in zip(models, points)
        ]
        np.testing.assert_allclose(*logps)
        np.testing.assert_allclose(*gradients)


@pytest.mark.parametrize("prior", [normal_prior(), 0.5])
@pytest.mark.parametrize("additionals", [(), ("a ~ 1",), ("a ~ 0 + 1",)])
def test_marginal_coefficient_prior_alias_and_output(prior, additionals, mock_pymc_sample):
    model = bmb.Model(
        bmb.Formula("y ~ a * x", *additionals, nlpars=("a",)), linear_data(), priors={"a": prior}
    )
    coefficient = model.parameters["mu"].nonlinear_coefficients["a"]
    assert isinstance(coefficient, MarginalCoefficient)
    model.set_alias({"a": "slope"})
    model.build()
    assert "slope" in str(model)
    assert "a_Intercept" not in model.backend.model.named_vars
    assert model.backend.model["slope"].ndim == 0
    prior_draws = model.backend.prior_predictive(draws=3, prior_only=True, random_seed=123)
    assert "slope" in prior_draws.prior
    posterior = model.fit(draws=3, chains=1, random_seed=123)
    assert "slope" in posterior.posterior
    model.set_priors({"a": bmb.Prior("Normal", mu=0.7, sigma=0.2)})
    assert coefficient.prior.args == {"mu": 0.7, "sigma": 0.2}
    model.build()
    assert "slope" in model.backend.model.named_vars


def test_nl_wrapper_accepts_underscore_marginal_coefficient():
    model = bmb.Model(bmb.Formula("y ~ 0 + nl(_a * x)", nlpars=("_a",)), linear_data())
    model.build()
    assert "_a" in model.backend.model.named_vars


def test_nonlinear_weighted_response_matches_manual_logp():
    from scipy.stats import halfnorm, norm

    data = weighted_nonlinear_data()
    model = weighted_nonlinear_model(data)
    model.build()
    point = {"a_Intercept": 0.3, "a_z": -0.2, "b": 1.1, "sigma_log__": np.log(1.2)}
    sigma = np.exp(point["sigma_log__"])
    mu = point["a_Intercept"] + point["a_z"] * data.z.to_numpy() + point["b"] * data.x.to_numpy()

    actual_mu = model.backend.model.compile_fn(
        model.backend.model["mu"],
        inputs=[model.backend.model[name] for name in ("a_Intercept", "a_z", "b")],
    )({name: point[name] for name in ("a_Intercept", "a_z", "b")})
    np.testing.assert_allclose(actual_mu, mu)

    expected = np.sum(data.w * norm.logpdf(data.y, loc=mu, scale=sigma))
    expected += sum(
        norm.logpdf(point[name], loc=0, scale=2) for name in ("a_Intercept", "a_z", "b")
    )
    expected += halfnorm.logpdf(sigma, scale=1) + np.log(sigma)
    np.testing.assert_allclose(model.backend.model.compile_logp()(point), expected)


@pytest.mark.usefixtures("mock_pymc_sample")
def test_nonlinear_weighted_prediction_and_log_likelihood():
    model = weighted_nonlinear_model()
    idata = model.fit(draws=3, chains=1, random_seed=106)
    new_data = pd.DataFrame({"x": [0.2, -0.4], "z": [0.5, -1.0]})

    predictions = model.predict(
        idata, data=new_data, kind="response", inplace=False, random_seed=107
    )
    weighted_predictions = model.predict(
        idata, data=new_data.assign(w=[0.25, 4.0]), kind="response", inplace=False, random_seed=107
    )
    np.testing.assert_array_equal(
        predictions.predictions[model.response_term.label],
        weighted_predictions.predictions[model.response_term.label],
    )
    assert predictions.predictions["mu"].shape == (1, 3, 2)
    assert predictions.predictions[model.response_term.label].shape == (1, 3, 2)
    single = model.predict(idata, data=new_data.iloc[:1], kind="response", inplace=False)
    assert single.predictions["mu"].shape == (1, 3, 1)

    observed = new_data.assign(y=[0.1, -0.3], w=[0.25, 4.0])
    likelihood = model.compute_log_likelihood(idata, data=observed, inplace=False)
    assert likelihood.log_likelihood[model.response_term.label].shape == (1, 3, 2)
    pointwise = likelihood.log_likelihood[model.response_term.label].values.copy()
    assert np.isfinite(pointwise).all()

    unit_weights = model.compute_log_likelihood(
        idata, data=observed.drop(columns="w"), inplace=False
    )
    np.testing.assert_allclose(
        pointwise / unit_weights.log_likelihood[model.response_term.label].values,
        np.broadcast_to(observed.w.to_numpy(), pointwise.shape),
    )
    np.testing.assert_array_equal(
        model.backend.model["w_data"].get_value(), weighted_nonlinear_data().w.to_numpy()
    )


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("training_status", ["left", "right"])
def test_nonlinear_censored_prediction_and_new_status_log_likelihood(training_status):
    data = censored_nonlinear_data(("none", training_status, "none", training_status))
    model = censored_nonlinear_model(data)
    model.set_alias({"y": "outcome"})
    idata = model.fit(draws=3, chains=1, random_seed=107)
    in_sample = model.predict(idata, kind="response", inplace=False)
    assert in_sample.posterior_predictive[model.response_term.label].shape == (1, 3, 4)
    assert in_sample.posterior["mu"].shape == (1, 3, 4)

    new_data = pd.DataFrame({"x": [0.2, -0.4], "z": [0.5, -1.0]})
    predictions = model.predict(idata, data=new_data, kind="response", inplace=False)
    assert predictions.predictions["mu"].shape == (1, 3, 2)
    assert predictions.predictions[model.response_term.label].shape == (1, 3, 2)
    single = model.predict(idata, data=new_data.iloc[:1], kind="response", inplace=False)
    assert single.predictions[model.response_term.label].shape == (1, 3, 1)
    assert single.predictions["mu"].shape == (1, 3, 1)

    conditional_data = new_data.assign(y=[0.2, -0.3], status=["left", "right"])
    with pytest.raises(ValueError, match="Conditional censored-response predictions require"):
        model.predict(idata, data=new_data, kind="response_conditional", inplace=False)
    conditional = model.predict(
        idata, data=conditional_data, kind="response_conditional", inplace=False
    )
    draws = conditional.predictions[model.response_term.label].to_numpy()
    assert (draws[..., 0] <= conditional_data.y.iloc[0]).all()
    assert (draws[..., 1] >= conditional_data.y.iloc[1]).all()
    conditional_single = model.predict(
        idata, data=conditional_data.iloc[:1], kind="response_conditional", inplace=False
    )
    conditional_single_draws = conditional_single.predictions[model.response_term.label].to_numpy()
    assert conditional_single_draws.shape == (1, 3, 1)
    assert (conditional_single_draws[..., 0] <= conditional_data.y.iloc[0]).all()
    np.testing.assert_array_equal(
        model.backend.model["status_data"].get_value(), model.response_term.data[:, 1]
    )

    from scipy.stats import norm

    observed = pd.DataFrame(
        {
            "x": [0.2, -0.4, 0.7],
            "z": [0.5, -1.0, 0.2],
            "y": [0.2, -0.3, 0.8],
            "status": ["left", "right", "none"],
        }
    )
    likelihood = model.compute_log_likelihood(idata, data=observed, inplace=False)
    actual = likelihood.log_likelihood[model.response_term.label].to_numpy()
    likelihood_prediction = model.predict(
        idata, data=observed.drop(columns=["y", "status"]), kind="response", inplace=False
    )
    posterior = idata.posterior
    mu = (
        posterior["a_Intercept"].to_numpy()[..., None]
        + posterior["a_z"].to_numpy()[..., None] * observed.z.to_numpy()
        + posterior["rate"].to_numpy()[..., None] * observed.x.to_numpy()
    )
    np.testing.assert_allclose(likelihood_prediction.predictions["mu"], mu)
    sigma = idata.posterior["sigma"].to_numpy()[..., None]
    expected = np.where(
        observed.status.to_numpy() == "left",
        norm.logcdf(observed.y.to_numpy(), loc=mu, scale=sigma),
        np.where(
            observed.status.to_numpy() == "right",
            norm.logsf(observed.y.to_numpy(), loc=mu, scale=sigma),
            norm.logpdf(observed.y.to_numpy(), loc=mu, scale=sigma),
        ),
    )
    np.testing.assert_allclose(actual, expected)
    for variable in ("y", "status"):
        with pytest.raises(ValueError, match="Censored response log-likelihood requires variables"):
            model.compute_log_likelihood(idata, data=observed.drop(columns=variable), inplace=False)


def truncated_nonlinear_data():
    return pd.DataFrame(
        {
            "x": [-0.8, 0.0, 0.5, 0.9],
            "z": [0.0, 1.0, 0.5, -0.5],
            "y": [0.0, 0.4, 1.1, 0.4],
            "lower": [-1.0, -0.5, 0.0, -0.25],
            "upper": [0.5, 1.2, 1.8, 1.0],
        }
    )


def truncated_nonlinear_model(data, response="truncated(y, lower, upper)"):
    formula = bmb.Formula(f"{response} ~ a + rate * x", "a ~ 1 + z", nlpars=("a", "rate"))
    priors = {
        "a": {"Intercept": normal_prior(), "z": normal_prior()},
        "rate": normal_prior(),
        "sigma": bmb.Prior("HalfNormal", sigma=1),
    }
    return bmb.Model(formula, data, priors=priors, center_predictors=False)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize(
    ("response", "constrained", "bounds_from_data"),
    [
        ("truncated(y, lower, upper)", False, True),
        ("truncated(y, -1, 2)", False, False),
        ("constrained(y, -1, 2)", True, False),
    ],
)
def test_nonlinear_truncated_prediction_and_new_data_log_likelihood(
    response, constrained, bounds_from_data
):
    model = truncated_nonlinear_model(truncated_nonlinear_data(), response)
    model.set_alias({"y": "outcome"})
    idata = model.fit(draws=3, chains=1, random_seed=108)
    original_model = model.backend.model
    original_data = {
        name: model.backend.model[name].get_value().copy()
        for name in ("y_data", "lower_data", "upper_data")
        if name in model.backend.model.named_vars
    }
    new_data = pd.DataFrame({"x": [0.2, -0.4], "z": [0.5, -1.0]})
    in_sample = model.predict(idata, kind="response", inplace=False)
    assert in_sample.posterior["mu"].shape == (1, 3, 4)
    assert in_sample.posterior_predictive[model.response_term.label].shape == (1, 3, 4)
    params = model.predict(idata, data=new_data, kind="response_params", inplace=False)
    assert params.predictions["mu"].shape == (1, 3, 2)
    if bounds_from_data:
        assert "lower_data" not in params.predictions_constant_data
        assert "upper_data" not in params.predictions_constant_data
    single_params = model.predict(
        idata, data=new_data.iloc[:1], kind="response_params", inplace=False
    )
    assert single_params.predictions["mu"].shape == (1, 3, 1)
    predictions = model.predict(idata, data=new_data, kind="response", inplace=False)
    samples = predictions.predictions[model.response_term.label]
    assert samples.shape == (1, 3, 2)
    single = model.predict(idata, data=new_data.iloc[:1], kind="response", inplace=False)
    assert single.predictions[model.response_term.label].shape == (1, 3, 1)

    bounds_data = new_data.assign(y=[0.1, 0.8])
    lower, upper = -1.0, 2.0
    if bounds_from_data:
        bounds_data = bounds_data.assign(lower=[-0.8, 0.3], upper=[0.8, 1.6])
        lower, upper = bounds_data.lower.to_numpy(), bounds_data.upper.to_numpy()
    if constrained:
        assert (samples > lower).all() and (samples < upper).all()
    else:
        conditional = model.predict(
            idata, data=bounds_data, kind="response_conditional", inplace=False
        )
        conditional_samples = conditional.predictions[model.response_term.label]
        assert (conditional_samples > lower).all()
        assert (conditional_samples < upper).all()
        if bounds_from_data:
            conditional_single = model.predict(
                idata, data=bounds_data.iloc[:1], kind="response_conditional", inplace=False
            )
            assert conditional_single.predictions[model.response_term.label].shape == (1, 3, 1)
            with pytest.raises(
                ValueError, match="Use kind='response' for unconditional predictions"
            ):
                model.predict(
                    idata,
                    data=bounds_data.drop(columns="upper"),
                    kind="response_conditional",
                    inplace=False,
                )

    likelihood = model.compute_log_likelihood(idata, data=bounds_data, inplace=False)
    actual = likelihood.log_likelihood[model.response_term.label].to_numpy()
    posterior = idata.posterior
    mu = (
        posterior["a_Intercept"].to_numpy()[..., None]
        + posterior["a_z"].to_numpy()[..., None] * bounds_data.z.to_numpy()
        + posterior["rate"].to_numpy()[..., None] * bounds_data.x.to_numpy()
    )
    np.testing.assert_allclose(predictions.predictions["mu"], mu)
    sigma = idata.posterior["sigma"].to_numpy()[..., None]
    from scipy.stats import truncnorm

    expected = truncnorm.logpdf(
        bounds_data.y.to_numpy(),
        (np.asarray(lower) - mu) / sigma,
        (np.asarray(upper) - mu) / sigma,
        loc=mu,
        scale=sigma,
    )
    np.testing.assert_allclose(actual, expected, atol=5e-7)
    if bounds_from_data:
        with pytest.raises(
            ValueError, match="Truncated response log-likelihood requires bound variables"
        ):
            model.compute_log_likelihood(
                idata, data=bounds_data.drop(columns="upper"), inplace=False
            )

    assert model.backend.model is original_model
    for name, values in original_data.items():
        np.testing.assert_array_equal(model.backend.model[name].get_value(), values)


def vector_parent_data():
    data = linear_data(6)
    data["y"] = pd.Categorical(
        ["middle", "last", "first"] * 2, categories=["middle", "first", "last"], ordered=True
    )
    data["c1"] = [1, 2, 0, 3, 1, 2]
    data["c2"] = [2, 0, 3, 1, 2, 1]
    data["c3"] = [1, 2, 1, 0, 1, 1]
    data["n"] = data[["c1", "c2", "c3"]].sum(axis=1)
    data["group"] = ["g", "h"] * 3
    return data


@pytest.mark.parametrize("family", ["categorical", "multinomial", "dirichlet_multinomial"])
@pytest.mark.parametrize("mixed", [False, True])
def test_nonlinear_vector_parent_matches_pymc(family, mixed):
    data = vector_parent_data()
    response = "y" if family == "categorical" else "counts(c1, c2, c3, n=n)"
    parent = "a" if family == "dirichlet_multinomial" else "p"
    size = 3 if family == "dirichlet_multinomial" else 2
    rhs = "z + nl(shared + rate * x)" if mixed else "shared + rate * x"
    formula = bmb.Formula(
        f"{response} ~ {rhs}", "rate ~ 1 + z", "shared ~ 1", nlpars=("rate", "shared")
    )
    priors = {"shared": normal_prior(), "rate": {"Intercept": normal_prior(), "z": normal_prior()}}
    if mixed:
        priors[parent] = {"Intercept": normal_prior(), "z": normal_prior()}
    model = bmb.Model(formula, data, family=family, priors=priors, center_predictors=False)
    model.build()
    backend = model.backend.model
    dim = next(iter(backend.__bambi_attrs__["response_coords"]))
    coord_key = "response_coords" if size == 3 else "response_coords_reduced"
    coef_dim = next(iter(backend.__bambi_attrs__[coord_key]))
    assert backend.named_vars_to_dims["rate"] == ("__obs__", coef_dim)
    assert backend.named_vars_to_dims[parent] == ("__obs__", dim)
    assert backend.named_vars_to_dims["shared"] == ()
    levels = ["middle", "first", "last"] if family == "categorical" else ["c1", "c2", "c3"]
    assert list(backend.coords[dim]) == levels
    if size == 2:
        assert list(backend.coords[coef_dim]) == levels[1:]
    point = {
        "shared": np.array(0.4),
        "rate_Intercept": np.linspace(-0.3, 0.5, size),
        "rate_z": np.linspace(0.2, -0.1, size),
    }
    with pm.Model() as direct:
        shared = pm.Normal("shared", 0, 2)
        intercept = pm.Normal("rate_Intercept", 0, 2, shape=size)
        slope = pm.Normal("rate_z", 0, 2, shape=size)
        eta = shared + (intercept + data.z.to_numpy()[:, None] * slope) * data.x.to_numpy()[:, None]
        if mixed:
            base = pm.Normal("Intercept", 0, 2, shape=size)
            linear = pm.Normal("z", 0, 2, shape=size)
            eta += base + data.z.to_numpy()[:, None] * linear
            point.update(Intercept=np.linspace(0.1, 0.3, size), z=np.linspace(-0.2, 0.1, size))
        if family == "dirichlet_multinomial":
            value = pt.exp(eta)
            pm.DirichletMultinomial(
                "observed", n=data.n, a=value, observed=data[["c1", "c2", "c3"]]
            )
        else:
            value = pt.special.softmax(
                pt.concatenate([pt.zeros((len(data), 1)), eta], axis=1), axis=1
            )
            if family == "categorical":
                pm.Categorical("observed", p=value, observed=data.y.cat.codes)
            else:
                pm.Multinomial("observed", n=data.n, p=value, observed=data[["c1", "c2", "c3"]])
        pm.Deterministic("value", value)
    np.testing.assert_allclose(backend.compile_logp()(point), direct.compile_logp()(point))
    actual = backend.compile_fn(backend[parent], inputs=backend.free_RVs)(point)
    expected = direct.compile_fn(direct["value"], inputs=direct.free_RVs)(point)
    np.testing.assert_allclose(actual, expected)
    if parent == "p":
        np.testing.assert_allclose(actual.sum(axis=-1), 1)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize(
    ("family", "expression"),
    [
        ("categorical", "z + nl(0.7)"),
        ("multinomial", "shared * x"),
        ("dirichlet_multinomial", "rate * x + shared"),
    ],
)
def test_nonlinear_vector_prediction(family, expression):
    from scipy.special import softmax

    data = vector_parent_data()
    response = "y" if family == "categorical" else "counts(c1, c2, c3, n=n)"
    names = tuple(name for name in ("shared", "rate") if name in expression)
    additionals = ("rate ~ 1 + z",) if "rate" in names else ()
    model = bmb.Model(
        bmb.Formula(f"{response} ~ {expression}", *additionals, nlpars=names), data, family=family
    )
    model.set_alias({model.family.likelihood.parent: {model.family.likelihood.parent: "parameter"}})
    idata = model.fit(draws=3, chains=1, random_seed=101)
    posterior = idata.posterior
    for count in (len(data), 2, 1):
        observed = data.iloc[:count]
        new_data = None if count == len(data) else observed.copy()
        result = model.predict(
            idata, data=new_data, kind="response", inplace=False, random_seed=102
        )
        x = observed.x.to_numpy()[None, None, :, None]
        centered_z = (observed.z.to_numpy() - data.z.mean())[None, None, :, None]
        if family == "categorical":
            eta = posterior.Intercept_centered.values[:, :, None, :]
            eta = eta + posterior.z.values[:, :, None, :] * centered_z + 0.7
        elif family == "multinomial":
            eta = np.broadcast_to(posterior.shared.values[:, :, None, None] * x, (1, 3, count, 2))
        else:
            rate = posterior.rate_Intercept_centered.values[:, :, None, :]
            rate = rate + posterior.rate_z.values[:, :, None, :] * centered_z
            eta = rate * x + posterior.shared.values[:, :, None, None]
        if family == "dirichlet_multinomial":
            expected = np.exp(eta)
        else:
            reference = np.zeros_like(eta[..., :1])
            expected = softmax(np.concatenate([reference, eta], axis=-1), axis=-1)
        group = result.posterior if new_data is None else result.predictions
        np.testing.assert_allclose(group["parameter"], expected)
        sampled = result.posterior_predictive if new_data is None else result.predictions
        draws = sampled[model.response_term.label].values
        if family == "categorical":
            assert draws.shape == (1, 3, count)
            assert np.isin(draws, [0, 1, 2]).all()
            expected_loglik = np.log(expected[:, :, np.arange(count), observed.y.cat.codes])
        else:
            np.testing.assert_array_equal(draws.sum(axis=-1), np.full((1, 3, count), 4))
            counts = observed[["c1", "c2", "c3"]].to_numpy()
            if family == "multinomial":
                expected_loglik = stats.multinomial.logpmf(counts, n=observed.n, p=expected)
            else:
                expected_loglik = stats.dirichlet_multinomial.logpmf(
                    counts, alpha=expected, n=observed.n
                )
        likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
        np.testing.assert_allclose(
            likelihood.log_likelihood[model.response_term.label], expected_loglik
        )


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
@pytest.mark.parametrize("family", ["categorical", "dirichlet_multinomial"])
def test_nonlinear_vector_group_prediction(monkeypatch, sparse_dot, family):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = vector_parent_data()
    response = "y" if family == "categorical" else "counts(c1, c2, c3, n=n)"
    parent = "a" if family == "dirichlet_multinomial" else "p"
    dim = "y_dim_reduced" if family == "categorical" else "c1_c2_c3_dim"
    model = bmb.Model(
        bmb.Formula(f"{response} ~ rate * x", "rate ~ 0 + z + (1 | group)", nlpars=("rate",)),
        data,
        family=family,
        center_predictors=False,
    )
    model.set_alias({"rate": {"rate": "r", "1|group": "group_effect", "z": "slope"}})
    idata = model.fit(draws=3, chains=1, random_seed=103)
    assert idata.posterior["group_effect"].dims == ("chain", "draw", "group_dim", dim)
    for groups in (["g", "h"], ["g"], ["new", "new"]):
        new_data = pd.DataFrame(
            {"x": [0.3] * len(groups), "z": [0.2] * len(groups), "group": groups, "n": 4}
        )
        result = model.predict(idata, data=new_data, inplace=False, random_seed=104)
        values = result.predictions[parent].values
        assert values.shape == (1, 3, len(groups), 3)
        if parent == "p":
            np.testing.assert_allclose(values.sum(axis=-1), 1)
        if groups[0] == "new":
            np.testing.assert_allclose(values[:, :, 0], values[:, :, 1])
        else:
            effects = idata.posterior.group_effect.sel(group_dim=groups).values
            slope = idata.posterior.slope.values[:, :, None, :]
            eta = (effects + 0.2 * slope) * 0.3
            weights = np.exp(np.concatenate([np.zeros((*eta.shape[:-1], 1)), eta], axis=-1))
            expected = weights / weights.sum(axis=-1, keepdims=True)
            if parent == "a":
                expected = np.exp(eta)
            np.testing.assert_allclose(values, expected)
        population = model.predict(
            idata, data=new_data, include_group_specific=False, inplace=False
        )
        eta = idata.posterior.slope.values[:, :, None, :] * 0.2 * 0.3
        weights = np.exp(np.concatenate([np.zeros((*eta.shape[:-1], 1)), eta], axis=-1))
        expected = weights / weights.sum(axis=-1, keepdims=True)
        if parent == "a":
            expected = np.exp(eta)
        np.testing.assert_allclose(
            population.predictions[parent], np.broadcast_to(expected, values.shape)
        )


@pytest.mark.usefixtures("mock_pymc_sample")
def test_nonlinear_categorical_interpret():
    from bambi.interpret.effects import comparisons, predictions, slopes

    model = bmb.Model(
        bmb.Formula("y ~ shared * x + rate", "rate ~ 0 + z", nlpars=("shared", "rate")),
        vector_parent_data(),
        family="categorical",
    )
    idata = model.fit(draws=5, chains=2, random_seed=105)
    conditional = {"x": [-0.2, 0.3], "z": [0.1]}
    result = predictions(model, idata, conditional=conditional).summary
    assert len(result) == 6
    assert set(result.y_dim) == {"middle", "first", "last"}
    np.testing.assert_allclose(result.groupby("x").estimate.sum(), 1)
    for function, kwargs in (
        (comparisons, {"contrast": {"x": [-0.2, 0.3]}}),
        (slopes, {"wrt": "x"}),
    ):
        result = function(model, idata, conditional={"z": [0.1]}, **kwargs).summary
        assert set(result.y_dim) == {"middle", "first", "last"}
        np.testing.assert_allclose(result.estimate.sum(), 0, atol=1e-8)


def test_nonlinear_dirichlet_multinomial_constant_concentration():
    data = vector_parent_data()
    model = bmb.Model(
        bmb.Formula("counts(c1, c2, c3, n=n) ~ 0 + nl(0.7)"), data, family="dirichlet_multinomial"
    )
    model.build()
    np.testing.assert_allclose(
        model.backend.model["a"].eval(), np.full((len(data), 3), np.exp(0.7))
    )


def fixed_parameter_idata():
    posterior = xr.Dataset(coords={"chain": [2, 4], "draw": [10, 11, 12]})
    return xr.DataTree.from_dict({"posterior": posterior})


def test_nonlinear_fixed_gaussian_prediction_and_log_likelihood():
    data = pd.DataFrame({"x": [-1.0, 0.0, 1.0], "y": [-0.6, 0.1, 0.9]})
    formula = bmb.Formula("y ~ 0 + nl(0.7*x)")
    model = bmb.Model(formula, data, priors={"sigma": 0.5})
    model.build()
    assert not model.backend.model.free_RVs

    idata = fixed_parameter_idata()
    original_graph = model.backend.model
    original_mu = model.backend.model["mu"].eval().copy()
    params = model.predict(idata, kind="response_params", inplace=False)
    assert params.posterior.coords["chain"].to_numpy().tolist() == [2, 4]
    assert params.posterior.coords["draw"].to_numpy().tolist() == [10, 11, 12]
    np.testing.assert_allclose(
        params.posterior["mu"], np.broadcast_to(original_mu, (2, 3, len(data)))
    )
    inplace_idata = fixed_parameter_idata()
    assert model.predict(inplace_idata, kind="response_params") is None
    assert "mu" in inplace_idata.posterior
    predictions = model.predict(params, kind="response", inplace=False, random_seed=109)
    samples = predictions.posterior_predictive["y"].to_numpy()
    assert samples.shape == (2, 3, len(data))
    assert np.unique(samples).size > 1
    fresh_predictions = model.predict(params, kind="response", inplace=False, random_seed=114)
    assert not np.array_equal(samples, fresh_predictions.posterior_predictive["y"].to_numpy())
    from scipy.stats import norm

    expected_log_likelihood = norm.logpdf(data.y.to_numpy(), loc=original_mu, scale=0.5)
    likelihood = model.compute_log_likelihood(params, inplace=False)
    np.testing.assert_allclose(
        likelihood.log_likelihood["y"],
        np.broadcast_to(expected_log_likelihood, (2, 3, len(data))),
    )
    new_data_likelihood = model.compute_log_likelihood(params, data=data.iloc[:2], inplace=False)
    np.testing.assert_allclose(
        new_data_likelihood.log_likelihood["y"],
        np.broadcast_to(expected_log_likelihood[:2], (2, 3, 2)),
    )

    for count in (2, 1):
        new_data = data.iloc[:count][["x"]]
        predicted_params = model.predict(
            params, data=new_data, kind="response_params", inplace=False
        )
        predicted_response = model.predict(
            params, data=new_data, kind="response", inplace=False, random_seed=110
        )
        assert predicted_params.predictions["mu"].shape == (2, 3, count)
        assert predicted_response.predictions["y"].shape == (2, 3, count)

    truncated_data = data.assign(lower=[-1.0, -0.5, 0.0], upper=[0.0, 0.5, 1.0])
    truncated_model = bmb.Model(
        bmb.Formula("truncated(y, lower, upper) ~ 0 + nl(0.7*x)"),
        truncated_data,
        priors={"sigma": 0.5},
    )
    truncated_model.build()
    bounded = truncated_model.predict(
        fixed_parameter_idata(), kind="response_conditional", inplace=False, random_seed=113
    )
    bounded_samples = bounded.posterior_predictive["y"].to_numpy()
    assert bounded_samples.shape == (2, 3, len(data))
    assert (bounded_samples > truncated_data.lower.to_numpy()).all()
    assert (bounded_samples < truncated_data.upper.to_numpy()).all()

    assert model.backend.model is original_graph
    np.testing.assert_allclose(model.backend.model["mu"].eval(), original_mu)


@pytest.mark.parametrize("family", ["categorical", "multinomial"])
def test_nonlinear_fixed_vector_prediction_and_log_likelihood(family):
    data = vector_parent_data()
    response = "y" if family == "categorical" else "counts(c1, c2, c3, n=n)"
    model = bmb.Model(bmb.Formula(f"{response} ~ 0 + nl(0.7)"), data, family=family)
    model.set_alias({"p": {"p": "probability"}})
    model.build()
    assert not model.backend.model.free_RVs

    idata = fixed_parameter_idata()
    params = model.predict(idata, kind="response_params", inplace=False)
    parent = "probability"
    probabilities = params.posterior[parent].to_numpy()
    assert probabilities.shape == (2, 3, len(data), 3)
    expected_probabilities = np.exp([0, 0.7, 0.7])
    expected_probabilities /= expected_probabilities.sum()
    np.testing.assert_allclose(
        probabilities, np.broadcast_to(expected_probabilities, probabilities.shape)
    )
    response_prediction = model.predict(params, kind="response", inplace=False, random_seed=111)
    samples = response_prediction.posterior_predictive[model.response_term.label].to_numpy()
    sample_shape = (2, 3, len(data), 3) if family == "multinomial" else (2, 3, len(data))
    assert samples.shape == sample_shape

    if family == "categorical":
        assert set(np.unique(samples)).issubset({0, 1, 2})
        expected = np.log(probabilities[:, :, np.arange(len(data)), data.y.cat.codes])
    else:
        assert np.all(samples >= 0)
        assert np.all(samples.sum(axis=-1) == data.n.to_numpy())
        from scipy.special import gammaln

        counts = data[["c1", "c2", "c3"]].to_numpy()
        expected = (
            gammaln(data.n.to_numpy() + 1)
            - gammaln(counts + 1).sum(axis=-1)
            + (counts * np.log(probabilities)).sum(axis=-1)
        )

    likelihood = model.compute_log_likelihood(params, inplace=False)
    np.testing.assert_allclose(likelihood.log_likelihood[model.response_term.label], expected)

    for count in (2, 1):
        observed = data.iloc[:count].copy()
        if family == "multinomial":
            observed["n"] = np.arange(count) + 5
            observed["c3"] = observed.n - observed.c1 - observed.c2
        new_data = observed[["x", "z"]].copy()
        if family == "multinomial":
            new_data["n"] = observed.n.to_numpy()
        predicted_params = model.predict(
            params, data=new_data, kind="response_params", inplace=False
        )
        predicted_response = model.predict(
            params, data=new_data, kind="response", inplace=False, random_seed=112
        )
        assert predicted_params.predictions[parent].shape == (2, 3, count, 3)
        response_shape = (2, 3, count, 3) if family == "multinomial" else (2, 3, count)
        assert predicted_response.predictions[model.response_term.label].shape == response_shape
        np.testing.assert_array_equal(predicted_response.predictions.chain, [2, 4])
        np.testing.assert_array_equal(predicted_response.predictions.draw, [10, 11, 12])
        if family == "multinomial":
            np.testing.assert_array_equal(
                predicted_response.predictions[model.response_term.label].to_numpy().sum(axis=-1),
                np.broadcast_to(observed.n.to_numpy(), (2, 3, count)),
            )
            counts = observed[["c1", "c2", "c3"]].to_numpy()
            expected_new = (
                gammaln(observed.n.to_numpy() + 1)
                - gammaln(counts + 1).sum(axis=-1)
                + (counts * np.log(probabilities[:, :, :count])).sum(axis=-1)
            )
        else:
            expected_new = np.log(probabilities[:, :, np.arange(count), observed.y.cat.codes])
        new_likelihood = model.compute_log_likelihood(params, data=observed, inplace=False)
        pointwise = new_likelihood.log_likelihood[model.response_term.label]
        assert pointwise.dims == ("chain", "draw", "__obs__")
        np.testing.assert_allclose(pointwise, expected_new)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("categories", [2, 3])
def test_nonlinear_vector_dependencies_and_offsets(categories):
    data = vector_parent_data()
    if categories == 2:
        data = data[data.y != "last"].copy()
        data["y"] = data.y.cat.remove_unused_categories()
    formula = bmb.Formula(
        "y ~ offset(x) + nl(curve)",
        "curve ~ rate * x + shared",
        "rate ~ 0 + z",
        nlpars=("curve", "rate", "shared"),
    )
    model = bmb.Model(formula, data, family="categorical", center_predictors=False)
    idata = model.fit(draws=3, chains=1, random_seed=106)
    for count in (len(data), 1):
        new_data = data.iloc[:count]
        result = model.predict(idata, data=new_data, inplace=False)
        eta = (
            idata.posterior.rate_z.values[:, :, None, :]
            * new_data.z.to_numpy()[None, None, :, None]
            * new_data.x.to_numpy()[None, None, :, None]
            + idata.posterior.shared.values[:, :, None, None]
            + idata.posterior.Intercept.values[:, :, None, :]
            + new_data.x.to_numpy()[None, None, :, None]
        )
        weights = np.exp(np.concatenate([np.zeros((*eta.shape[:-1], 1)), eta], axis=-1))
        np.testing.assert_allclose(
            result.predictions.p, weights / weights.sum(axis=-1, keepdims=True)
        )


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
def test_nonlinear_count_interpret(family):
    from bambi.interpret.effects import predictions

    model = bmb.Model(
        bmb.Formula(
            "counts(c1, c2, c3, n=n) ~ shared * x + rate",
            "rate ~ 0 + z",
            nlpars=("shared", "rate"),
        ),
        vector_parent_data(),
        family=family,
    )
    idata = model.fit(draws=5, chains=2, random_seed=107)
    conditional = {"x": [-0.2, 0.3], "z": [0.1], "n": [4]}
    summary = predictions(model, idata, conditional=conditional).summary
    assert len(summary) == 6
    assert set(summary.c1_c2_c3_dim) == {"c1", "c2", "c3"}
    data = pd.DataFrame({"x": [-0.2, 0.3], "z": [0.1, 0.1], "n": [4, 4]})
    result = model.predict(idata, data=data, inplace=False)
    expected = result.predictions[model.family.likelihood.parent].mean(("chain", "draw"))
    actual = summary.pivot(index="x", columns="c1_c2_c3_dim", values="estimate")
    np.testing.assert_allclose(actual[["c1", "c2", "c3"]], expected)


def exponential_competing_risks_loglik(mu, data):
    times = data.time.to_numpy()
    status = (data.status.to_numpy() == "right").astype(int)
    causes = np.array(
        [0 if cause == "none" else ["cause_a", "cause_b"].index(cause) + 1 for cause in data.cause]
    )
    log_survival = -times * (1 / mu).sum(axis=-1)
    rows = np.arange(len(data))
    cause_index = np.maximum(causes - 1, 0)
    event_loglik = log_survival - np.log(mu[rows, cause_index])
    return np.where(status == 1, log_survival, event_loglik)


@pytest.mark.usefixtures("mock_pymc_sample")
def test_nonlinear_competing_risks_predictions_and_log_likelihood():
    from bambi.interpret.effects import predictions

    data = competing_risks_nonlinear_data()
    formula = bmb.Formula(
        "risks(time, status, cause) ~ 0 + nl(rate*x) + z",
        "rate ~ 1 + z",
        nlpars=("rate",),
    )
    prior = bmb.Prior("Normal", mu=0, sigma=2)
    model = bmb.Model(
        formula,
        data,
        family="exponential",
        priors={"rate": {"Intercept": prior, "z": prior}, "z": prior},
        center_predictors=False,
    )
    model.set_alias({"mu": {"mu": "event_rate"}})
    model.build()
    assert model.backend.model.named_vars_to_dims["rate"] == ("__obs__", "cause_dim")
    assert model.backend.model.named_vars_to_dims["event_rate"] == ("__obs__", "cause_dim")

    original_data = {
        name: model.backend.model[name].get_value().copy()
        for name in ("time_data", "status_data", "cause_data")
    }
    idata = model.fit(draws=3, chains=2)
    coefficient_values = {
        "rate_Intercept": np.array([0.25, -0.15]),
        "rate_z": np.array([0.1, 0.2]),
        "z": np.array([0.3, -0.2]),
    }
    posterior = idata.posterior.to_dataset().copy(deep=True)
    for name, value in coefficient_values.items():
        posterior[name] = posterior[name].copy(data=np.broadcast_to(value, posterior[name].shape))
    idata["posterior"] = posterior

    x = data.x.to_numpy()[:, None]
    z = data.z.to_numpy()[:, None]
    mu = np.exp(
        (coefficient_values["rate_Intercept"] + coefficient_values["rate_z"] * z) * x
        + coefficient_values["z"] * z
    )
    parameter_prediction = model.predict(idata, kind="response_params", inplace=False)
    expected_mu = np.broadcast_to(mu, (2, 3, *mu.shape))
    np.testing.assert_allclose(parameter_prediction.posterior["event_rate"], expected_mu)
    training_loglik = model.compute_log_likelihood(idata, inplace=False)
    expected_loglik = exponential_competing_risks_loglik(mu, data)
    np.testing.assert_allclose(
        training_loglik.log_likelihood["time"],
        np.broadcast_to(expected_loglik, (2, 3, len(data))),
    )

    for new_data in (
        data.iloc[[0]].assign(time=[1.25], status=["right"], cause=["none"]),
        data.iloc[[0, 1]].assign(status=["right", "event"], cause=["none", "cause_b"]),
    ):
        parameters = model.predict(
            idata, data=new_data[["x", "z"]], kind="response_params", inplace=False
        )
        predicted_mu = parameters.predictions["event_rate"].values[0, 0]
        new_x = new_data.x.to_numpy()[:, None]
        new_z = new_data.z.to_numpy()[:, None]
        eta = (
            coefficient_values["rate_Intercept"] + coefficient_values["rate_z"] * new_z
        ) * new_x + coefficient_values["z"] * new_z
        np.testing.assert_allclose(predicted_mu, np.exp(eta))
        likelihood = model.compute_log_likelihood(idata, data=new_data, inplace=False)
        expected = exponential_competing_risks_loglik(predicted_mu, new_data)
        np.testing.assert_allclose(
            likelihood.log_likelihood["time"], np.broadcast_to(expected, (2, 3, len(new_data)))
        )
        draws = model.predict(
            idata,
            data=new_data[["x", "z"]],
            kind="response",
            inplace=False,
            random_seed=21,
        ).predictions["time"]
        assert draws.shape == (2, 3, len(new_data))

    conditional = model.predict(
        idata, data=data, kind="response_conditional", inplace=False, random_seed=22
    ).predictions["time"]
    right_censored = data.status.to_numpy() == "right"
    assert (conditional.values[..., right_censored] >= data.time.to_numpy()[right_censored]).all()
    single = model.predict(
        idata,
        data=data.iloc[[1]],
        kind="response_conditional",
        inplace=False,
        random_seed=23,
    ).predictions["time"]
    assert single.shape == (2, 3, 1)
    assert (single.values >= data.time.iloc[[1]].to_numpy()).all()

    summary = predictions(model, idata, conditional={"x": [0.2], "z": [0.1]}).summary
    assert set(summary.cause_dim) == {"cause_a", "cause_b"}
    for name, values in original_data.items():
        np.testing.assert_array_equal(model.backend.model[name].get_value(), values)


@pytest.mark.usefixtures("mock_pymc_sample")
def test_nonlinear_competing_risks_marginal_cause_parameter_dependency():
    data = competing_risks_nonlinear_data()
    formula = bmb.Formula(
        "risks(time, status, cause) ~ 0 + nl(rate*x + alpha)",
        "rate ~ 1 + z",
        nlpars=("rate",),
    )
    model = bmb.Model(formula, data, family="weibull_ph", center_predictors=False)
    model.build()
    assert model.backend.model.named_vars_to_dims["alpha"] == ("cause_dim",)

    idata = model.fit(draws=3, chains=2)
    posterior = idata.posterior.to_dataset().copy(deep=True)
    values = {
        "alpha": np.array([0.5, 1.2]),
        "rate_Intercept": np.array([0.1, 0.3]),
        "rate_z": np.array([0.05, -0.1]),
    }
    for name, value in values.items():
        posterior[name] = posterior[name].copy(data=np.broadcast_to(value, posterior[name].shape))
    idata["posterior"] = posterior

    params = model.predict(idata, kind="response_params", inplace=False)
    x = data.x.to_numpy()[:, None]
    z = data.z.to_numpy()[:, None]
    eta = (values["rate_Intercept"] + values["rate_z"] * z) * x + values["alpha"]
    expected_lam = np.exp(eta)
    np.testing.assert_allclose(
        params.posterior["lam"], np.broadcast_to(expected_lam, (2, 3, *expected_lam.shape))
    )


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_nonlinear_competing_risks_group_prediction(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = competing_risks_nonlinear_data().assign(group=["g", "h"] * 4)
    model = bmb.Model(
        bmb.Formula(
            "risks(time, status, cause) ~ rate*x", "rate ~ 0 + z + (1 | group)", nlpars=("rate",)
        ),
        data,
        family="exponential",
        center_predictors=False,
    )
    model.set_alias(
        {
            "mu": {"mu": "event_scale"},
            "rate": {"rate": "r", "z": "slope", "1|group": "group_effect"},
        }
    )
    idata = model.fit(draws=3, chains=1, random_seed=126)
    assert idata.posterior.group_effect.dims == ("chain", "draw", "group_dim", "cause_dim")
    fixed_eta = 0.2 * idata.posterior.slope.values * 0.3
    for groups in (["g", "h"], ["g"], ["new", "new"]):
        new_data = pd.DataFrame({"x": 0.3, "z": 0.2, "group": groups})
        result = model.predict(idata, data=new_data, inplace=False, random_seed=127)
        values = result.predictions.event_scale.values
        assert values.shape == (1, 3, len(groups), 2)
        assert np.isfinite(values).all() and (values > 0).all()
        if groups[0] == "new":
            np.testing.assert_allclose(values[:, :, 0], values[:, :, 1])
        else:
            effects = idata.posterior.group_effect.sel(group_dim=groups).values
            expected = np.exp(fixed_eta[:, :, None, :] + effects * 0.3)
            np.testing.assert_allclose(values, expected)
        population = model.predict(
            idata, data=new_data, include_group_specific=False, inplace=False
        )
        expected = np.exp(fixed_eta[:, :, None, :])
        np.testing.assert_allclose(
            population.predictions.event_scale, np.broadcast_to(expected, values.shape)
        )


def test_nonlinear_fixed_competing_risks_time_and_cause():
    data = competing_risks_nonlinear_data()
    model = bmb.Model(
        bmb.Formula("risks(time, status, cause) ~ 0 + nl(0.4)"), data, family="exponential"
    )
    model.set_alias({"mu": {"mu": "event_scale"}, "time": "outcome"})
    model.build()
    assert not model.backend.model.free_RVs
    idata = fixed_parameter_idata()
    for new_data in (None, data.iloc[:1]):
        result = model.predict(
            idata, data=new_data, kind="time_and_cause", inplace=False, random_seed=128
        )
        group = result.posterior_predictive if new_data is None else result.predictions
        count = len(data) if new_data is None else len(new_data)
        assert group.outcome_time.shape == (2, 3, count)
        assert group.outcome_cause.shape == (2, 3, count)
        np.testing.assert_array_equal(group.chain, [2, 4])
        np.testing.assert_array_equal(group.draw, [10, 11, 12])
        assert (group.outcome_time > 0).all()
        assert set(np.unique(group.outcome_cause)).issubset({1, 2})


@pytest.mark.usefixtures("mock_pymc_sample")
def test_nonlinear_numeric_vector_response_fixed_values_and_predictions():
    from scipy.stats import multivariate_normal

    data = vector_normal_data()
    formula = bmb.Formula(
        "join(y1, y2) ~ 0 + nl(rate*x) + z",
        "rate ~ 1 + z",
        nlpars=("rate",),
    )
    prior = bmb.Prior("Normal", mu=0, sigma=2)
    model = bmb.Model(
        formula,
        data,
        family=vector_normal_family(),
        priors={"rate": {"Intercept": prior, "z": prior}, "z": prior},
        extra_namespace={"join": join_vector_response},
        center_predictors=False,
    )
    model.set_alias({"mu": {"mu": "mean"}, "y1": "outcome"})
    model.build()
    backend = model.backend.model
    response_name = model.response_term.label
    event_dim = f"{model.response_term.name}_dim"
    assert response_name == "outcome"
    assert backend.named_vars_to_dims["mean"] == ("__obs__", event_dim)
    assert backend.named_vars_to_dims["rate"] == ("__obs__", event_dim)
    np.testing.assert_array_equal(backend.coords[event_dim], [0, 1])

    coefficients = {
        "rate_Intercept": np.array([0.3, -0.2]),
        "rate_z": np.array([0.1, 0.25]),
        "z": np.array([-0.15, 0.2]),
    }
    z = data.z.to_numpy()[:, None]
    x = data.x.to_numpy()[:, None]
    expected_mu = (coefficients["rate_Intercept"] + coefficients["rate_z"] * z) * x
    expected_mu += coefficients["z"] * z
    actual_mu = backend.compile_fn(backend["mean"], inputs=backend.free_RVs)(coefficients)
    np.testing.assert_allclose(actual_mu, expected_mu)

    idata = model.fit(draws=3, chains=2, random_seed=129)
    posterior = idata.posterior.to_dataset().copy(deep=True)
    for name, value in coefficients.items():
        posterior[name] = posterior[name].copy(data=np.broadcast_to(value, posterior[name].shape))
    idata["posterior"] = posterior
    original_response = model.response_term.data.copy()
    params = model.predict(idata, kind="response_params", inplace=False)
    np.testing.assert_allclose(
        params.posterior["mean"], np.broadcast_to(expected_mu, (2, 3, *expected_mu.shape))
    )
    prediction = model.predict(params, kind="response", inplace=False, random_seed=130)
    draws = prediction.posterior_predictive[response_name]
    assert draws.shape == (2, 3, len(data), 2)
    likelihood = model.compute_log_likelihood(params, inplace=False)
    expected_logp = np.array(
        [
            multivariate_normal.logpdf(row, mean=mu, cov=VECTOR_NORMAL_COV)
            for row, mu in zip(data[["y1", "y2"]].to_numpy(), expected_mu)
        ]
    )
    np.testing.assert_allclose(
        likelihood.log_likelihood[response_name], np.broadcast_to(expected_logp, (2, 3, len(data)))
    )

    for count in (2, 1):
        new_data = data.iloc[:count]
        predicted = model.predict(
            params,
            data=new_data[["x", "z"]],
            kind="response_params",
            inplace=False,
        )
        new_mu = expected_mu[:count]
        np.testing.assert_allclose(
            predicted.predictions["mean"], np.broadcast_to(new_mu, (2, 3, count, 2))
        )
        response_draws = model.predict(
            params,
            data=new_data[["x", "z"]],
            kind="response",
            inplace=False,
            random_seed=131,
        )
        assert response_draws.predictions[response_name].shape == (2, 3, count, 2)
        new_likelihood = model.compute_log_likelihood(params, data=new_data, inplace=False)
        np.testing.assert_allclose(
            new_likelihood.log_likelihood[response_name],
            np.broadcast_to(expected_logp[:count], (2, 3, count)),
        )
    np.testing.assert_array_equal(model.response_term.data, original_response)
    from bambi.interpret.effects import predictions

    summary = predictions(model, params, conditional={"x": [-0.2, 0.3], "z": [0.1]}).summary
    actual = summary.pivot(index="x", columns=event_dim, values="estimate")
    rate = coefficients["rate_Intercept"] + 0.1 * coefficients["rate_z"]
    expected = rate * np.array([[-0.2], [0.3]]) + 0.1 * coefficients["z"]
    np.testing.assert_allclose(actual[[0, 1]], expected)


def test_nonlinear_numeric_vector_response_constant_expression():
    from scipy.stats import multivariate_normal

    data = vector_normal_data()
    model = bmb.Model(
        bmb.Formula("join(y1, y2) ~ 0 + nl(0.7)"),
        data,
        family=vector_normal_family(),
        extra_namespace={"join": join_vector_response},
    )
    model.build()
    assert not model.backend.model.free_RVs
    result = model.predict(fixed_parameter_idata(), kind="response_params", inplace=False)
    assert result.posterior["mu"].shape == (2, 3, len(data), 2)
    np.testing.assert_allclose(result.posterior["mu"], 0.7)
    likelihood = model.compute_log_likelihood(result, inplace=False)
    expected = np.array(
        [
            multivariate_normal.logpdf(row, mean=[0.7, 0.7], cov=VECTOR_NORMAL_COV)
            for row in data[["y1", "y2"]].to_numpy()
        ]
    )
    np.testing.assert_allclose(
        likelihood.log_likelihood[model.response_term.label],
        np.broadcast_to(expected, (2, 3, len(data))),
    )
    draws = model.predict(result, kind="response", inplace=False, random_seed=132)
    assert draws.posterior_predictive[model.response_term.label].shape == (2, 3, len(data), 2)
    for count in (2, 1):
        new_data = pd.DataFrame(index=range(count))
        params = model.predict(result, data=new_data, kind="response_params", inplace=False)
        assert params.predictions["mu"].shape == (2, 3, count, 2)
        response = model.predict(
            result,
            data=new_data,
            kind="response",
            inplace=False,
            random_seed=133,
        )
        assert response.predictions[model.response_term.label].shape == (2, 3, count, 2)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_nonlinear_numeric_vector_response_group_prediction(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = vector_normal_data().assign(group=["g", "h"] * 3)
    model = bmb.Model(
        bmb.Formula("join(y1, y2) ~ rate*x", "rate ~ 0 + z + (1 | group)", nlpars=("rate",)),
        data,
        family=vector_normal_family(),
        extra_namespace={"join": join_vector_response},
        center_predictors=False,
    )
    model.set_alias(
        {
            "mu": {"mu": "mean"},
            "rate": {"rate": "r", "z": "slope", "1|group": "group_effect"},
            "y1": "outcome",
        }
    )
    idata = model.fit(draws=3, chains=1, random_seed=134)
    event_dim = f"{model.response_term.name}_dim"
    assert idata.posterior.group_effect.dims == ("chain", "draw", "group_dim", event_dim)
    fixed_mean = 0.2 * idata.posterior.slope.values * 0.3
    for groups in (["g", "h"], ["g"], ["new", "new"]):
        new_data = pd.DataFrame({"x": 0.3, "z": 0.2, "group": groups})
        result = model.predict(
            idata, data=new_data, kind="response", inplace=False, random_seed=135
        )
        values = result.predictions["mean"].values
        assert values.shape == (1, 3, len(groups), 2)
        assert result.predictions.outcome.shape == values.shape
        if groups[0] == "new":
            np.testing.assert_allclose(values[:, :, 0], values[:, :, 1])
        else:
            effects = idata.posterior.group_effect.sel(group_dim=groups).values
            expected = fixed_mean[:, :, None, :] + effects * 0.3
            np.testing.assert_allclose(values, expected)
        population = model.predict(
            idata, data=new_data, include_group_specific=False, inplace=False
        )
        np.testing.assert_allclose(
            population.predictions["mean"], np.broadcast_to(fixed_mean[:, :, None, :], values.shape)
        )


@pytest.mark.usefixtures("mock_pymc_sample")
def test_conditional_coefficient_uses_scalar_auxiliary_dimension():
    from scipy.stats import multivariate_t

    family = vector_student_t_family()
    data = vector_normal_data()
    formula = bmb.Formula(
        "join(y1, y2) ~ 0 + nl(0.7)",
        "nu ~ 0 + nl(2 + exp(rate*x))",
        "rate ~ 1 + z",
        nlpars=("rate",),
    )
    prior = bmb.Prior("Normal", mu=0, sigma=1)
    model = bmb.Model(
        formula,
        data,
        family=family,
        priors={"rate": {"Intercept": prior, "z": prior}},
        extra_namespace={"join": join_vector_response},
        center_predictors=False,
    )
    model.set_alias(
        {
            "mu": {"mu": "mean"},
            "nu": {"nu": "df"},
            "rate": {"Intercept": "rate0", "z": "rate_z"},
        }
    )
    model.build()
    backend = model.backend.model
    coefficient = model.parameters["nu"].nonlinear_coefficients["rate"]
    assert coefficient.get_coefs_dim() is None
    assert backend.named_vars_to_dims["rate"] == ("__obs__",)

    point = {"rate0": 0.3, "rate_z": -0.2}
    rate = point["rate0"] + point["rate_z"] * data.z.to_numpy()
    expected_nu = 2 + np.exp(rate * data.x.to_numpy())
    expected_mu = np.full((len(data), 2), 0.7)
    actual_nu, actual_mu = backend.compile_fn(
        [backend["df"], backend["mean"]], inputs=backend.free_RVs
    )(point)
    np.testing.assert_allclose(actual_nu, expected_nu)
    np.testing.assert_allclose(actual_mu, expected_mu)

    with pm.Model() as direct:
        rate0 = pm.Normal("rate0", 0, 1)
        rate_z = pm.Normal("rate_z", 0, 1)
        nu = 2 + pt.exp((rate0 + rate_z * data.z.to_numpy()) * data.x.to_numpy())
        mu = np.full((len(data), 2), 0.7)
        pm.MvStudentT(
            "observed",
            nu=nu,
            mu=mu,
            scale=VECTOR_NORMAL_COV,
            observed=data[["y1", "y2"]].to_numpy(),
        )
    direct_point = {"rate0": point["rate0"], "rate_z": point["rate_z"]}
    actual_logp = backend.compile_logp()(point)
    expected_logp = direct.compile_logp()(direct_point)
    assert np.isfinite(actual_logp) and np.isfinite(expected_logp)
    np.testing.assert_allclose(actual_logp, expected_logp)

    idata = model.fit(draws=3, chains=2, random_seed=221)
    posterior = idata.posterior.to_dataset().copy(deep=True)
    for name, value in point.items():
        posterior[name] = posterior[name].copy(data=np.full(posterior[name].shape, value))
    idata["posterior"] = posterior
    params = model.predict(idata, kind="response_params", inplace=False)
    np.testing.assert_allclose(
        params.posterior["df"], np.broadcast_to(expected_nu, (2, 3, len(data)))
    )
    np.testing.assert_allclose(
        params.posterior["mean"], np.broadcast_to(expected_mu, (2, 3, len(data), 2))
    )
    response_name = model.response_term.label
    prediction = model.predict(params, kind="response", inplace=False, random_seed=222)
    assert prediction.posterior_predictive[response_name].shape == (2, 3, len(data), 2)

    for count in (2, 1):
        new_data = data.iloc[:count]
        predicted = model.predict(
            params, data=new_data[["x", "z"]], kind="response_params", inplace=False
        )
        new_nu = expected_nu[:count]
        assert predicted.predictions["df"].shape == (2, 3, count)
        np.testing.assert_allclose(predicted.predictions["mean"], np.full((2, 3, count, 2), 0.7))
        np.testing.assert_allclose(
            predicted.predictions["df"], np.broadcast_to(new_nu, (2, 3, count))
        )
        response_prediction = model.predict(
            params,
            data=new_data[["x", "z"]],
            kind="response",
            inplace=False,
            random_seed=223,
        )
        assert response_prediction.predictions[response_name].shape == (2, 3, count, 2)
        log_likelihood = model.compute_log_likelihood(params, data=new_data, inplace=False)
        expected_logp = np.array(
            [
                multivariate_t.logpdf(
                    row, loc=np.full(2, 0.7), shape=VECTOR_NORMAL_COV, df=nu_value
                )
                for row, nu_value in zip(new_data[["y1", "y2"]].to_numpy(), new_nu)
            ]
        )
        np.testing.assert_allclose(
            log_likelihood.log_likelihood[response_name],
            np.broadcast_to(expected_logp, (2, 3, count)),
        )


def test_scalar_auxiliary_coefficient_chain_stops_at_likelihood_parameter():
    data = vector_normal_data()
    prior = normal_prior(sigma=1)
    chained = bmb.Model(
        bmb.Formula(
            "join(y1, y2) ~ 0 + nl(nu*x)",
            "nu ~ 0 + nl(2 + exp(k*x))",
            "k ~ 0 + z + nl(j*x)",
            "j ~ 0 + z",
            nlpars=("k", "j"),
        ),
        data,
        family=vector_student_t_family(),
        priors={"j": {"z": prior}, "k": {"z": prior}},
        extra_namespace={"join": join_vector_response},
        center_predictors=False,
    )
    chained.build()
    j_coefficient = chained.parameter_graph.nodes["k"].nonlinear_coefficients["j"]
    assert j_coefficient.get_coefs_dim() is None
    assert chained.backend.model.named_vars_to_dims["j"] == ("__obs__",)
    assert chained.backend.model.named_vars_to_dims["k"] == ("__obs__",)
    chained_point = {"j_z": 0.2, "k_z": 0.1}
    expected_j = chained_point["j_z"] * data.z.to_numpy()
    expected_k = chained_point["k_z"] * data.z.to_numpy() + expected_j * data.x.to_numpy()
    expected_nu = 2 + np.exp(expected_k * data.x.to_numpy())
    expected_mu = np.broadcast_to(expected_nu[:, None] * data.x.to_numpy()[:, None], (len(data), 2))
    actual_nu, actual_mu = chained.backend.model.compile_fn(
        [chained.backend.model["nu"], chained.backend.model["mu"]],
        inputs=chained.backend.model.free_RVs,
    )(chained_point)
    np.testing.assert_allclose(actual_nu, expected_nu)
    np.testing.assert_allclose(actual_mu, expected_mu)


def test_marginal_coefficient_remains_shared_across_scalar_and_vector_consumers():
    family = vector_student_t_family()
    data = vector_normal_data()
    model = bmb.Model(
        bmb.Formula(
            "join(y1, y2) ~ 0 + nl(shared)",
            "nu ~ 0 + nl(shared)",
            nlpars=("shared",),
        ),
        data,
        family=family,
        priors={"shared": 0.7},
        extra_namespace={"join": join_vector_response},
    )
    model.build()
    coefficient = model.parameter_graph.nonlinear_coefficients["shared"]
    assert isinstance(coefficient, MarginalCoefficient)
    np.testing.assert_allclose(model.backend.model["mu"].eval(), 0.7)
    np.testing.assert_allclose(model.backend.model["nu"].eval(), 0.7)

    explicit = bmb.Model(
        bmb.Formula(
            "join(y1, y2) ~ 0 + nl(shared)",
            "nu ~ 0 + nl(shared)",
            "shared ~ 1",
            nlpars=("shared",),
        ),
        data,
        family=vector_student_t_family(),
        priors={"shared": bmb.Prior("Normal", mu=0.7, sigma=0.1)},
        extra_namespace={"join": join_vector_response},
    )
    explicit.build()
    explicit_coefficient = explicit.parameter_graph.nonlinear_coefficients["shared"]
    assert isinstance(explicit_coefficient, MarginalCoefficient)
    actual_mu, actual_nu = explicit.backend.model.compile_fn(
        [explicit.backend.model["mu"], explicit.backend.model["nu"]],
        inputs=explicit.backend.model.free_RVs,
    )({"shared": 0.5})
    np.testing.assert_allclose(actual_mu, 0.5)
    np.testing.assert_allclose(actual_nu, 0.5)


@pytest.mark.usefixtures("mock_pymc_sample")
@pytest.mark.parametrize("sparse_dot", [False, True])
def test_scalar_auxiliary_mixed_coefficient_group_prediction(monkeypatch, sparse_dot):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = vector_normal_data().assign(group=["g", "h"] * 3)
    model = bmb.Model(
        bmb.Formula(
            "join(y1, y2) ~ 0 + nl(0.7)",
            "nu ~ 0 + nl(2 + exp(k*x))",
            "k ~ 0 + z + nl(j*x) + (1 | group)",
            "j ~ 0 + x",
            nlpars=("k", "j"),
        ),
        data,
        family=vector_student_t_family(),
        extra_namespace={"join": join_vector_response},
        center_predictors=False,
    )
    idata = model.fit(draws=3, chains=1, random_seed=224)
    assert idata.posterior["k_1|group"].dims == ("chain", "draw", "group_dim")
    fixed_k = 0.2 * idata.posterior.k_z.values + 0.09 * idata.posterior.j_x.values
    for groups in (["g", "h"], ["g"], ["new", "new"]):
        new_data = pd.DataFrame({"x": 0.3, "z": 0.2, "group": groups})
        result = model.predict(idata, data=new_data, inplace=False, random_seed=225)
        nu = result.predictions.nu.values
        assert nu.shape == (1, 3, len(groups))
        if groups[0] == "new":
            np.testing.assert_allclose(nu[:, :, 0], nu[:, :, 1])
        else:
            effects = idata.posterior["k_1|group"].sel(group_dim=groups).values
            expected = 2 + np.exp((fixed_k[..., None] + effects) * 0.3)
            np.testing.assert_allclose(nu, expected)
        population = model.predict(
            idata, data=new_data, include_group_specific=False, inplace=False
        )
        expected = 2 + np.exp(fixed_k[..., None] * 0.3)
        np.testing.assert_allclose(population.predictions.nu, np.broadcast_to(expected, nu.shape))
