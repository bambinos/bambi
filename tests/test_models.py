from os.path import dirname, join

import pytest

import bambi as bmb
import numpy as np
import pandas as pd

from bambi.terms import GroupSpecificTerm

TUNE = 50
DRAWS = 50


# TODO: Test drop_na


@pytest.fixture(scope="module")
def crossed_data():
    """
    Group specific effects:
    10 subjects, 12 items, 5 sites
    Subjects crossed with items, nested in sites
    Items crossed with sites

    common effects:
    A continuous predictor, a numeric dummy, and a three-level category
    (levels a,b,c)

    Structure:
    Subjects nested in dummy (e.g., gender), crossed with threecats
    Items crossed with dummy, nested in threecats
    Sites partially crossed with dummy (4/5 see a single dummy, 1/5 sees both
    dummies)
    Sites crossed with threecats
    """

    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "crossed_random.csv"))
    data["subj"] = data["subj"].astype(str)
    data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    return data


class FitPredictParent:
    def fit_and_predict(self, model):
        idata = model.fit(tune=TUNE, draws=DRAWS)
        return idata


class TestGaussianRegression(FitPredictParent):
    def test_intercept_only_model(self, crossed_data):
        model = bmb.Model("Y ~ 1", crossed_data)
        self.fit_and_predict(model)

    def test_slope_only_model(self, crossed_data):
        model = bmb.Model("Y ~ 0 + continuous", crossed_data)
        self.fit_and_predict(model)

    def test_cell_means_parameterization(self, crossed_data):
        model = bmb.Model("Y ~ 0 + threecats", crossed_data)
        idata = self.fit_and_predict(model)
        assert list(idata.posterior["threecats_dim"]) == ["a", "b", "c"]

    def test_2_factors_saturated(self, crossed_data):
        model = bmb.Model("Y ~ threecats*fourcats", crossed_data)
        idata = self.fit_and_predict(model)
        assert list(idata.posterior.data_vars) == [
            "Intercept",
            "threecats",
            "fourcats",
            "threecats:fourcats",
            "Y_sigma",
        ]
        assert list(idata.posterior["threecats_dim"].values) == ["b", "c"]
        assert list(idata.posterior["fourcats_dim"].values) == ["b", "c", "d"]
        assert list(idata.posterior["threecats:fourcats_dim"].values) == [
            "b, b",
            "b, c",
            "b, d",
            "c, b",
            "c, c",
            "c, d",
        ]

    def test_2_factors_no_intercept(self, crossed_data):
        model = bmb.Model("Y ~ 0 + threecats*fourcats", crossed_data)
        idata = self.fit_and_predict(model)
        assert list(idata.posterior.data_vars) == [
            "threecats",
            "fourcats",
            "threecats:fourcats",
            "Y_sigma",
        ]
        assert list(idata.posterior["threecats_dim"].values) == ["a", "b", "c"]
        assert list(idata.posterior["fourcats_dim"].values) == ["b", "c", "d"]
        assert list(idata.posterior["threecats:fourcats_dim"].values) == [
            "b, b",
            "b, c",
            "b, d",
            "c, b",
            "c, c",
            "c, d",
        ]

    def test_2_factors_cell_means(self, crossed_data):
        model = bmb.Model("Y ~ 0 + threecats:fourcats", crossed_data)
        idata = self.fit_and_predict(model)
        assert list(idata.posterior.data_vars) == ["threecats:fourcats", "Y_sigma"]
        assert list(idata.posterior["threecats:fourcats_dim"].values) == [
            "a, a",
            "a, b",
            "a, c",
            "a, d",
            "b, a",
            "b, b",
            "b, c",
            "b, d",
            "c, a",
            "c, b",
            "c, c",
            "c, d",
        ]

    def test_cell_means_with_covariate(self, crossed_data):
        model = bmb.Model("Y ~ 0 + threecats + continuous", crossed_data)
        idata = self.fit_and_predict(model)
        assert list(idata.posterior.data_vars) == ["threecats", "continuous", "Y_sigma"]
        assert list(idata.posterior["threecats_dim"].values) == ["a", "b", "c"]

    def test_many_common_many_group_specific(self, crossed_data):
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

        model0 = bmb.Model("Y ~ " + " + ".join(terms0), crossed_data)
        self.fit_and_predict(model0)

        model1 = bmb.Model("Y ~ " + " + ".join(terms1), crossed_data)
        self.fit_and_predict(model1)

        # Check that the group specific effects design matrices have the same shape
        X0 = pd.concat(
            [pd.DataFrame(t.data) for t in model0.response_component.group_specific_terms.values()],
            axis=1,
        )
        X1 = pd.concat(
            [pd.DataFrame(t.data) for t in model1.response_component.group_specific_terms.values()],
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
        for term in model0.response_component.common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X0_list.append(tuple(term.data[:, level_idx]))
            else:
                X0_list.append(tuple(term.data))

        for term in model1.response_component.common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X1_list.append(tuple(term.data[:, level_idx]))
            else:
                X1_list.append(tuple(term.data))

        assert set(X0_list) == set(X1_list)

        # check that models have same priors for common effects
        priors0 = {
            x.name: x.prior.args
            for x in model0.response_component.terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args
            for x in model1.response_component.terms.values()
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
            for x in model0.response_component.group_specific_terms.values()
        }
        priors1 = {
            x.name: x.prior.args["sigma"].args
            for x in model1.response_component.group_specific_terms.values()
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

    def test_cell_means_with_many_group_specific_effects(self, crossed_data):
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
        model0 = bmb.Model("Y ~ " + " + ".join(terms0), crossed_data)
        self.fit_and_predict(model0)

        model1 = bmb.Model("Y ~ " + " + ".join(terms1), crossed_data)
        self.fit_and_predict(model1)

        # check that the group specific effects design matrices have the same shape
        X0 = pd.concat(
            [
                pd.DataFrame(t.data)
                if not isinstance(t.data, dict)
                else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
                for t in model0.response_component.group_specific_terms.values()
            ],
            axis=1,
        )
        X1 = pd.concat(
            [
                pd.DataFrame(t.data)
                if not isinstance(t.data, dict)
                else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
                for t in model0.response_component.group_specific_terms.values()
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
                for t in model0.response_component.common_terms.values()
                for lev in range(len(t.levels))
            ]
        )
        X1 = set(
            [
                tuple(t.data[:, lev])
                for t in model1.response_component.common_terms.values()
                for lev in range(len(t.levels))
            ]
        )
        assert X0 == X1

        # check that fit and add models have same priors for common effects
        priors0 = {
            x.name: x.prior.args
            for x in model0.response_component.terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args
            for x in model1.response_component.terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        assert set(priors0) == set(priors1)

        # check that fit and add models have same priors for group specific effects
        priors0 = {
            x.name: x.prior.args["sigma"].args
            for x in model0.response_component.terms.values()
            if isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args["sigma"].args
            for x in model1.response_component.terms.values()
            if isinstance(x, GroupSpecificTerm)
        }
        assert set(priors0) == set(priors1)

    def test_group_specific_categorical_interaction(self, crossed_data):
        model = bmb.Model("Y ~ continuous + (threecats:fourcats|site)", crossed_data)
        idata = self.fit_and_predict(model)

        assert list(idata.posterior.data_vars) == [
            "Intercept",
            "continuous",
            "Y_sigma",
            "1|site_sigma",
            "threecats:fourcats|site_sigma",
            "1|site",
            "threecats:fourcats|site",
        ]
        assert list(idata.posterior["threecats:fourcats|site"].coords) == [
            "chain",
            "draw",
            "site__factor_dim",
            "threecats:fourcats__expr_dim",
        ]
        assert list(idata.posterior["1|site"].coords) == ["chain", "draw", "site__factor_dim"]
        assert list(idata.posterior["1|site_sigma"].coords) == ["chain", "draw"]
        assert list(idata.posterior["threecats:fourcats|site_sigma"].coords) == [
            "chain",
            "draw",
            "threecats:fourcats__expr_dim",
        ]

        assert list(idata.posterior["threecats:fourcats__expr_dim"].values) == [
            "b, b",
            "b, c",
            "b, d",
            "c, b",
            "c, c",
            "c, d",
        ]
        assert list(idata.posterior["site__factor_dim"].values) == ["0", "1", "2", "3", "4"]


class TestLogisticRegression(FitPredictParent):

    def test_logistic_regression_empty_index(self, data_100):
        model = bmb.Model("b1 ~ n1", data_100, family="bernoulli")
        idata = self.fit_and_predict(model)
        # TODO: Test something in the outcome

    def test_logistic_regression_good_numeric(self, data_100):
        model = bmb.Model("b0 ~ n1", data_100, family="bernoulli")
        idata = self.fit_and_predict(model)
        # TODO: Test something in the outcome

    def test_logistic_regression_bad_numeric(self):
        error_msg = "Numeric response must be all 0 and 1 for 'bernoulli' family"
        rng = np.random.default_rng(1234)
        data = pd.DataFrame({"y": rng.choice([1, 2], 50), "x": rng.normal(size=50)})
        with pytest.raises(ValueError, match=error_msg):
            model = bmb.Model("y ~ x", data, family="bernoulli")
            idata = self.fit_and_predict(model)

    # FIXME
    def test_binomial_regression():
        data = pd.DataFrame(
            {
                "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
                "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
                "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
            }
        )

        model = bmb.Model("prop(y, n) ~ x", data, family="binomial")
        model.fit(draws=10, tune=10)

        # Using constant instead of variable in data frame
        model = bmb.Model("prop(y, 62) ~ x", data, family="binomial")
        model.fit(draws=10, tune=10)


class TestPoissonRegression(FitPredictParent):

    # TODO: create model fixture so it's used in several places

    def test_poisson_regression(self, crossed_data):
        # TODO: fixme
        # build model using fit and pymc
        crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
        model0 = bmb.Model("count ~ dummy + continuous + threecats", crossed_data, family="poisson")
        model0.fit(tune=0, draws=1)

        # build model using add
        model1 = bmb.Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
        model1.fit(tune=0, draws=1)

        # check that term names agree
        assert set(model0.response_component.terms) == set(model1.response_component.terms)

        # check that common effect design matrices are the same,
        # even if term names / level names / order of columns is different

        X0_list = []
        X1_list = []
        for term in model0.response_component.common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X0_list.append(tuple(term.data[:, level_idx]))
            else:
                X0_list.append(tuple(term.data))

        for term in model1.response_component.common_terms.values():
            if term.levels is not None:
                for level_idx in range(len(term.levels)):
                    X1_list.append(tuple(term.data[:, level_idx]))
            else:
                X1_list.append(tuple(term.data))

        assert set(X0_list) == set(X1_list)

        # check that models have same priors for common effects
        priors0 = {
            x.name: x.prior.args
            for x in model0.response_component.terms.values()
            if not isinstance(x, GroupSpecificTerm)
        }
        priors1 = {
            x.name: x.prior.args
            for x in model1.response_component.terms.values()
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


    def test_prior_predictive(crossed_data):
        crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
        # New default priors are too wide for this case... something to keep investigating
        model = bmb.Model(
            "count ~ threecats + continuous + dummy",
            crossed_data,
            family="poisson",
        )
        model.build()
        pps = model.prior_predictive(draws=500, random_seed=1234)

        keys = ["Intercept", "threecats", "continuous", "dummy"]
        shapes = [(1, 500), (1, 500, 2), (1, 500), (1, 500)]

        for key, shape in zip(keys, shapes):
            assert pps.prior[key].shape == shape

        assert pps.prior_predictive["count"].shape == (1, 500, 120)
        assert pps.observed_data["count"].shape == (120,)

        pps = model.prior_predictive(draws=500, var_names=["count"], random_seed=1234)
        assert pps.groups() == ["prior_predictive", "observed_data"]

        pps = model.prior_predictive(draws=500, var_names=["Intercept"], random_seed=1234)
        assert pps.groups() == ["prior", "observed_data"]

    # TODO: implement
    def test_posterior_predictive(crossed_data):
        crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
        model = bmb.Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
        fitted = model.fit(tune=0, draws=10, chains=2)
        pps = model.predict(fitted, kind="pps", inplace=False)

        assert pps.posterior_predictive["count"].shape == (2, 10, 120)

        pps = model.predict(fitted, kind="pps", inplace=True)

        assert pps is None
        assert fitted.posterior_predictive["count"].shape == (2, 10, 120)


class TestLaplaceRegression(FitPredictParent):

    def test_laplace_regression(self, data_100):
        model = bmb.Model("n1 ~ n2", data_100, family="laplace")
        idata = self.fit_and_predict(model)
        # TODO: test something?


class TestGammaRegression(FitPredictParent):

    # FIXME
    def test_gamma_regression(dm):
        # simulated data
        rng = np.random.default_rng(1234)
        a, b, N, shape_true = 0.5, 1.2, 100, 10  # alpha
        x = rng.uniform(-1, 1, N)
        y_true = np.exp(a + b * x)

        y = rng.gamma(shape_true, y_true / shape_true, N)
        data = pd.DataFrame({"x": x, "y": y})
        model = bmb.Model("y ~ x", data, family="gamma", link="log")
        model.fit(draws=10, tune=10)

        # Real data, categorical predictor.
        data = dm[["order", "ind_mg_dry"]]
        model = bmb.Model("ind_mg_dry ~ order", data, family="gamma", link="log")
        model.fit(draws=10, tune=10)


class TestBetaRegression(FitPredictParent):
    # FIXME
    def test_beta_regression():
        # FIXME
        data_dir = join(dirname(__file__), "data")
        data = pd.read_csv(join(data_dir, "gasoline.csv"))
        model = bmb.Model("yield ~  temp + batch", data, family="beta", categorical="batch")
        model.fit(draws=10, tune=10, target_accept=0.9)


class TestTRegression(FitPredictParent):
    # FIXME
    def test_t_regression(data_100):
        bmb.Model("n1 ~ n2", data_100, family="t").fit(draws=10, tune=10)


class TestVonMisesRegression(FitPredictParent):
    # FIXME
    def test_vonmises_regression():
        rng = np.random.default_rng(1234)
        data = pd.DataFrame({"y": rng.vonmises(0, 1, size=100), "x": rng.normal(size=100)})
        bmb.Model("y ~ x", data, family="vonmises").fit(draws=10, tune=10)


class TestAsymmetricLaplaceRegression(FitPredictParent):
    def test_quantile_regression():
        rng = np.random.default_rng(1234)
        x = rng.uniform(2, 10, 100)
        y = 2 * x + rng.normal(0, 0.6 * x**0.75)
        data = pd.DataFrame({"x": x, "y": y})
        bmb_model0 = bmb.Model("y ~ x", data, family="asymmetriclaplace", priors={"kappa": 9})
        idata0 = bmb_model0.fit()
        bmb_model0.predict(idata0)

        bmb_model1 = bmb.Model("y ~ x", data, family="asymmetriclaplace", priors={"kappa": 0.1})
        idata1 = bmb_model1.fit()
        bmb_model1.predict(idata1)

        assert np.all(
            idata0.posterior["y_mean"].mean(("chain", "draw"))
            > idata1.posterior["y_mean"].mean(("chain", "draw"))
        )

class TestCategoricalRegression(FitPredictParent):
    # FIXME
    def test_categorical_family(inhaler):
        model = bmb.Model("rating ~ period + carry + treat", inhaler, family="categorical")
        model.fit(draws=10, tune=10)


    def test_categorical_family_varying_intercept(inhaler):
        model = bmb.Model(
            "rating ~ period + carry + treat + (1|subject)", inhaler, family="categorical"
        )
        model.fit(draws=10, tune=10)


    def test_categorical_family_categorical_predictors(categorical_family_categorical_predictor):
        formula = "response ~ group + city"
        model = bmb.Model(formula, categorical_family_categorical_predictor, family="categorical")
        model.fit(draws=10, tune=10)

# things like splines, include mean, etc
class TestSpecificFeatures(FitPredictParent):

    def test_fit_include_mean(crossed_data):
        draws = 500
        model = bmb.Model("Y ~ continuous * threecats", crossed_data)
        idata = model.fit(tune=draws, draws=draws, include_mean=True)
        assert idata.posterior["Y_mean"].shape[1:] == (draws, 120)

        # Compare with the mean obtained with `model.predict()`
        mean = idata.posterior["Y_mean"].stack(sample=("chain", "draw")).values.mean(1)

        model.predict(idata)
        predicted_mean = idata.posterior["Y_mean"].stack(sample=("chain", "draw")).values.mean(1)

        assert np.array_equal(mean, predicted_mean)