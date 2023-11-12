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

# TODO: Add out of sample prediction
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


# TODO: Logistic -> Bernoulli and Binomial

class TestBernoulli(FitPredictParent):

    # TODO: Make this other types of models
    # FIXME: This is the first place where we have out of sample predictions, we should test it
    #        in more cases, possibly, add a method
    def test_predict_bernoulli(data_bernoulli):
        data = data_bernoulli
        model = bmb.Model("y ~ x1*x2", data, family="bernoulli")
        idata = model.fit(tune=100, draws=100, target_accept=0.9)

        # In sample prediction
        model.predict(idata, kind="mean")
        model.predict(idata, kind="pps")

        assert (0 < idata.posterior["y_mean"]).all() & (idata.posterior["y_mean"] < 1).all()
        assert (idata.posterior_predictive["y"].isin([0, 1])).all()

        # Out of sample prediction
        model.predict(idata, kind="mean", data=data.iloc[:20, :])
        model.predict(idata, kind="pps", data=data.iloc[:20, :])

        assert (0 < idata.posterior["y_mean"]).all() & (idata.posterior["y_mean"] < 1).all()
        assert (idata.posterior_predictive["y"].isin([0, 1])).all()


class TestPoisson(FitPredictParent):

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
    # TODO: Test out of sample predictions


class TestNegativeBinomial(FitPredictParent):
    # FIXME
    def test_predict_negativebinomial(data_count):
        data = data_count

        model = bmb.Model("y ~ x", data, family="negativebinomial")
        idata = model.fit(tune=100, draws=100)

        model.predict(idata, kind="mean")
        model.predict(idata, kind="pps")

        assert (0 < idata.posterior["y_mean"]).all()
        assert (np.equal(np.mod(idata.posterior_predictive["y"].values, 1), 0)).all()

        model.predict(idata, kind="mean", data=data.iloc[:20, :])
        model.predict(idata, kind="pps", data=data.iloc[:20, :])

        assert (0 < idata.posterior["y_mean"]).all()
        assert (np.equal(np.mod(idata.posterior_predictive["y"].values, 1), 0)).all()


class TestLaplace(FitPredictParent):

    def test_laplace_regression(self, data_100):
        model = bmb.Model("n1 ~ n2", data_100, family="laplace")
        idata = self.fit_and_predict(model)
        # TODO: test something?


class TestGamma(FitPredictParent):

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

    def test_predict_gamma(data_gamma):
        data = data_gamma

        model = bmb.Model("y ~ x", data, family="gamma", link="log")
        idata = model.fit(tune=100, draws=100)

        model.predict(idata, kind="mean")
        model.predict(idata, kind="pps")

        assert (0 < idata.posterior["y_mean"]).all()
        assert (0 < idata.posterior_predictive["y"]).all()

        model.predict(idata, kind="mean", data=data.iloc[:20, :])
        model.predict(idata, kind="pps", data=data.iloc[:20, :])

        assert (0 < idata.posterior["y_mean"]).all()
        assert (0 < idata.posterior_predictive["y"]).all()



class TestBeta(FitPredictParent):
    # FIXME
    def test_beta_regression():
        # FIXME
        data_dir = join(dirname(__file__), "data")
        data = pd.read_csv(join(data_dir, "gasoline.csv"))
        model = bmb.Model("yield ~  temp + batch", data, family="beta", categorical="batch")
        model.fit(draws=10, tune=10, target_accept=0.9)



    def test_predict_beta(data_beta):
        data = data_beta
        data["batch"] = pd.Categorical(data["batch"], [10, 1, 2, 3, 4, 5, 6, 7, 8, 9], ordered=True)
        model = bmb.Model("yield ~ temp + batch", data, family="beta")
        idata = model.fit(tune=100, draws=100, target_accept=0.90)

        model.predict(idata, kind="mean")
        model.predict(idata, kind="pps")

        assert (0 < idata.posterior["yield_mean"]).all() & (idata.posterior["yield_mean"] < 1).all()
        assert (0 < idata.posterior_predictive["yield"]).all() & (
            idata.posterior_predictive["yield"] < 1
        ).all()

        model.predict(idata, kind="mean", data=data.iloc[:20, :])
        model.predict(idata, kind="pps", data=data.iloc[:20, :])

        assert (0 < idata.posterior["yield_mean"]).all() & (idata.posterior["yield_mean"] < 1).all()
        assert (0 < idata.posterior_predictive["yield"]).all() & (
            idata.posterior_predictive["yield"] < 1
        ).all()


class TestStudentT(FitPredictParent):
    # FIXME
    def test_t_regression(data_100):
        bmb.Model("n1 ~ n2", data_100, family="t").fit(draws=10, tune=10)
    # TODO: test predictions


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

    # FIXME
    def test_posterior_predictive_categorical(inhaler):
        model = bmb.Model("rating ~ period", data=inhaler, family="categorical")
        idata = model.fit(tune=100, draws=100)
        model.predict(idata, kind="pps")
        pps = idata.posterior_predictive["rating"].to_numpy()

        assert pps.shape[-1] == inhaler.shape[0]
        assert (np.unique(pps) == [0, 1, 2, 3]).all()

    # FIXME
    def test_predict_categorical_group_specific():
        # see https://github.com/bambinos/bambi/issues/447
        rng = np.random.default_rng(1234)
        size = 100

        data = pd.DataFrame(
            {
                "y": rng.choice([0, 1], size=size),
                "x1": rng.choice(list("abcd"), size=size),
                "x2": rng.choice(list("XY"), size=size),
                "x3": rng.normal(size=size),
            }
        )

        model = bmb.Model("y ~ x1 + (0 + x2|x1) + (0 + x3|x1 + x2)", data, family="bernoulli")

        idata = model.fit(tune=100, draws=100, chains=2)

        model.predict(idata, data=data)

        assert idata.posterior.y_mean.values.shape == (2, 100, 100)
        assert (idata.posterior.y_mean.values > 0).all() and (idata.posterior.y_mean.values < 1).all()


class TestZIPoisson(FitPredictParent):
    # FIXME: Use fixture data, drop pm.draw
    def test_zero_inflated_poisson(self):
        rng = np.random.default_rng(121195)

        # Basic intercept-only model
        x = np.concatenate([np.zeros(250), rng.poisson(lam=3, size=750)])
        df = pd.DataFrame({"response": x})

        model = bmb.Model("response ~ 1", df, family="zero_inflated_poisson")
        idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
        model.predict(idata, kind="pps")

        # Distributional model
        x = np.sort(rng.uniform(0.2, 3, size=1000))

        b0, b1 = 0.2, 0.9
        a0, a1 = 2.5, -0.7
        mu = np.exp(b0 + b1 * x)
        psi = expit(a0 + a1 * x)

        y = pm.draw(pm.ZeroInflatedPoisson.dist(mu=mu, psi=psi))
        df = pd.DataFrame({"y": y, "x": x})

        formula = bmb.Formula("y ~ x", "psi ~ x")
        model = bmb.Model(formula, df, family="zero_inflated_poisson")
        idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
        model.predict(idata, kind="pps")

class TestZIBinomial(FitPredictParent):
    # FIXME: Use fixture data, drop pm.draw
    def test_zero_inflated_binomial(self):
        rng = np.random.default_rng(121195)

        # Basic intercept-only model
        y = pm.draw(pm.ZeroInflatedBinomial.dist(p=0.5, n=30, psi=0.7), draws=500, random_seed=1234)
        df = pd.DataFrame({"y": y})
        model = bmb.Model("p(y, 30) ~ 1", df, family="zero_inflated_binomial")
        idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
        model.predict(idata, kind="pps")

        # Distributional model
        x = np.sort(rng.uniform(0.2, 3, size=500))
        b0, b1 = -0.5, 0.5
        a0, a1 = 2, -0.7
        p = expit(b0 + b1 * x)
        psi = expit(a0 + a1 * x)

        y = pm.draw(pm.ZeroInflatedBinomial.dist(p=p, psi=psi, n=30))
        df = pd.DataFrame({"y": y, "x": x})

        formula = bmb.Formula("prop(y, 30) ~ x", "psi ~ x")
        model = bmb.Model(formula, df, family="zero_inflated_binomial")
        idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
        model.predict(idata, kind="pps")


class TestZINegativeBinomial(FitPredictParent):
    # FIXME: Use fixture data, drop pm.draw
    def test_zero_inflated_negativebinomial():
        rng = np.random.default_rng(121195)

        # Basic intercept-only model
        y = pm.draw(
            pm.ZeroInflatedNegativeBinomial.dist(mu=5, alpha=30, psi=0.7), draws=500, random_seed=1234
        )
        df = pd.DataFrame({"y": y})
        priors = {"alpha": bmb.Prior("HalfNormal", sigma=20)}
        model = bmb.Model("y ~ 1", df, family="zero_inflated_negativebinomial", priors=priors)
        idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
        model.predict(idata, kind="pps")

        # Distributional model
        x = np.sort(rng.uniform(0.2, 3, size=500))
        b0, b1 = 0.5, 0.35
        a0, a1 = 2, -0.7
        mu = np.exp(b0 + b1 * x)
        psi = expit(a0 + a1 * x)

        y = pm.draw(pm.ZeroInflatedNegativeBinomial.dist(mu=mu, alpha=30, psi=psi))
        df = pd.DataFrame({"y": y, "x": x})

        priors = {"alpha": bmb.Prior("HalfNormal", sigma=20)}
        formula = bmb.Formula("y ~ x", "psi ~ x")
        model = bmb.Model(formula, df, family="zero_inflated_negativebinomial", priors=priors)
        idata = model.fit(chains=2, tune=200, draws=200, random_seed=121195)
        model.predict(idata, kind="pps")



class TestHurdle(FitPredictParent):
    # FIXME: Drop usage of pm.draw, use fixture data, parametrize test
    def test_hurlde_families():
        df = pd.DataFrame({"y": pm.draw(pm.HurdlePoisson.dist(0.5, mu=3.5), 1000)})
        model = bmb.Model("y ~ 1", df, family="hurdle_poisson")
        idata = model.fit()
        model.predict(idata, kind="pps")

        df = pd.DataFrame({"y": pm.draw(pm.HurdleNegativeBinomial.dist(0.6, 5, 3), 1000)})
        model = bmb.Model("y ~ 1", df, family="hurdle_negativebinomial")
        idata = model.fit()
        model.predict(idata, kind="pps")

        df = pd.DataFrame({"y": pm.draw(pm.HurdleGamma.dist(0.8, alpha=10, beta=1), 1000)})
        model = bmb.Model("y ~ 1", df, family="hurdle_gamma")
        idata = model.fit()
        model.predict(idata, kind="pps")

        df = pd.DataFrame({"y": pm.draw(pm.HurdleLogNormal.dist(0.7, mu=0, sigma=0.2), 1000)})
        model = bmb.Model("y ~ 1", df, family="hurdle_lognormal")
        idata = model.fit()
        model.predict(idata, kind="pps")
        

class TestOrdinal(FitPredictParent):
    # FIXME
    @pytest.mark.parametrize(
        "family, link",
        [
            ("cumulative", "logit"),
            ("cumulative", "probit"),
            ("cumulative", "cloglog"),
            ("sratio", "logit"),
            ("sratio", "probit"),
            ("sratio", "cloglog"),
        ],
    )
    def test_ordinal_families(inhaler, family, link):
        data = inhaler.copy()
        data["carry"] = pd.Categorical(data["carry"])  # To have both numeric and categoric predictors
        model = bmb.Model("rating ~ period + carry + treat", data, family=family, link=link)
        idata = model.fit(tune=100, draws=100)
        model.predict(idata, kind="pps")
        assert np.allclose(idata.posterior["rating_mean"].sum("rating_dim").to_numpy(), 1)
        assert np.all(np.unique(idata.posterior_predictive["rating"]) == np.array([0, 1, 2, 3]))

    def test_cumulative_family_priors(inhaler):
        priors = {
            "threshold": bmb.Prior(
                "Normal",
                mu=[-0.5, 0, 0.5],
                sigma=1.5,
                transform=pm.distributions.transforms.ordered,
            )
        }
        model = bmb.Model(
            "rating ~ 0 + period + carry + treat", inhaler, family="cumulative", priors=priors
        )
        model.fit(tune=100, draws=100)



class TestCensoredResponses(FitPredictParent):
    def test_censored_response():
        data = bmb.load_data("kidney")
        data["status"] = np.where(data["censored"] == 0, "none", "right")

        # Model 1, with intercept
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
            "sex": bmb.Prior("Normal", mu=0, sigma=2),
            "age": bmb.Prior("Normal", mu=0, sigma=1),
            "alpha": bmb.Prior("Gamma", alpha=3, beta=5),
        }
        model = bmb.Model(
            "censored(time, status) ~ 1 + sex + age", data, family="weibull", link="log", priors=priors
        )
        idata = model.fit(tune=100, draws=100, random_seed=121195)
        model.predict(idata, kind="pps")
        model.predict(idata, data=data, kind="pps")

        # Model 2, without intercept
        priors = {
            "sex": bmb.Prior("Normal", mu=0, sigma=2),
            "age": bmb.Prior("Normal", mu=0, sigma=1),
            "alpha": bmb.Prior("Gamma", alpha=3, beta=5),
        }
        model = bmb.Model(
            "censored(time, status) ~ 0 + sex + age", data, family="weibull", link="log", priors=priors
        )
        idata = model.fit(tune=100, draws=100, random_seed=121195)
        model.predict(idata, kind="pps")
        model.predict(idata, data=data, kind="pps")

        # Model 3, with group-specific effects
        priors = {
            "alpha": bmb.Prior("Gamma", alpha=3, beta=5),
            "sex": bmb.Prior("Normal", mu=0, sigma=1),
            "age": bmb.Prior("Normal", mu=0, sigma=1),
            "1|patient": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("InverseGamma", alpha=5, beta=10)),
        }
        model = bmb.Model(
            "censored(time, status) ~ 1 + sex + age + (1|patient)",
            data,
            family="weibull",
            link="log",
            priors=priors,
        )
        idata = model.fit(tune=100, draws=100, random_seed=121195)
        model.predict(idata, kind="pps")
        model.predict(idata, data=data, kind="pps")


class TestTruncatedResponse(FitPredictParent):
    # FIXME
    def test_truncated_response():
        rng = np.random.default_rng(12345)
        slope, intercept, sigma, N = 1, 0, 2, 200
        x = rng.uniform(-10, 10, N)
        y = rng.normal(loc=slope * x + intercept, scale=sigma)
        bounds = [-5, 5]
        keep = (y >= bounds[0]) & (y <= bounds[1])
        xt = x[keep]
        yt = y[keep]

        df = pd.DataFrame({"x": xt, "y": yt})
        priors = {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
            "x": bmb.Prior("Normal", mu=0, sigma=1),
            "sigma": bmb.Prior("HalfNormal", sigma=1),
        }
        model = bmb.Model("truncated(y, -5, 5) ~ x", df, priors=priors)
        idata = model.fit(tune=100, draws=100, random_seed=1234)
        model.predict(idata, kind="pps")


class TestMultinomial(FitPredictParent):
    # FIXME
    def test_predict_multinomial(inhaler):
        df = inhaler.groupby(["treat", "carry", "rating"], as_index=False).size()
        df = df.pivot(index=["treat", "carry"], columns="rating", values="size").reset_index()
        df.columns = ["treat", "carry", "y1", "y2", "y3", "y4"]

        # Intercept only
        model = bmb.Model("c(y1, y2, y3, y4) ~ 1", df, family="multinomial")
        idata = model.fit(tune=100, draws=100)

        model.predict(idata)
        model.predict(idata, data=df.iloc[:3, :])

        # Numerical predictors
        model = bmb.Model("c(y1, y2, y3, y4) ~ treat + carry", df, family="multinomial")
        idata = model.fit(tune=100, draws=100)

        model.predict(idata)
        model.predict(idata, data=df.iloc[:3, :])

        # Categorical predictors
        df["treat"] = df["treat"].replace({-0.5: "A", 0.5: "B"})
        df["carry"] = df["carry"].replace({-1: "a", 0: "b", 1: "c"})

        model = bmb.Model("c(y1, y2, y3, y4) ~ treat + carry", df, family="multinomial")
        idata = model.fit(tune=100, draws=100)

        model.predict(idata)
        model.predict(idata, data=df.iloc[:3, :])

        data = pd.DataFrame(
            {
                "state": ["A", "B", "C"],
                "y1": [35298, 1885, 5775],
                "y2": [167328, 20731, 21564],
                "y3": [212682, 37716, 20222],
                "y4": [37966, 5196, 3277],
            }
        )

        # Contains group-specific effect
        model = bmb.Model(
            "c(y1, y2, y3, y4) ~ 1 + (1 | state)", data, family="multinomial", noncentered=False
        )
        idata = model.fit(tune=100, draws=100, random_seed=0)

        model.predict(idata)
        model.predict(idata, kind="pps")


    def test_posterior_predictive_multinomial(inhaler):
        df = inhaler.groupby(["treat", "carry", "rating"], as_index=False).size()
        df = df.pivot(index=["treat", "carry"], columns="rating", values="size").reset_index()
        df.columns = ["treat", "carry", "y1", "y2", "y3", "y4"]

        # Intercept only
        model = bmb.Model("c(y1, y2, y3, y4) ~ 1", df, family="multinomial")
        idata = model.fit(tune=100, draws=100)

        # The sum across the columns of the response is the same for all the chain and draws.
        model.predict(idata, kind="pps")
        assert np.all(idata.posterior_predictive["c(y1, y2, y3, y4)"].values.sum(-1).var((0, 1)) == 0)


# FIXME
def test_posterior_predictive_dirichlet_multinomial(inhaler):
    df = inhaler.groupby(["treat", "rating"], as_index=False).size()
    df = df.pivot(index=["treat"], columns="rating", values="size").reset_index()
    df.columns = ["treat", "y1", "y2", "y3", "y4"]

    # Intercept only
    model = bmb.Model("c(y1, y2, y3, y4) ~ 1", df, family="dirichlet_multinomial")
    idata = model.fit(tune=100, draws=100)

    # The sum across the columns of the response is the same for all the chain and draws.
    model.predict(idata, kind="pps")
    assert np.all(idata.posterior_predictive["c(y1, y2, y3, y4)"].values.sum(-1).var((0, 1)) == 0)

    # With predictor only
    model = bmb.Model("c(y1, y2, y3, y4) ~ 0 + treat", df, family="dirichlet_multinomial")
    idata = model.fit(tune=100, draws=100)

    # The sum across the columns of the response is the same for all the chain and draws.
    model.predict(idata, kind="pps")
    assert np.all(idata.posterior_predictive["c(y1, y2, y3, y4)"].values.sum(-1).var((0, 1)) == 0)


# FIXME
def test_posterior_predictive_beta_binomial():
    data = pd.DataFrame(
        {
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )

    model = bmb.Model("prop(y, n) ~ x", data, family="beta_binomial")
    idata = model.fit(draws=100, tune=100)
    model.predict(idata, kind="pps")

    n = data["n"].to_numpy()
    assert np.all(idata.posterior_predictive["prop(y, n)"].values <= n[np.newaxis, np.newaxis, :])




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