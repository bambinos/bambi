from os.path import dirname, join

import pytest

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm

from bambi.terms import GroupSpecificTerm
from scipy.special import expit

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
def beetle_data():
    return pd.DataFrame(
        {
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )


@pytest.fixture(scope="module")
def gasoline_data():
    data_dir = join(dirname(__file__), "data")
    return pd.read_csv(join(data_dir, "gasoline.csv"))


@pytest.fixture(scope="module")
def inhaler():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "inhaler.csv"))
    data["rating"] = pd.Categorical(data["rating"], categories=[1, 2, 3, 4])
    return data


class FitPredictParent:
    def fit(self, model, **kwargs):
        return model.fit(tune=TUNE, draws=DRAWS, **kwargs)

    def predict_oos(self, model, idata, data=None):
        # Reuse the original data
        if data is None:
            data = model.data.head()
        return model.predict(idata, kind="pps", data=data, inplace=False)


class TestGaussian(FitPredictParent):
    def test_intercept_only_model(self, crossed_data):
        model = bmb.Model("Y ~ 1", crossed_data)
        idata = self.fit(model)
        self.predict_oos(model, idata)

    def test_slope_only_model(self, crossed_data):
        model = bmb.Model("Y ~ 0 + continuous", crossed_data)
        idata = self.fit(model)
        self.predict_oos(model, idata)

    def test_cell_means_parameterization(self, crossed_data):
        model = bmb.Model("Y ~ 0 + threecats", crossed_data)
        idata = self.fit(model)
        assert list(idata.posterior["threecats_dim"]) == ["a", "b", "c"]
        self.predict_oos(model, idata)

    def test_2_factors_saturated(self, crossed_data):
        model = bmb.Model("Y ~ threecats*fourcats", crossed_data)
        idata = self.fit(model)
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
        self.predict_oos(model, idata)

    def test_2_factors_no_intercept(self, crossed_data):
        model = bmb.Model("Y ~ 0 + threecats*fourcats", crossed_data)
        idata = self.fit(model)
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
        self.predict_oos(model, idata)

    def test_2_factors_cell_means(self, crossed_data):
        model = bmb.Model("Y ~ 0 + threecats:fourcats", crossed_data)
        idata = self.fit(model)
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
        self.predict_oos(model, idata)

    def test_cell_means_with_covariate(self, crossed_data):
        model = bmb.Model("Y ~ 0 + threecats + continuous", crossed_data)
        idata = self.fit(model)
        assert list(idata.posterior.data_vars) == ["threecats", "continuous", "Y_sigma"]
        assert list(idata.posterior["threecats_dim"].values) == ["a", "b", "c"]
        self.predict_oos(model, idata)

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
        idata0 = self.fit(model0)
        self.predict_oos(model0, idata0)

        model1 = bmb.Model("Y ~ " + " + ".join(terms1), crossed_data)
        idata1 = self.fit(model1)
        self.predict_oos(model1, idata1)

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
        idata0 = self.fit(model0)
        self.predict_oos(model0, idata0)

        model1 = bmb.Model("Y ~ " + " + ".join(terms1), crossed_data)
        idata1 = self.fit(model1)
        self.predict_oos(model1, idata1)

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
        idata = self.fit(model)
        self.predict_oos(model, idata)

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


class TestBernoulli(FitPredictParent):
    def assert_posterior_predictive_range(self, model, idata):
        y_name = model.response_component.response_term.name
        y_posterior_predictive = idata.posterior_predictive[y_name].to_numpy()
        assert set(np.unique(y_posterior_predictive)) == {0, 1}

    def assert_mean_range(self, model, idata):
        y_mean_name = model.response_component.response_term.name + "_mean"
        y_mean_posterior = idata.posterior[y_mean_name].to_numpy()
        assert ((0 < y_mean_posterior) & (y_mean_posterior < 1)).all()

    def test_bernoulli_empty_index(self, data_n100):
        model = bmb.Model("b1 ~ 1 + y1", data_n100, family="bernoulli")
        idata = self.fit(model)
        model.predict(idata, kind="pps")
        self.assert_mean_range(model, idata)
        self.assert_posterior_predictive_range(model, idata)

        # out of sample prediction
        idata = self.predict_oos(model, idata)
        self.assert_mean_range(model, idata)
        self.assert_posterior_predictive_range(model, idata)

    def test_bernoulli_good_numeric(self, data_n100):
        model = bmb.Model("b1 ~ y1", data_n100, family="bernoulli")
        idata = self.fit(model)
        model.predict(idata, kind="pps")
        self.assert_mean_range(model, idata)
        self.assert_posterior_predictive_range(model, idata)

    def test_bernoulli_bad_numeric(self, data_n100):
        error_msg = "Numeric response must be all 0 and 1 for 'bernoulli' family"
        with pytest.raises(ValueError, match=error_msg):
            model = bmb.Model("y1 ~ y2", data_n100, family="bernoulli")
            self.fit(model)

    def test_categorical_group_specific(self, data_n100):
        # see https://github.com/bambinos/bambi/issues/447

        formula = "b1 ~ 0 + cat2 + y2 + (0 + cat2|cat5) + (0 + y2| cat4 + cat5)"
        model = bmb.Model(formula, data_n100, family="bernoulli")
        idata = self.fit(model, chains=2)
        idata = self.predict_oos(model, idata)

        self.assert_mean_range(model, idata)
        self.assert_posterior_predictive_range(model, idata)


class TestBinomial(FitPredictParent):
    def assert_mean_range(self, model, idata):
        y_mean_name = model.response_component.response_term.name + "_mean"
        y_mean_posterior = idata.posterior[y_mean_name].to_numpy()
        assert ((0 < y_mean_posterior) & (y_mean_posterior < 1)).all()

    def test_binomial_regression(self, beetle_data):
        model = bmb.Model("prop(y, n) ~ x", beetle_data, family="binomial")
        idata = self.fit(model)
        model.predict(idata, kind="pps")
        self.assert_mean_range(model, idata)
        y_reshaped = beetle_data["n"].to_numpy()[None, None, :]

        assert (idata.posterior_predictive["prop(y, n)"].to_numpy() <= y_reshaped).all()
        assert (0 <= idata.posterior_predictive["prop(y, n)"].to_numpy()).all()

        y_reshaped = beetle_data["n"].to_numpy()[None, None, :3]
        idata = self.predict_oos(model, idata, data=model.data.head(3))
        self.assert_mean_range(model, idata)
        assert (idata.posterior_predictive["prop(y, n)"].to_numpy() <= y_reshaped).all()

    def test_binomial_regression_constant(self, beetle_data):
        # Uses a constant instead of variable in data frame
        model = bmb.Model("prop(y, 62) ~ x", beetle_data, family="binomial")
        idata = self.fit(model)
        model.predict(idata, kind="pps")
        self.assert_mean_range(model, idata)
        assert (idata.posterior_predictive["prop(y, 62)"].to_numpy() <= 62).all()
        assert (0 <= idata.posterior_predictive["prop(y, 62)"].to_numpy()).all()

        # Out of sample prediction
        idata = self.predict_oos(model, idata)
        self.assert_mean_range(model, idata)
        assert (idata.posterior_predictive["prop(y, 62)"].to_numpy() <= 62).all()


class TestPoisson(FitPredictParent):
    def assert_mean_range(self, model, idata):
        y_mean_name = model.response_component.response_term.name + "_mean"
        y_mean_posterior = idata.posterior[y_mean_name].to_numpy()
        assert (y_mean_posterior > 0).all()

    def test_poisson_regression(self, crossed_data):
        crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
        model0 = bmb.Model("count ~ dummy + continuous + threecats", crossed_data, family="poisson")
        idata0 = self.fit(model0)
        idata0 = self.predict_oos(model0, idata0)
        self.assert_mean_range(model0, idata0)

        # build model using add
        model1 = bmb.Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
        idata1 = self.fit(model1)
        idata1 = self.predict_oos(model1, idata1)
        self.assert_mean_range(model1, idata1)

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

        # Now test prior predictive
        pps = model1.prior_predictive(draws=200, random_seed=1234)

        keys = ["Intercept", "threecats", "continuous", "dummy"]
        shapes = [(1, 200), (1, 200, 2), (1, 200), (1, 200)]

        for key, shape in zip(keys, shapes):
            assert pps.prior[key].shape == shape

        assert pps.prior_predictive["count"].shape == (1, 200, 120)
        assert pps.observed_data["count"].shape == (120,)

        pps = model1.prior_predictive(draws=200, var_names=["count"], random_seed=1234)
        assert pps.groups() == ["prior_predictive", "observed_data"]

        pps = model1.prior_predictive(draws=200, var_names=["Intercept"], random_seed=1234)
        assert pps.groups() == ["prior", "observed_data"]

        # Now test posterior predictive
        # Fit again to make sure we fix the number of chainS
        idata = model1.fit(tune=50, draws=50, chains=2)
        pps = model1.predict(idata, kind="pps", inplace=False)
        assert pps.posterior_predictive["count"].shape == (2, 50, 120)

        pps = model1.predict(idata, kind="pps", inplace=True)
        assert pps is None
        assert idata.posterior_predictive["count"].shape == (2, 50, 120)


class TestNegativeBinomial(FitPredictParent):
    # To Do: Could be modified to follow the same format than the others
    def test_predict_negativebinomial(self, data_n100):

        model = bmb.Model("n1 ~ y1", data_n100, family="negativebinomial")
        idata = self.fit(model)

        model.predict(idata, kind="mean")
        model.predict(idata, kind="pps")

        assert (0 < idata.posterior["y_mean"]).all()
        assert (np.equal(np.mod(idata.posterior_predictive["y"].values, 1), 0)).all()

        model.predict(idata, kind="mean", data=data_n100.iloc[:20, :])
        model.predict(idata, kind="pps", data=data_n100.iloc[:20, :])

        assert (0 < idata.posterior["y_mean"]).all()
        assert (np.equal(np.mod(idata.posterior_predictive["y"].values, 1), 0)).all()


class TestLaplace(FitPredictParent):
    def test_laplace_regression(self, data_n100):
        model = bmb.Model("y1 ~ y2", data_n100, family="laplace")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "y2", "y1_b"}
        assert (idata.posterior["y1_b"] > 0).all().item()

        idata = self.predict_oos(model, idata)
        assert "y1_mean" in idata.posterior


class TestGamma(FitPredictParent):
    def test_gamma_regression(self, data_n100):
        # Construct a positive variable
        data_n100["o"] = np.exp(data_n100["y1"])
        model = bmb.Model("o ~ y2 + y3 + n1 + cat4", data_n100, family="gamma", link="log")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "y2", "y3", "n1", "cat4", "o_alpha"}
        idata = self.predict_oos(model, idata)
        assert (idata.posterior_predictive["o"] > 0).all().item()

    def test_gamma_regression_categoric(self, data_n100):
        data_n100["o"] = np.exp(data_n100["y1"])
        model = bmb.Model("o ~ 0 + cat2:cat4", data_n100, family="gamma", link="log")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"cat2:cat4", "o_alpha"}
        idata = self.predict_oos(model, idata)
        assert (idata.posterior_predictive["o"] > 0).all().item()


class TestBeta(FitPredictParent):
    def test_beta_regression(self, gasoline_data):
        model = bmb.Model(
            "yield ~  temp + batch", gasoline_data, family="beta", categorical="batch"
        )
        idata = self.fit(model, target_accept=0.9, random_seed=1234)

        # To Do: Could be adjusted but this is what we had before
        model.predict(idata, kind="mean")
        model.predict(idata, kind="pps")

        assert (0 < idata.posterior["yield_mean"]).all() & (idata.posterior["yield_mean"] < 1).all()
        assert (0 < idata.posterior_predictive["yield"]).all() & (
            idata.posterior_predictive["yield"] < 1
        ).all()

        model.predict(idata, kind="mean", data=gasoline_data.iloc[:20, :])
        model.predict(idata, kind="pps", data=gasoline_data.iloc[:20, :])

        assert (0 < idata.posterior["yield_mean"]).all() & (idata.posterior["yield_mean"] < 1).all()
        assert (0 < idata.posterior_predictive["yield"]).all() & (
            idata.posterior_predictive["yield"] < 1
        ).all()


class TestStudentT(FitPredictParent):
    def test_t_regression(self, data_n100):
        model = bmb.Model("y1 ~ y2", data_n100, family="t")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "y2", "y1_nu", "y1_sigma"}
        self.predict_oos(model, idata)


class TestVonMises(FitPredictParent):
    def test_vonmises_regression(self):
        rng = np.random.default_rng(1234)
        data = pd.DataFrame({"y": rng.vonmises(0, 1, size=100), "x": rng.normal(size=100)})
        model = bmb.Model("y ~ x", data, family="vonmises")
        idata = self.fit(model)
        assert set(idata.posterior.data_vars) == {"Intercept", "x", "y_kappa"}
        idata = self.predict_oos(model, idata)
        assert (idata.posterior_predictive["y"].min() >= -np.pi).item() and (
            idata.posterior_predictive["y"].max() <= np.pi
        ).item()


class TestAsymmetricLaplace(FitPredictParent):
    # This test doesn't follow the previous pattern but it works...
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


class TestCategorical(FitPredictParent):
    # FIXME
    def test_categorical_family(self, inhaler):
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
            pm.ZeroInflatedNegativeBinomial.dist(mu=5, alpha=30, psi=0.7),
            draws=500,
            random_seed=1234,
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
        # To have both numeric and categoric predictors
        inhaler["carry"] = pd.Categorical(inhaler["carry"])
        model = bmb.Model("rating ~ period + carry + treat", inhaler, family=family, link=link)
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
            "censored(time, status) ~ 1 + sex + age",
            data,
            family="weibull",
            link="log",
            priors=priors,
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
            "censored(time, status) ~ 0 + sex + age",
            data,
            family="weibull",
            link="log",
            priors=priors,
        )
        idata = model.fit(tune=100, draws=100, random_seed=121195)
        model.predict(idata, kind="pps")
        model.predict(idata, data=data, kind="pps")

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
        assert np.all(
            idata.posterior_predictive["c(y1, y2, y3, y4)"].values.sum(-1).var((0, 1)) == 0
        )


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
