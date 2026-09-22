import pytest

import numpy as np
import pandas as pd
import preliz as pz

from formulae.transforms import CyclicCubicSpline, NaturalCubicSpline
from xarray import DataTree

import bambi as bmb
from bambi.terms.smooth import SmoothTerm
from bambi.transformations import CCSpline, CRSpline

RANDOM_SEED = sum(map(ord, "Test smooths"))


@pytest.fixture
def smooth_data():
    x = np.linspace(0, 5, 40)
    eps = pz.Normal(mu=0, sigma=0.1).rvs(len(x), random_state=RANDOM_SEED)
    return pd.DataFrame({"x": x, "y": np.sin(6 * x) + eps})


@pytest.fixture
def grouped_smooth_data():
    x = np.concatenate([np.linspace(-i, i + 1, 30 + i) for i in range(3)])
    group = np.repeat(["a", "b", "c"], [30, 31, 32])
    eps = pz.Normal(mu=0, sigma=0.05).rvs(len(x), random_state=RANDOM_SEED)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.sin(3 * x) + eps,
            "group": pd.Categorical(group, categories=["c", "a", "b"], ordered=True),
        }
    ).sample(frac=1, random_state=RANDOM_SEED)


@pytest.fixture
def one_group_smooth_data(grouped_smooth_data):
    data = grouped_smooth_data.loc[grouped_smooth_data.group == "b"].copy()
    data["group"] = data.group.cat.remove_unused_categories()
    return data


@pytest.fixture
def prediction_data(grouped_smooth_data):
    return {
        "smooth": pd.DataFrame({"x": [-0.2, 0.1, 0.8, 1.2]}),
        "grouped": grouped_smooth_data.iloc[[10, 30, 70]],
    }


def posterior_draws(prior):
    return DataTree.from_dict({"posterior": prior["prior"].to_dataset()})


def direct_grouped_prediction(training, new_data, coefficients, center):
    expected = np.zeros((*coefficients.shape[:2], len(new_data)))
    for i, level in enumerate(training.group.cat.categories):
        spline = NaturalCubicSpline()
        spline(training.loc[training.group == level, "x"], df=6, center=center)
        rows = np.asarray(new_data.group == level)
        basis = spline.to_random(spline.eval(new_data.loc[rows, "x"]))
        expected[..., rows] = coefficients[..., i, :] @ basis.T
    return expected


def default_priors(center):
    priors = {
        "linear": bmb.Prior("Normal", mu=0, sigma=1),
        "curvature": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)),
    }
    if not center:
        priors["constant"] = bmb.Prior("Normal", mu=0, sigma=1)
    return priors


def assert_scaled_default_priors(model, term, response):
    response_std = response.std(ddof=0)
    linear_index = term.null_space_dimension - 1
    linear = term.prior["linear"]
    curvature = term.prior["curvature"]

    if term.by_levels is None:
        expected_linear_sigma = 2.5 * response_std / term.data[:, linear_index].std(ddof=0)
    else:
        expected_linear_sigma = [
            2.5
            * response_std
            / term.data[
                term.transform.by_indexes == i, i * term.basis_dimension + linear_index
            ].std(ddof=0)
            for i in range(len(term.by_levels))
        ]

    assert linear.args["mu"] == 0
    np.testing.assert_allclose(linear.args["sigma"], expected_linear_sigma)
    assert curvature.args["mu"] == 0
    assert curvature.args["sigma"].name == "HalfNormal"
    assert curvature.args["sigma"].args["sigma"] == response_std

    constant = term.prior.get("constant")
    if constant is not None:
        common_terms = tuple(model.parameters["mu"].common_terms.values())
        if common_terms:
            sigma = np.hstack([common_term.prior.args["sigma"] for common_term in common_terms])
            mean = np.hstack([common_term.data.mean(axis=0) for common_term in common_terms])
            expected_constant_sigma = np.sqrt((2.5 * response_std) ** 2 + np.dot(sigma**2, mean**2))
        else:
            expected_constant_sigma = 2.5 * response_std

        assert constant.args["mu"] == response.mean()
        assert constant.args["sigma"] == expected_constant_sigma


class TestCr:
    @staticmethod
    def name(center):
        return f"cr(x, df=6, center={center})"

    @staticmethod
    def formula(name, center):
        return f"y ~ {name}" if center else f"y ~ 0 + {name}"

    @pytest.mark.parametrize("center", [False, True])
    def test_default_priors(self, smooth_data, center):
        name = self.name(center)
        model = bmb.Model(self.formula(name, center), smooth_data)
        term = model.parameters["mu"].terms[name]

        assert isinstance(term, SmoothTerm)
        model.build()
        assert_scaled_default_priors(model, term, smooth_data.y)

    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
    def test_custom_priors(self, smooth_data, center, distribution):
        name = self.name(center)
        kwargs = {"nu": 4} if distribution == "StudentT" else {}
        priors = {
            "linear": bmb.Prior(distribution, mu=2, sigma=4, **kwargs),
            "curvature": bmb.Prior(
                distribution,
                mu=0,
                sigma=bmb.Prior("HalfNormal", sigma=3),
                **kwargs,
            ),
        }
        if not center:
            priors["constant"] = bmb.Prior(distribution, mu=1, sigma=3, **kwargs)

        model = bmb.Model(self.formula(name, center), smooth_data, priors={name: priors})
        term = model.parameters["mu"].terms[name]

        assert term.prior["linear"].args["mu"] == 2
        assert term.prior["curvature"].args["sigma"].args["sigma"] == 3
        assert all(not prior.auto_scale for prior in term.prior.values())
        model.build()

    @pytest.mark.parametrize("center", [False, True])
    def test_custom_prior_mixed(self, smooth_data, center):
        name = self.name(center)
        priors = default_priors(center)
        priors["linear"] = bmb.Prior("StudentT", nu=4, mu=0, sigma=1)

        model = bmb.Model(self.formula(name, center), smooth_data, priors={name: priors})
        model.build()
        prior = model.prior_predictive(draws=2, random_seed=RANDOM_SEED)["prior"]

        assert prior[name].shape == (1, 2, 6)
        assert prior[f"{name}_linear"].shape == (1, 2)

    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("new_data", [False, True])
    def test_predict(self, smooth_data, prediction_data, center, new_data):
        name = self.name(center)
        model = bmb.Model(self.formula(name, center), smooth_data)
        model.set_alias({name: "curve"})
        model.build()
        prior = model.prior_predictive(draws=3, random_seed=RANDOM_SEED)
        draws = posterior_draws(prior)
        data = prediction_data["smooth"] if new_data else None

        result = model.predict(draws, data=data, inplace=False)
        basis = (
            model.parameters["mu"]
            .terms[name]
            .term.eval_new_data(smooth_data if data is None else data)
        )
        expected = result["posterior"]["curve"].values @ basis.T
        if center:
            expected += result["posterior"]["Intercept"].values[..., None]
        group = "posterior" if data is None else "predictions"
        np.testing.assert_allclose(result[group]["mu"].values, expected, atol=1e-10)

    @pytest.mark.parametrize("center", [False, True])
    def test_fixed_curvature(self, smooth_data, center):
        name = self.name(center)
        priors = default_priors(center)
        priors["curvature"] = 0

        with pytest.warns(UserWarning, match="curvature.*usually modeled"):
            model = bmb.Model(self.formula(name, center), smooth_data, priors={name: priors})
        model.build()
        prior = model.prior_predictive(draws=2, random_seed=RANDOM_SEED)["prior"]

        curvature_start = 1 if center else 2
        np.testing.assert_array_equal(prior[name].values[..., curvature_start:], 0)

    def test_incompatible_arr_rejected(self, smooth_data):
        name = self.name(center=True)
        priors = {
            "linear": bmb.Prior("Normal", mu=0, sigma=1),
            "curvature": bmb.Prior("Normal", mu=[0, 1], sigma=1),
        }
        model = bmb.Model(self.formula(name, center=True), smooth_data, priors={name: priors})

        with pytest.raises(ValueError):
            model.build()

    def test_explicit_knots(self, smooth_data):
        spline = CRSpline()
        spline(smooth_data.x, knots=[1, 2, 3, 4], lower_bound=0, upper_bound=5)

        np.testing.assert_array_equal(spline._knots, [0, 1, 2, 3, 4, 5])

    def test_alias(self, smooth_data):
        name = self.name(center=True)
        model = bmb.Model(self.formula(name, center=True), smooth_data)
        model.set_alias({name: "curve"})
        model.build()

        assert model.backend.model.named_vars_to_dims["curve"] == ("curve_dim",)

    @pytest.mark.parametrize("center", [False, True])
    def test_random_basis(self, smooth_data, center):
        original, adapted = NaturalCubicSpline(), CRSpline()
        original(smooth_data.x, df=6, center=center)

        np.testing.assert_allclose(
            adapted(smooth_data.x, df=6, center=center),
            original.to_random(),
        )

    @pytest.mark.parametrize(
        "center, missing",
        [(True, "linear"), (True, "curvature"), (False, "constant"), (False, "linear")],
    )
    def test_incomplete_priors_rejected(self, smooth_data, center, missing):
        name = f"cr(x, df=6, center={center})"
        priors = default_priors(center)
        del priors[missing]

        with pytest.raises(ValueError, match="must specify exactly"):
            bmb.Model(f"y ~ 0 + {name}", smooth_data, priors={name: priors})

    @pytest.mark.parametrize("block", ["linear", "curvature"])
    def test_normal_smooth_requires_mu_and_sigma(self, smooth_data, block):
        name = "cr(x, df=6)"
        priors = {
            "linear": bmb.Prior("StudentT", nu=4, mu=0, sigma=1),
            "curvature": bmb.Prior("StudentT", nu=4, mu=0, sigma=1),
        }
        priors[block] = bmb.Prior("Normal", mu=0, tau=1)

        with pytest.raises(ValueError, match="require 'mu' and 'sigma'"):
            bmb.Model(f"y ~ {name}", smooth_data, priors={name: priors})

    @pytest.mark.parametrize("value", [None, "Normal", {"mu": 0}, ["a", "b"]])
    def test_non_numeric_smooth_constant_rejected(self, smooth_data, value):
        name = "cr(x, df=6)"
        priors = {"linear": value, "curvature": bmb.Prior("Normal", mu=0, sigma=1)}

        with pytest.raises(ValueError, match="must be a Prior or a numeric constant"):
            bmb.Model(f"y ~ {name}", smooth_data, priors={name: priors})

    def test_gaussian_curvature_random_mean_rejected(self, smooth_data):
        name = "cr(x, df=6)"
        priors = {
            "linear": bmb.Prior("Normal", mu=0, sigma=1),
            "curvature": bmb.Prior("Normal", mu=bmb.Prior("Normal", mu=0, sigma=1), sigma=1),
        }

        with pytest.raises(
            ValueError, match="mean of the curvature prior must be a numeric constant"
        ):
            bmb.Model(f"y ~ {name}", smooth_data, priors={name: priors})

    @pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
    @pytest.mark.parametrize("block", ["constant", "linear"])
    @pytest.mark.parametrize("argument", ["mu", "sigma"])
    def test_unpenalized_hyperpriors_rejected(self, smooth_data, distribution, block, argument):
        name = "cr(x, df=6, center=False)"
        priors = default_priors(center=False)
        kwargs = {"nu": 4} if distribution == "StudentT" else {}
        priors[block] = bmb.Prior(distribution, mu=0, sigma=1, **kwargs)
        priors[block].update(**{argument: bmb.Prior("HalfNormal", sigma=1)})

        with pytest.raises(ValueError, match=f"'{block}'.*random variable arguments"):
            bmb.Model(f"y ~ 0 + {name}", smooth_data, priors={name: priors})

    def test_group_specific_smooth_rejected(self, grouped_smooth_data):
        with pytest.raises(NotImplementedError, match="Group-specific smooths"):
            bmb.Model("y ~ (cr(x, df=6) | group)", grouped_smooth_data)

    def test_grouped_explicit_knots(self, grouped_smooth_data):
        spline = CRSpline()
        spline(
            grouped_smooth_data.x,
            by=grouped_smooth_data.group,
            knots=[0.2, 0.4, 0.6, 0.8],
            lower_bound=-2,
            upper_bound=3,
        )

        for group_spline in spline.group_splines:
            np.testing.assert_array_equal(group_spline._knots, [-2, 0.2, 0.4, 0.6, 0.8, 3])



    @staticmethod
    def formula(name, center):
        return f"y ~ group + {name}" if center else f"y ~ 0 + group + {name}"

    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("shared", [False, True])
    def test_default_priors(self, grouped_smooth_data, center, shared):
        name = self.name(center, shared)
        model = bmb.Model(self.formula(name, center), grouped_smooth_data)
        term = model.parameters["mu"].terms[name]

        assert term.shared is shared
        model.build()
        assert_scaled_default_priors(model, term, grouped_smooth_data.y)
        scale = model.backend.model[f"{name}_sigma"]
        assert scale.type.shape == (() if shared else (3,))

    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("shared", [False, True])
    @pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
    def test_custom_priors(self, grouped_smooth_data, center, shared, distribution):
        name = self.name(center, shared)
        kwargs = {"nu": 4} if distribution == "StudentT" else {}
        curvature_size = 5 if center else 4
        priors = {
            "linear": bmb.Prior(distribution, mu=[2, 3, 4], sigma=4, **kwargs),
            "curvature": bmb.Prior(
                distribution,
                mu=np.arange(curvature_size)[None, :],
                sigma=bmb.Prior("HalfNormal", sigma=3),
                **kwargs,
            ),
        }
        if not center:
            priors["constant"] = bmb.Prior(distribution, mu=[1], sigma=3, **kwargs)

        model = bmb.Model(self.formula(name, center), grouped_smooth_data, priors={name: priors})
        term = model.parameters["mu"].terms[name]

        np.testing.assert_array_equal(term.prior["linear"].args["mu"], [2, 3, 4])
        assert all(not prior.auto_scale for prior in term.prior.values())
        model.build()

    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("component", ["constant", "linear", "curvature"])
    def test_custom_prior_mixed(self, grouped_smooth_data, center, component):
        if center and component == "constant":
            pytest.skip("Centered smooths do not have a constant component.")

        name = self.name(center, shared=False)
        priors = default_priors(center)
        priors[component] = bmb.Prior("StudentT", nu=4, mu=0, sigma=1)

        model = bmb.Model(self.formula(name, center), grouped_smooth_data, priors={name: priors})
        model.build()
        prior = model.prior_predictive(draws=2, random_seed=RANDOM_SEED)["prior"]

        expected_size = 5 if center else 4
        assert prior[f"{name}_{component}"].shape[-1] == (
            expected_size if component == "curvature" else 3
        )

    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("shared", [False, True])
    @pytest.mark.parametrize("new_data", [False, True])
    def test_predict(self, grouped_smooth_data, prediction_data, center, shared, new_data):
        name = self.name(center, shared)
        model = bmb.Model(f"y ~ 0 + {name}", grouped_smooth_data)
        model.set_alias({name: "curve"})
        model.build()
        prior = model.prior_predictive(draws=3, random_seed=RANDOM_SEED)
        draws = posterior_draws(prior)
        data = prediction_data["grouped"] if new_data else None
        prediction = model.predict(draws, data=data, inplace=False)
        expected_data = grouped_smooth_data if data is None else data
        expected = direct_grouped_prediction(
            grouped_smooth_data,
            expected_data,
            prior["prior"]["curve"].values,
            center,
        )

        group = "posterior" if data is None else "predictions"
        np.testing.assert_allclose(prediction[group]["mu"].values, expected, atol=1e-10)

    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("shared", [False, True])
    def test_fixed_curvature(self, grouped_smooth_data, center, shared):
        name = self.name(center, shared)
        priors = default_priors(center)
        priors["curvature"] = 0

        with pytest.warns(UserWarning, match="curvature.*usually modeled"):
            model = bmb.Model(f"y ~ 0 + {name}", grouped_smooth_data, priors={name: priors})
        model.build()
        prior = model.prior_predictive(draws=2, random_seed=RANDOM_SEED)["prior"]

        curvature_start = 1 if center else 2
        np.testing.assert_array_equal(prior[name].values[..., curvature_start:], 0)

    @pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
    def test_incompatible_arr_rejected(self, grouped_smooth_data, distribution):
        name = self.name(center=True, shared=False)
        kwargs = {"nu": 4} if distribution == "StudentT" else {}
        priors = {
            "linear": bmb.Prior("Normal", mu=0, sigma=1),
            "curvature": bmb.Prior(distribution, mu=[0, 1], sigma=1, **kwargs),
        }
        model = bmb.Model(f"y ~ {name}", grouped_smooth_data, priors={name: priors})

        with pytest.raises(ValueError):
            model.build()
            model.prior_predictive(draws=1, random_seed=RANDOM_SEED)

    @pytest.mark.parametrize("ordered", [False, True])
    def test_coord_matches_order(self, grouped_smooth_data, ordered):
        data = grouped_smooth_data.copy()
        data["group"] = data.group.cat.set_categories(["b", "c", "a"], ordered=ordered)
        name = "cr(x, df=6, by=group)"
        model = bmb.Model(f"y ~ 0 + group + {name}", data)
        model.build()

        expected = ("b", "c", "a") if ordered else ("a", "b", "c")
        assert model.backend.model.coords["group_dim"] == expected

    def test_explicit_knots(self, grouped_smooth_data):
        spline = CRSpline()
        spline(
            grouped_smooth_data.x,
            by=grouped_smooth_data.group,
            knots=[0.2, 0.4, 0.6, 0.8],
            lower_bound=-2,
            upper_bound=3,
        )

        for group_spline in spline.group_splines:
            np.testing.assert_array_equal(group_spline._knots, [-2, 0.2, 0.4, 0.6, 0.8, 3])

    @pytest.mark.parametrize("shared", [False, True])
    @pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
    def test_hyperpriors(self, grouped_smooth_data, shared, distribution):
        name = self.name(center=True, shared=shared)
        curvature = bmb.Prior(
            distribution,
            mu=0,
            sigma=bmb.Prior("HalfNormal", sigma=bmb.Prior("HalfNormal", sigma=1)),
        )
        if distribution == "StudentT":
            curvature.update(nu=4)
        priors = {
            "linear": bmb.Prior("Normal", mu=0, sigma=1),
            "curvature": curvature,
        }
        model = bmb.Model(
            self.formula(name, center=True), grouped_smooth_data, priors={name: priors}
        )
        model.set_alias({name: "curve"})
        model.build()

        scale_name = "curve_sigma" if distribution == "Normal" else "curve_curvature_sigma"
        expected_shape = () if shared else (3,)
        assert model.backend.model[scale_name].type.shape == expected_shape

    @pytest.mark.parametrize("shared", [False, True])
    def test_hyperpriors_one_level(self, one_group_smooth_data, shared):
        name = self.name(center=True, shared=shared)
        priors = {
            "linear": bmb.Prior("Normal", mu=2, sigma=1),
            "curvature": bmb.Prior(
                "StudentT",
                nu=4,
                mu=0,
                sigma=bmb.Prior("HalfNormal", sigma=1),
            ),
        }
        model = bmb.Model(
            self.formula(name, center=True), one_group_smooth_data, priors={name: priors}
        )
        model.build()
        prior = model.prior_predictive(draws=2, random_seed=RANDOM_SEED)["prior"]

        assert prior[name].shape == (1, 2, 1, 6)

    def test_alias(self, grouped_smooth_data):
        name = self.name(center=True, shared=False)
        model = bmb.Model(self.formula(name, center=True), grouped_smooth_data)
        model.set_alias({name: "curve"})
        model.build()

        assert model.backend.model.named_vars_to_dims["curve"] == ("group_dim", "curve_dim")


class TestCc:
    @staticmethod
    def name(center):
        return f"cc(x, period=5, df=6, center={center})"

    @pytest.mark.parametrize(
        "center, prior_keys", [(False, ["constant", "curvature"]), (True, ["curvature"])]
    )
    def test_model_builds_with_default_priors(self, smooth_data, center, prior_keys):
        name = self.name(center)
        formula = f"y ~ 0 + {name}" if not center else f"y ~ {name}"
        model = bmb.Model(formula, smooth_data)

        term = model.parameters["mu"].terms[name]
        assert list(term.prior) == prior_keys
        model.build()

    @pytest.mark.parametrize("center", [False, True])
    def test_random_basis(self, smooth_data, center):
        original, adapted = CyclicCubicSpline(), CCSpline()
        original(smooth_data.x, period=5, df=6, center=center)

        np.testing.assert_allclose(
            adapted(smooth_data.x, period=5, df=6, center=center),
            original.to_random(),
        )

    def test_grouped_model_builds(self, grouped_smooth_data):
        name = "cc(x, period=5, df=6, by=group, shared=True)"
        model = bmb.Model(f"y ~ group + {name}", grouped_smooth_data)

        assert model.parameters["mu"].terms[name].shared
        model.build()


@pytest.mark.parametrize(
    "center, missing",
    [(True, "linear"), (True, "curvature"), (False, "constant"), (False, "linear")],
)
def test_incomplete_priors_rejected(smooth_data, center, missing):
    name = f"cr(x, df=6, center={center})"
    priors = default_priors(center)
    del priors[missing]

    with pytest.raises(ValueError, match="must specify exactly"):
        bmb.Model(f"y ~ 0 + {name}", smooth_data, priors={name: priors})


@pytest.mark.parametrize("block", ["linear", "curvature"])
def test_normal_smooth_requires_mu_and_sigma(smooth_data, block):
    name = "cr(x, df=6)"
    priors = {
        "linear": bmb.Prior("StudentT", nu=4, mu=0, sigma=1),
        "curvature": bmb.Prior("StudentT", nu=4, mu=0, sigma=1),
    }
    priors[block] = bmb.Prior("Normal", mu=0, tau=1)

    with pytest.raises(ValueError, match="require 'mu' and 'sigma'"):
        bmb.Model(f"y ~ {name}", smooth_data, priors={name: priors})


@pytest.mark.parametrize("value", [None, "Normal", {"mu": 0}, ["a", "b"]])
def test_non_numeric_smooth_constant_rejected(smooth_data, value):
    name = "cr(x, df=6)"
    priors = {"linear": value, "curvature": bmb.Prior("Normal", mu=0, sigma=1)}

    with pytest.raises(ValueError, match="must be a Prior or a numeric constant"):
        bmb.Model(f"y ~ {name}", smooth_data, priors={name: priors})


def test_gaussian_curvature_random_mean_rejected(smooth_data):
    name = "cr(x, df=6)"
    priors = {
        "linear": bmb.Prior("Normal", mu=0, sigma=1),
        "curvature": bmb.Prior("Normal", mu=bmb.Prior("Normal", mu=0, sigma=1), sigma=1),
    }

    with pytest.raises(ValueError, match="mean of the curvature prior must be a numeric constant"):
        bmb.Model(f"y ~ {name}", smooth_data, priors={name: priors})


@pytest.mark.parametrize("distribution", ["Normal", "StudentT"])
@pytest.mark.parametrize("block", ["constant", "linear"])
@pytest.mark.parametrize("argument", ["mu", "sigma"])
def test_unpenalized_hyperpriors_rejected(smooth_data, distribution, block, argument):
    name = "cr(x, df=6, center=False)"
    priors = default_priors(center=False)
    kwargs = {"nu": 4} if distribution == "StudentT" else {}
    priors[block] = bmb.Prior(distribution, mu=0, sigma=1, **kwargs)
    priors[block].update(**{argument: bmb.Prior("HalfNormal", sigma=1)})

    with pytest.raises(ValueError, match=f"'{block}'.*random variable arguments"):
        bmb.Model(f"y ~ 0 + {name}", smooth_data, priors={name: priors})


def test_group_specific_smooth_rejected(grouped_smooth_data):
    with pytest.raises(NotImplementedError, match="Group-specific smooths"):
        bmb.Model("y ~ (cr(x, df=6) | group)", grouped_smooth_data)
