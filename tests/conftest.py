import pathlib

import numpy as np
import pandas as pd
import pytest
from pymc.testing import mock_sample_setup_and_teardown

import bambi as bmb

# Register as a pytest fixture
# `mock_pymc_sample` is a fixture that allows to replace `pm.sample` with
# `pm.sample_prior_predictive` during test execution.
mock_pymc_sample = pytest.fixture(scope="function")(mock_sample_setup_and_teardown)


@pytest.fixture(scope="module")
def data_random_n100():
    size = 100
    rng = np.random.default_rng(121195)

    data = pd.DataFrame(
        {
            "continuous1": rng.normal(size=size),
            "continuous2": rng.normal(size=size),
            "continuous3": rng.normal(size=size),
            "binary_num": rng.binomial(n=1, p=0.5, size=size),
            "binary_cat": rng.choice(["a", "b"], size=size),
            "count1": rng.poisson(lam=2, size=size),
            "count2": 1 + rng.poisson(lam=3, size=size),
            "categorical1": rng.choice(list("MNOP"), size=size),
            "categorical2": rng.choice(list("FGHIJK"), size=size),
        }
    )
    return data


@pytest.fixture(scope="module")
def data_anes():
    return bmb.load_data("ANES")


@pytest.fixture(scope="module")
def data_diabetes():
    data = pd.read_csv(
        pathlib.Path(__file__).parent / "data" / "diabetes.txt", sep="\t"
    )
    data["age_grp"] = 0
    data.loc[data["AGE"] > 40, "age_grp"] = 1
    data.loc[data["AGE"] > 60, "age_grp"] = 2
    return data


@pytest.fixture(scope="module")
def data_crossed():
    """
    Group specific effects:
    10 subjects, 12 items, 5 sites
    Subjects crossed with items, nested in sites
    Items crossed with sites

    Common effects:
    A continuous predictor, a numeric dummy, and a three-level category
    (levels a,b,c)

    Structure:
    Subjects nested in dummy (e.g., gender), crossed with threecats
    Items crossed with dummy, nested in threecats
    Sites partially crossed with dummy (4/5 see a single dummy, 1/5 sees both
    dummies)
    Sites crossed with threecats
    """
    data = pd.read_csv(pathlib.Path(__file__).parent / "data" / "crossed_random.csv")
    data["subj"] = data["subj"].astype(str)
    data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    return data


@pytest.fixture(scope="module")
def data_beetle():
    return pd.DataFrame(
        {
            "x": np.array(
                [1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]
            ),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )


@pytest.fixture(scope="module")
def data_gasoline():
    return pd.read_csv(pathlib.Path(__file__).parent / "data" / "gasoline.csv")


@pytest.fixture(scope="module")
def data_inhaler():
    data = pd.read_csv(pathlib.Path(__file__).parent / "data" / "inhaler.csv")
    data["rating"] = pd.Categorical(data["rating"], categories=[1, 2, 3, 4])
    return data


@pytest.fixture(scope="module")
def data_kidney():
    data = bmb.load_data("kidney")
    data["status"] = np.where(data["censored"] == 0, "none", "right")
    return data


@pytest.fixture(scope="module")
def data_multinomial(data_inhaler):
    df = data_inhaler.groupby(["treat", "carry", "rating"], as_index=False).size()
    df = df.pivot(
        index=["treat", "carry"], columns="rating", values="size"
    ).reset_index()
    df.columns = ["treat", "carry", "y1", "y2", "y3", "y4"]
    return df


@pytest.fixture(scope="module")
def data_sleepstudy():
    return bmb.load_data("sleepstudy")


# NOTE: scope is session so we fit it once per session.
@pytest.fixture(scope="session")
def mtcars_fixture():
    """Model with common level effects only"""
    data = bmb.load_data("mtcars")
    data["hp"] = data["hp"].astype(pd.Float32Dtype())
    data["drat"] = data["drat"].astype(pd.Float32Dtype())
    data["am"] = pd.Categorical(data["am"], categories=[0, 1], ordered=True)
    model = bmb.Model("mpg ~ hp * drat * am", data)
    idata = model.fit(tune=500, draws=500, chains=2, random_seed=1234)
    return model, idata


@pytest.fixture(scope="session")
def sleep_study():
    """Model with common and group specific effects"""
    data = bmb.load_data("sleepstudy")
    model = bmb.Model("Reaction ~ 1 + Days + (Days | Subject)", data)
    idata = model.fit(tune=500, draws=200, chains=2, random_seed=1234)
    return model, idata


@pytest.fixture(scope="session")
def food_choice():
    """Model a categorical response using the 'categorical' family"""
    length = [
        1.3, 1.32, 1.32, 1.4, 1.42, 1.42, 1.47, 1.47, 1.5, 1.52,
        1.63, 1.65, 1.65, 1.65, 1.65, 1.68, 1.7, 1.73, 1.78, 1.78,
        1.8, 1.85, 1.93, 1.93, 1.98, 2.03, 2.03, 2.31, 2.36, 2.46,
        3.25, 3.28, 3.33, 3.56, 3.58, 3.66, 3.68, 3.71, 3.89,
        1.24, 1.3, 1.45, 1.45, 1.55, 1.6, 1.6, 1.65, 1.78, 1.78,
        1.8, 1.88, 2.16, 2.26, 2.31, 2.36, 2.39, 2.41, 2.44, 2.56,
        2.67, 2.72, 2.79, 2.84,
    ]
    choice = [
        "I", "F", "F", "F", "I", "F", "I", "F", "I", "I",
        "I", "O", "O", "I", "F", "F", "I", "O", "F", "O",
        "F", "F", "I", "F", "I", "F", "F", "F", "F", "F",
        "O", "O", "F", "F", "F", "F", "O", "F", "F",
        "I", "I", "I", "O", "I", "I", "I", "F", "I", "O",
        "I", "I", "F", "F", "F", "F", "F", "F", "F", "O",
        "F", "I", "F", "F",
    ]
    sex = ["Male"] * 32 + ["Female"] * 31
    data = pd.DataFrame({"choice": choice, "length": length, "sex": sex})
    data["choice"] = pd.Categorical(
        data["choice"].map({"I": "Invertebrates", "F": "Fish", "O": "Other"}),
        ["Other", "Invertebrates", "Fish"],
        ordered=True,
    )
    model = bmb.Model("choice ~ length + sex", data, family="categorical")
    idata = model.fit(tune=500, draws=200, chains=2, random_seed=1234)
    return model, idata


@pytest.fixture(scope="session")
def formulae_transform():
    """A model with a 'formulae' stateful transformation (polynomial) on a term."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=100)
    x2 = rng.normal(size=100)
    y = 2 + 3 * x1 + 1.5 * x1**2 + 2 * x2 + rng.normal(scale=1, size=100)
    data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    model = bmb.Model("y ~ poly(x1, 2) + x2", data)
    idata = model.fit(tune=500, draws=200, chains=2, random_seed=1234)
    return model, idata


@pytest.fixture(scope="session")
def nonformulae_transform():
    """A model with a non-formulae transformation on a term."""
    rng = np.random.default_rng(0)
    x1 = rng.uniform(1, 50, 50)
    noise = rng.normal(0, 1, 50)
    y = 3 * np.log(x1) + noise
    data = pd.DataFrame({"x1": x1, "y": y})
    model = bmb.Model("y ~ np.log(x1)", data)
    idata = model.fit(tune=500, draws=200, chains=2, random_seed=1234)
    return model, idata


@pytest.fixture(scope="session")
def distributional_fixture():
    """Gamma model with distributional component for testing target parameter."""
    rng = np.random.default_rng(121195)
    N = 200
    a, b = 0.5, 1.1
    x = rng.uniform(-1.5, 1.5, N)
    shape = np.exp(0.3 + x * 0.5 + rng.normal(scale=0.1, size=N))
    y = rng.gamma(shape, np.exp(a + b * x) / shape, N)
    data = pd.DataFrame({"x": x, "y": y})
    formula = bmb.Formula("y ~ x", "alpha ~ x")
    model = bmb.Model(formula, data, family="gamma")
    idata = model.fit(tune=100, draws=100, random_seed=1234)
    return model, idata


@pytest.fixture(scope="session")
def integer_data_fixture():
    """Model with integer-typed predictor for testing integer dtype paths."""
    rng = np.random.default_rng(42)
    n = 100
    x_int = rng.integers(1, 10, size=n)
    x_float = rng.normal(size=n)
    y = 2.0 + 0.5 * x_int + 1.5 * x_float + rng.normal(scale=0.5, size=n)
    data = pd.DataFrame({"x_int": x_int, "x_float": x_float, "y": y})
    model = bmb.Model("y ~ x_int + x_float", data)
    idata = model.fit(tune=500, draws=200, chains=2, random_seed=1234)
    return model, idata
