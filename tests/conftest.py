import pathlib

import pytest
import bambi as bmb
import numpy as np
import pandas as pd


from pymc.testing import mock_sample_setup_and_teardown

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
    data = pd.read_csv(pathlib.Path(__file__).parent / "data" / "diabetes.txt", sep="\t")
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
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
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
    df = df.pivot(index=["treat", "carry"], columns="rating", values="size").reset_index()
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
    data["am"] = pd.Categorical(data["am"], categories=[0, 1], ordered=True)
    model = bmb.Model("mpg ~ hp * drat * am", data)
    idata = model.fit(tune=500, draws=500, chains=2, random_seed=1234)
    return model, idata
