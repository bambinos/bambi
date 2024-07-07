import pathlib
import re

import bambi as bmb
import numpy as np
import pandas as pd

import pytest


DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"


@pytest.fixture(scope="module")
def data_1d_single_group():
    rng = np.random.default_rng(seed=121195)
    size = 100
    b, sigma = 5, 1.5
    x = np.linspace(0, 10, size)
    y = b * np.sin(x / 1.75) + rng.normal(scale=sigma, size=size)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def data_1d_multiple_groups():
    return pd.read_csv(DATA_DIR / "gam_data.csv")


@pytest.fixture(scope="module")
def data_2d_single_group():
    return pd.read_csv(DATA_DIR / "hsgp_2d_single_group.csv")


@pytest.fixture(scope="module")
def data_2d_multiple_groups():
    return pd.read_csv(DATA_DIR / "hsgp_2d_multiple_groups.csv")


def test_minimal_1d_fits(data_1d_single_group):
    model = bmb.Model("y ~ 0 + hsgp(x, c=1.5, m=10)", data_1d_single_group)
    idata = model.fit(tune=500, draws=500, chains=2, random_seed=1234)


def test_required_params(data_1d_single_group):
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'm'"):
        bmb.Model("y ~ 0 + hsgp(x, c=1.5)", data_1d_single_group)

    with pytest.raises(ValueError, match="Provide one of 'c' or 'L'"):
        bmb.Model("y ~ 0 + hsgp(x, m=10)", data_1d_single_group)


def test_conflicting_params(data_1d_single_group):
    with pytest.raises(ValueError, match="Provide one of 'c' or 'L'"):
        bmb.Model("y ~ 0 + hsgp(x, m=10, c=1.5, L=10)", data_1d_single_group)


def test_m_bad_shape(data_1d_multiple_groups, data_2d_multiple_groups):
    match = "'m' must be scalar or a sequence with length equal to the number of variables"
    with pytest.raises(ValueError, match=match):
        m = [12, 5]  # There's one variable, not two
        bmb.Model("y ~ 0 + hsgp(x2, c=2, m=m)", data_1d_multiple_groups)

    with pytest.raises(ValueError, match=match):
        m = [[12, 5], [10, 6], [11, 7]]  # 'm' can't vary by group
        bmb.Model("y ~ 0 + hsgp(x2, c=2, m=m, by=fac)", data_1d_multiple_groups)

    with pytest.raises(ValueError, match=match):
        m = [10, 10, 10]  # There are two variables, not three
        bmb.Model("outcome ~ 0 + hsgp(x, y, c=2, m=m)", data_2d_multiple_groups)

    with pytest.raises(ValueError, match=match):
        m = [[10, 10], [10, 10], [10, 10]]  # 'm' can't vary by group
        bmb.Model("outcome ~ 0 + hsgp(x, y, c=2, m=m, by=group)", data_2d_multiple_groups)


def test_c_bad_shape(data_1d_multiple_groups, data_2d_multiple_groups):
    match = re.escape("1D sequences must be of shape (variables_n, )")
    with pytest.raises(ValueError, match=match):
        c = [2, 1.5, 1.4]  # wrong because it's of shape (groups_n, )
        bmb.Model("y ~ 0 + hsgp(x2, by=fac, c=c, m=10)", data_1d_multiple_groups)

    match = re.escape("2D sequences must be of shape (groups_n, variables_n)")
    with pytest.raises(ValueError, match=match):
        c = [[2], [1.5]]  # wrong because it's of shape (groups_n - 1, variables_n)
        bmb.Model("y ~ 0 + hsgp(x2, by=fac, c=c, m=10)", data_1d_multiple_groups)

    match = re.escape("1D sequences must be of shape (variables_n, )")
    with pytest.raises(ValueError, match=match):
        c = [2]  # wrong because it's of shape (variables_n - 1,)
        bmb.Model("outcome ~ 0 + hsgp(x, y, c=c, m=10)", data_2d_multiple_groups)

    match = re.escape("2D sequences must be of shape (groups_n, variables_n)")
    with pytest.raises(ValueError, match=match):
        c = [[2, 1.5], [1.8, 1.6]]  # wrong because it's of shape (variables_n - 1, groups_n - 1)
        bmb.Model("outcome ~ 0 + hsgp(x, y, by=group, c=c, m=10)", data_2d_multiple_groups)


def test_L_bad_shape(data_1d_multiple_groups, data_2d_multiple_groups):
    match = re.escape("1D sequences must be of shape (variables_n, )")
    with pytest.raises(ValueError, match=match):
        L = [10, 12, 15]  # wrong because it's of shape (groups_n, )
        bmb.Model("y ~ 0 + hsgp(x2, by=fac, L=L, m=10)", data_1d_multiple_groups)

    match = re.escape("2D sequences must be of shape (groups_n, variables_n)")
    with pytest.raises(ValueError, match=match):
        L = [[10], [12]]  # wrong because it's of shape (groups_n - 1, variables_n)
        bmb.Model("y ~ 0 + hsgp(x2, by=fac, L=L, m=10)", data_1d_multiple_groups)

    match = re.escape("1D sequences must be of shape (variables_n, )")
    with pytest.raises(ValueError, match=match):
        L = [10]  # wrong because it's of shape (variables_n - 1,)
        bmb.Model("outcome ~ 0 + hsgp(x, y, L=L, m=10)", data_2d_multiple_groups)

    match = re.escape("2D sequences must be of shape (groups_n, variables_n)")
    with pytest.raises(ValueError, match=match):
        L = [[10, 12], [15, 20]]  # wrong because it's of shape (variables_n - 1, groups_n - 1)
        bmb.Model("outcome ~ 0 + hsgp(x, y, by=group, L=L, m=10)", data_2d_multiple_groups)


def test_m_good_shape(data_1d_multiple_groups, data_2d_multiple_groups):
    m = 10
    model = bmb.Model("y ~ 0 + hsgp(x2, c=2, m=m)", data_1d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x2, c=2, m=m)"]
    assert (term.m == np.array([m])).all()

    m = 10
    model = bmb.Model("y ~ 0 + hsgp(x2, by=fac, c=2, m=m)", data_1d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x2, by=fac, c=2, m=m)"]
    assert (term.m == np.array([m])).all()

    m = [20, 10]
    model = bmb.Model("outcome ~ 0 + hsgp(x, y, c=2, m=m)", data_2d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x, y, c=2, m=m)"]
    assert (np.array(m) == term.m).all()

    m = [20, 10]
    model = bmb.Model("outcome ~ 0 + hsgp(x, y, by=group, c=2, m=m)", data_2d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x, y, by=group, c=2, m=m)"]
    assert (np.array(m) == term.m).all()


def test_c_good_shape(data_1d_multiple_groups, data_2d_multiple_groups):
    c = [[2], [1.5], [1.4]]  # (groups_n, variables_n)
    model = bmb.Model("y ~ 0 + hsgp(x2, by=fac, c=c, m=10)", data_1d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x2, by=fac, c=c, m=10)"]
    assert (term.c == np.array(c)).all()

    c = [2, 1.5]  # (variables_n, )
    model = bmb.Model("outcome ~ 0 + hsgp(x, y, by=group, c=c, m=10)", data_2d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x, y, by=group, c=c, m=10)"]
    assert term.c.shape == (3, 2)
    assert (term.c == np.tile(np.array(c), (3, 1))).all()

    c = [[2, 1.5], [1.8, 1.4], [1.6, 1.3]]  # (groups_n, variables_n)
    model = bmb.Model("outcome ~ 0 + hsgp(x, y, by=group, c=c, m=10)", data_2d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x, y, by=group, c=c, m=10)"]
    assert term.c.shape == (3, 2)
    assert (term.c == np.array(c)).all()


def test_L_good_shape(data_1d_multiple_groups, data_2d_multiple_groups):
    L = [[10], [12], [15]]  # (groups_n, variables_n)
    model = bmb.Model("y ~ 0 + hsgp(x2, by=fac, L=L, m=10)", data_1d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x2, by=fac, L=L, m=10)"]
    assert (term.L == np.array(L)).all()

    L = [10, 12]  # (variables_n, )
    model = bmb.Model("outcome ~ 0 + hsgp(x, y, by=group, L=L, m=10)", data_2d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x, y, by=group, L=L, m=10)"]
    assert term.L.shape == (3, 2)
    assert (term.L == np.tile(np.array(L), (3, 1))).all()

    L = [[10, 12], [15, 13], [16, 14]]  # (groups_n, variables_n)
    model = bmb.Model("outcome ~ 0 + hsgp(x, y, by=group, L=L, m=10)", data_2d_multiple_groups)
    term = model.components["mu"].terms["hsgp(x, y, by=group, L=L, m=10)"]
    assert term.L.shape == (3, 2)
    assert (term.L == np.array(L)).all()


def test_custom_priors_1d(data_1d_single_group):
    priors = {
        "hsgp(x, c=1.5, m=10)": {
            "sigma": bmb.Prior("Exponential", lam=0.75),
            "ell": bmb.Prior("Exponential", lam=1.25),
        }
    }
    model = bmb.Model("y ~ 0 + hsgp(x, c=1.5, m=10)", data_1d_single_group, priors=priors)
    model.build()

    weights_rv = model.backend.model["hsgp(x, c=1.5, m=10)_weights_raw"]
    ell_rv = model.backend.model["hsgp(x, c=1.5, m=10)_ell"]
    sigma_rv = model.backend.model["hsgp(x, c=1.5, m=10)_sigma"]
    assert weights_rv.eval().shape == (10,)
    assert ell_rv.eval().shape == ()
    assert sigma_rv.eval().shape == ()

    # Weird, but this is how we can check it.
    # PyMC asks for lam but then PyTensor uses 1 / lam
    assert 1 / sigma_rv.owner.inputs[-1].eval() == 0.75
    assert 1 / ell_rv.owner.inputs[-1].eval() == 1.25
    assert weights_rv.owner.inputs[-2].eval() == 0  # mu
    assert weights_rv.owner.inputs[-1].eval() == 1  # sigma


def test_custom_priors_2d_by_groups(data_2d_multiple_groups):
    priors = {
        "hsgp(x, y, by=group, c=1.5, m=10, share_cov=False)": {
            "sigma": bmb.Prior("Exponential", lam=5),
            "ell": bmb.Prior("Exponential", lam=2),
        }
    }
    model = bmb.Model(
        "outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10, share_cov=False)",
        data_2d_multiple_groups,
        priors=priors,
    )
    model.set_alias({"hsgp(x, y, by=group, c=1.5, m=10, share_cov=False)": "phi"})
    model.build()

    weights_rv = model.backend.model["phi_weights_raw"]
    ell_rv = model.backend.model["phi_ell"]
    sigma_rv = model.backend.model["phi_sigma"]

    assert weights_rv.eval().shape == (100, 3)  # (10x10, 3)
    assert ell_rv.eval().shape == (3,)
    assert sigma_rv.eval().shape == (3,)

    # Weird, but this is how we can check it.
    # PyMC asks for lam but then PyTensor uses 1 / lam
    assert 1 / sigma_rv.owner.inputs[-1].eval() == 5
    assert 1 / ell_rv.owner.inputs[-1].eval() == 2
    assert weights_rv.owner.inputs[-2].eval() == 0  # mu
    assert weights_rv.owner.inputs[-1].eval() == 1  # sigma


def test_custom_priors_2d_by_groups_anisotropic(data_2d_multiple_groups):
    priors = {
        "hsgp(x, y, by=group, c=1.5, m=10, share_cov=False, iso=False)": {
            "sigma": bmb.Prior("Exponential", lam=5),
            "ell": bmb.Prior("Exponential", lam=2),
        }
    }
    model = bmb.Model(
        "outcome ~ 0 + hsgp(x, y, by=group, c=1.5, m=10, share_cov=False, iso=False)",
        data_2d_multiple_groups,
        priors=priors,
    )
    model.set_alias({"hsgp(x, y, by=group, c=1.5, m=10, share_cov=False, iso=False)": "phi"})
    model.build()

    weights_rv = model.backend.model["phi_weights_raw"]
    ell_rv = model.backend.model["phi_ell"]
    sigma_rv = model.backend.model["phi_sigma"]

    assert weights_rv.eval().shape == (100, 3)  # (10x10, 3)
    assert ell_rv.eval().shape == (2, 3)  # (vars_n, groups_n)
    assert sigma_rv.eval().shape == (3,)  # (groups_n, )

    # Weird, but this is how we can check it.
    # PyMC asks for lam but then PyTensor uses 1 / lam
    assert 1 / sigma_rv.owner.inputs[-1].eval() == 5
    assert 1 / ell_rv.owner.inputs[-1].eval() == 2
    assert weights_rv.owner.inputs[-2].eval() == 0  # mu
    assert weights_rv.owner.inputs[-1].eval() == 1  # sigma


def test_bad_prior(data_1d_single_group):
    match = "The priors for an HSGP term must be passed within a dictionary"
    priors = {"hsgp(x, m=10, c=2)": bmb.Prior("Exponential", lam=1)}
    with pytest.raises(ValueError, match=match):
        bmb.Model("y ~ 0 + hsgp(x, m=10, c=2)", data_1d_single_group, priors=priors)


def test_minimal_1d_predicts(data_1d_single_group):
    model = bmb.Model("y ~ 0 + hsgp(x, c=1.5, m=10)", data_1d_single_group)
    idata = model.fit(tune=500, draws=500, chains=2, random_seed=1234)

    new_data = pd.DataFrame({"x": np.linspace(0, 5, num=10)})

    # Mean: In-sample
    new_idata = model.predict(idata, inplace=False)
    assert new_idata.posterior["mu"].dims == ("chain", "draw", "__obs__")
    assert new_idata.posterior["mu"].to_numpy().shape == (2, 500, 100)
    assert new_idata.posterior["hsgp(x, c=1.5, m=10)"].dims == ("chain", "draw", "__obs__")
    assert new_idata.posterior["hsgp(x, c=1.5, m=10)"].to_numpy().shape == (2, 500, 100)

    # Mean: Out-of-sample
    new_idata = model.predict(idata, data=new_data, inplace=False)
    assert new_idata.posterior["mu"].dims == ("chain", "draw", "__obs__")
    assert new_idata.posterior["mu"].to_numpy().shape == (2, 500, 10)
    assert new_idata.posterior["hsgp(x, c=1.5, m=10)"].dims == ("chain", "draw", "__obs__")
    assert new_idata.posterior["hsgp(x, c=1.5, m=10)"].to_numpy().shape == (2, 500, 10)

    # Posterior predictive: In-sample
    new_idata = model.predict(idata, kind="response", inplace=False)
    assert new_idata.posterior_predictive["y"].dims == ("chain", "draw", "__obs__")
    assert new_idata.posterior_predictive["y"].to_numpy().shape == (2, 500, 100)

    # Posterior predictive: Out-of-sample
    new_idata = model.predict(idata, data=new_data, kind="response", inplace=False)
    assert new_idata.posterior_predictive["y"].dims == ("chain", "draw", "__obs__")
    assert new_idata.posterior_predictive["y"].to_numpy().shape == (2, 500, 10)


def test_multiple_hsgp_and_by(data_1d_multiple_groups):
    rng = np.random.default_rng(1234)
    df = data_1d_multiple_groups.copy()
    df["fac2"] = rng.choice(["a", "b", "c"], size=df.shape[0])

    formula = "y ~ 1 + x0 + hsgp(x1, by=fac, m=10, c=2) + hsgp(x1, by=fac2, m=10, c=2)"
    model = bmb.Model(
        formula=formula,
        data=df,
        categorical=["fac"],
    )
    idata = model.fit(tune=400, draws=200, target_accept=0.9)

    bmb.interpret.plot_predictions(
        model, 
        idata, 
        conditional="x1", 
        subplot_kwargs={"main": "x1", "group": "fac2", "panel": "fac2"},
    );

    bmb.interpret.plot_predictions(
        model, 
        idata, 
        conditional={
            "x1": np.linspace(0, 1, num=100),
            "fac2": ["a", "b", "c"]
        }, 
        legend=False,
        subplot_kwargs={"main": "x1", "group": "fac2", "panel": "fac2"},
    );