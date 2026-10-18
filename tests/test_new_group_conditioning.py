import numpy as np
import pandas as pd
import pytest

import bambi as bmb


@pytest.fixture
def conditioning_data():
    return pd.DataFrame(
        {
            "y": np.arange(9, dtype=float),
            "x": np.tile([1.0, 2.0, 3.0], 3),
            "g": np.repeat(["a", "b", "c"], 3),
        }
    )


@pytest.mark.parametrize("sparse_dot", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("case", ["location", "scale", "slope", "alias", "categorical"])
def test_new_group_conditioning(conditioning_data, mock_pymc_sample, monkeypatch, sparse_dot, case):
    monkeypatch.setattr(bmb.config, "SPARSE_DOT", sparse_dot)
    data = conditioning_data
    term = "x|g" if case == "slope" else "1|g"
    formula = "y ~ 0 + (0 + x|g)" if case == "slope" else "y ~ 0 + (1|g)"
    family = "categorical" if case == "categorical" else "gaussian"
    if case == "categorical":
        data = data.assign(y=pd.Categorical(["a", "b", "c"] * 3))
    prior = bmb.Prior(
        "Normal",
        mu=bmb.Prior("Normal", mu=2, sigma=1),
        sigma=bmb.Prior("HalfNormal", sigma=1),
    )
    model = bmb.Model(formula, data, family=family, noncentered=False, priors={term: prior})
    if case == "alias":
        model.set_alias({term: "group_effect", "sigma": "group_sd"})
        term = "group_effect"
    idata = model.fit(draws=4, chains=2, random_seed=12)
    new_data = pd.DataFrame({"g": ["a", "new_a", "new_a", "new_b"], "x": [1.0, 2.0, 3.0, 4.0]})
    before = model.predict(idata, data=new_data, random_seed=42, inplace=False)
    changed = idata.copy(deep=True)
    location = f"{term}_mu"
    if case == "scale":
        changed.posterior[f"{term}_sigma"] *= 2
    else:
        changed.posterior[term] += 3
        changed.posterior[location] += 3
    after = model.predict(changed, data=new_data, random_seed=42, inplace=False)

    if case == "categorical":
        before_p = before.predictions["p"].values
        after_p = after.predictions["p"].values
        assert before_p.shape == after_p.shape == (2, 4, 4, 3)
        delta = np.log(after_p[..., 1:] / after_p[..., :1]) - np.log(
            before_p[..., 1:] / before_p[..., :1]
        )
        np.testing.assert_allclose(delta, 3, atol=1e-8, rtol=0)
        return

    before_mu = before.predictions["mu"].values
    after_mu = after.predictions["mu"].values
    assert before_mu.shape == after_mu.shape == (2, 4, 4)
    if case == "scale":
        np.testing.assert_allclose(after_mu[..., 0], before_mu[..., 0], atol=1e-8, rtol=0)
        mu = idata.posterior[location].values[..., None]
        np.testing.assert_allclose(
            after_mu[..., 1:] - mu, 2 * (before_mu[..., 1:] - mu), atol=1e-8, rtol=0
        )
    else:
        expected = 3 * new_data["x"].values if case == "slope" else np.full(4, 3.0)
        np.testing.assert_allclose(
            after_mu - before_mu, np.broadcast_to(expected, before_mu.shape), atol=1e-8, rtol=0
        )
        if case == "slope":
            np.testing.assert_allclose(
                before_mu[..., 1] / 2, before_mu[..., 2] / 3, atol=1e-8, rtol=0
            )
