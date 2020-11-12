from os.path import dirname, join

import numpy as np
import pandas as pd
import pytest

from bambi.models import Model, Term


@pytest.fixture(scope="module")
def diabetes_data():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "diabetes.txt"), sep="\t")
    data["age_grp"] = 0
    data.loc[data["AGE"] > 40, "age_grp"] = 1
    data.loc[data["AGE"] > 60, "age_grp"] = 2
    return data


@pytest.fixture(scope="module")
def base_model(diabetes_data):
    return Model(diabetes_data)


def test_term_init(diabetes_data):
    model = Model(diabetes_data)
    term = Term("BMI", diabetes_data["BMI"])
    # Test that all defaults are properly initialized
    assert term.name == "BMI"
    assert term.categorical == False
    assert not term.random
    assert term.levels is not None
    assert term.data.shape == (442, 1)


def test_distribute_random_effect_over(diabetes_data):
    # Random slopes
    model = Model(diabetes_data)
    model.add("BP ~ 1")
    model.add(random="C(age_grp)|BMI")
    model.build(backend="pymc")
    assert model.terms["C(age_grp)[T.1]|BMI"].data.shape == (442, 163)
    # Nested or crossed random intercepts
    model.reset()
    model.add("BP ~ 1")
    model.add(random="0+C(age_grp)|BMI")
    model.build(backend="pymc")
    assert model.terms["C(age_grp)[0]|BMI"].data.shape == (442, 163)
    # 163 unique levels of BMI in diabetes_data


def test_model_init_from_filename():
    from os.path import dirname, join

    data_dir = join(dirname(__file__), "data")
    filename = join(data_dir, "diabetes.txt")
    model = Model(filename)
    assert isinstance(model.data, pd.DataFrame)
    assert model.data.shape == (442, 11)
    assert "BMI" in model.data.columns


def test_model_term_names_property(diabetes_data):
    model = Model(diabetes_data)
    model.add("BMI ~ age_grp")
    model.add("BP")
    model.add("S1")
    model.build(backend="pymc")
    assert model.term_names == ["Intercept", "age_grp", "BP", "S1"]


def test_add_to_model(diabetes_data):
    model = Model(diabetes_data)
    model.add("BP ~ BMI")
    model.build(backend="pymc")
    assert isinstance(model.terms["BMI"], Term)
    model.add("age_grp")
    model.build(backend="pymc")
    assert set(model.terms.keys()) == {"Intercept", "BMI", "age_grp"}
    # Test that arguments are passed appropriately onto Term initializer
    model.add(random="C(age_grp)|BP")
    model.build(backend="pymc")
    assert isinstance(model.terms["C(age_grp)[T.1]|BP"], Term)
    assert "BP[108.0]" in model.terms["C(age_grp)[T.1]|BP"].levels


def test_one_shot_formula_fit(diabetes_data):
    model = Model(diabetes_data)
    model.fit("S3 ~ S1 + S2", draws=50, run=False)
    model.build(backend="pymc3")
    nv = model.backend.model.named_vars
    targets = ["S3", "S1", "Intercept"]
    assert len(set(nv.keys()) & set(targets)) == 3


def test_invalid_chars_in_random_effect(diabetes_data):
    model = Model(diabetes_data)
    with pytest.raises(ValueError):
        model.fit(random=["1+BP|age_grp"])


def test_add_formula_append(diabetes_data):
    model = Model(diabetes_data)
    model.add("S3 ~ 0")
    model.add("S1")
    model.build(backend="pymc")
    assert hasattr(model, "y") and model.y is not None and model.y.name == "S3"
    assert "S1" in model.terms
    model.add("S2", append=False)
    assert model.y is None
    model.add("S3 ~ 0")
    model.build(backend="pymc")
    assert "S2" in model.terms
    assert "S1" not in model.terms


def test_derived_term_search(diabetes_data):
    model = Model(diabetes_data)
    model.add("BMI ~ 1", random="age_grp|BP", categorical=["age_grp"])
    model.build(backend="pymc")
    terms = model._match_derived_terms("age_grp|BP")
    names = set([t.name for t in terms])
    assert names == {"1|BP", "age_grp[T.1]|BP", "age_grp[T.2]|BP"}

    term = model._match_derived_terms("1|BP")[0]
    assert term.name == "1|BP"

    # All of these should find nothing
    assert model._match_derived_terms("1|ZZZ") is None
    assert model._match_derived_terms("ZZZ|BP") is None
    assert model._match_derived_terms("BP") is None
    assert model._match_derived_terms("BP") is None


def test_categorical_term():
    np.random.seed(303456)
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=6),
            "x1": np.random.normal(size=6),
            "x2": [1, 1, 0, 0, 1, 1],
            "g1": ["a"] * 3 + ["b"] * 3,
            "g2": ["x", "x", "z", "z", "y", "y"],
        }
    )
    model = Model(data)
    model.add("y ~ x1 + x2 + g1", random=["g1|g2", "x2|g2"])
    model.build()
    terms = ["x1", "x2", "g1", "1|g2", "g1[T.b]|g2", "x2|g2"]
    expecteds = [False, False, True, False, True, False]

    for term, expected in zip(terms, expecteds):
        assert model.terms[term].categorical is expected
