from os.path import dirname, join

import numpy as np
import pandas as pd
import pytest

from bambi.models import Model, Term, GroupSpecificTerm, InteractionTerm


@pytest.fixture(scope="module")
def diabetes_data():
    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "diabetes.txt"), sep="\t")
    data["age_grp"] = 0
    data.loc[data["AGE"] > 40, "age_grp"] = 1
    data.loc[data["AGE"] > 60, "age_grp"] = 2
    return data


@pytest.fixture(scope="module")
def crossed_data():
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
    from os.path import dirname, join

    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "crossed_random.csv"))
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
    assert not term.group_specific
    assert term.levels is not None
    assert term.data.shape == (442, 1)


def test_distribute_group_specific_effect_over(diabetes_data):
    # Group_specific slopes
    model = Model(diabetes_data)
    model.add("BP ~ 1")
    model.add(group_specific="C(age_grp)|BMI")
    model.build(backend="pymc")
    assert model.terms["C(age_grp)[T.1]|BMI"].data.shape == (442, 163)
    # Nested or crossed group specific intercepts
    model.reset()
    model.add("BP ~ 1")
    model.add(group_specific="0+C(age_grp)|BMI")
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


def test_model_term_names_property_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = Model(crossed_data)
    fitted = model.fit("Y ~ threecats*fourcats")
    assert model.term_names == ["Intercept", "threecats", "fourcats", "threecats:fourcats"]


def test_model_terms_cleaned_levels_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = Model(crossed_data)
    fitted = model.fit("Y ~ threecats*fourcats")
    assert model.terms["threecats:fourcats"].cleaned_levels == [
        "threecats[b]:fourcats[b]",
        "threecats[c]:fourcats[b]",
        "threecats[b]:fourcats[c]",
        "threecats[c]:fourcats[c]",
        "threecats[b]:fourcats[d]",
        "threecats[c]:fourcats[d]",
    ]


def test_model_terms_cleaned_levels():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=50),
            "x": np.random.normal(size=50),
            "z": ["Group 1"] * 10
            + ["Group 2"] * 10
            + ["Group 3"] * 10
            + ["Group 1"] * 10
            + ["Group 2"] * 10,
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "subject": ["Subject 1"] * 10
            + ["Subject 2"] * 10
            + ["Subject 3"] * 10
            + ["Subject 4"] * 10
            + ["Subject 5"] * 10,
        }
    )
    model = Model(data)
    fitted = model.fit("y ~ x + z + time", group_specific=["time|subject"])
    model.terms["z"].cleaned_levels == ["Group 2", "Group 3"]
    model.terms["1|subject"].cleaned_levels == [
        "Subject 1",
        "Subject 2",
        "Subject 3",
        "Subject 4",
        "Subject 5",
    ]
    model.terms["time|subject"].cleaned_levels == [
        "Subject 1",
        "Subject 2",
        "Subject 3",
        "Subject 4",
        "Subject 5",
    ]


def test_model_term_classes():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=50),
            "x": np.random.normal(size=50),
            "s": ["s1"] * 25 + ["s2"] * 25,
            "g": np.random.choice(["a", "b", "c"], size=50),
        }
    )

    model = Model(data)
    fitted = model.fit("y ~ x*g", group_specific=["x|s"])

    assert isinstance(model.terms["g"], Term)
    assert isinstance(model.terms["x"], Term)
    assert isinstance(model.terms["x|s"], GroupSpecificTerm)
    assert isinstance(model.terms["1|s"], GroupSpecificTerm)
    assert isinstance(model.terms["x:g"], InteractionTerm)

    # Also check 'categorical' attribute is right
    assert model.terms["g"].categorical


def test_add_to_model(diabetes_data):
    model = Model(diabetes_data)
    model.add("BP ~ BMI")
    model.build(backend="pymc")
    assert isinstance(model.terms["BMI"], Term)
    model.add("age_grp")
    model.build(backend="pymc")
    assert set(model.terms.keys()) == {"Intercept", "BMI", "age_grp"}
    # Test that arguments are passed appropriately onto Term initializer
    model.add(group_specific="C(age_grp)|BP")
    model.build(backend="pymc")
    assert isinstance(model.terms["C(age_grp)[T.1]|BP"], Term)
    assert "108.0" in model.terms["C(age_grp)[T.1]|BP"].cleaned_levels


def test_one_shot_formula_fit(diabetes_data):
    model = Model(diabetes_data)
    model.fit("S3 ~ S1 + S2", draws=50, run=False)
    model.build(backend="pymc3")
    nv = model.backend.model.named_vars
    targets = ["S3", "S1", "Intercept"]
    assert len(set(nv.keys()) & set(targets)) == 3


def test_invalid_chars_in_group_specific_effect(diabetes_data):
    model = Model(diabetes_data)
    with pytest.raises(ValueError):
        model.fit(group_specific=["1+BP|age_grp"])


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
    model.add("BMI ~ 1", group_specific="age_grp|BP", categorical=["age_grp"])
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
    model.add("y ~ x1 + x2 + g1", group_specific=["g1|g2", "x2|g2"])
    model.build()
    terms = ["x1", "x2", "g1", "1|g2", "g1[T.b]|g2", "x2|g2"]
    expecteds = [False, False, True, False, True, False]

    for term, expected in zip(terms, expecteds):
        assert model.terms[term].categorical is expected


def test_keep_offsets_true():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = Model(data)
    fitted = model.fit("y ~ x1", group_specific=["x1|g1"], keep_offsets=True)
    offsets = [v for v in fitted.posterior.dims if "offset" in v]
    assert offsets == ["1|g1_offset_dim_0", "x1|g1_offset_dim_0"]


def test_keep_offsets_false():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = Model(data)
    fitted = model.fit("y ~ x1", group_specific=["x1|g1"], keep_offsets=False)
    offsets = [v for v in fitted.posterior.dims if "offset" in v]
    assert offsets == []
