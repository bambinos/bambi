from functools import reduce
from operator import add
from os.path import dirname, join

import numpy as np
import pandas as pd
import pytest

from formulae import design_matrices

from bambi.models import Model
from bambi.terms import Term, GroupSpecificTerm
from bambi.priors import Prior


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
    design = design_matrices("BMI", diabetes_data)
    term_info = design.common.terms_info["BMI"]
    term = Term("BMI", term_info, diabetes_data["BMI"])
    # Test that all defaults are properly initialized
    assert term.name == "BMI"
    assert not term.categorical
    assert not term.group_specific
    assert term.levels is not None
    assert term.data.shape == (442,)


def test_distribute_group_specific_effect_over(diabetes_data):
    # 163 unique levels of BMI in diabetes_data
    # With intercept
    model = Model(diabetes_data)
    model.fit("BP ~ (C(age_grp)|BMI)", run=False)
    # Since intercept is present, it uses treatment encoding
    lvls = sorted(list(diabetes_data["age_grp"].unique()))[1:]
    for lvl in lvls:
        assert model.terms[f"C(age_grp)[{lvl}]|BMI"].data.shape == (442, 163)
    assert "1|BMI" in model.terms

    # Without intercept
    model.reset()
    model.fit("BP ~ (0 + C(age_grp)|BMI)", run=False)
    assert model.terms["C(age_grp)[0]|BMI"].data.shape == (442, 163)
    assert model.terms["C(age_grp)[1]|BMI"].data.shape == (442, 163)
    assert model.terms["C(age_grp)[2]|BMI"].data.shape == (442, 163)
    assert not "1|BMI" in model.terms


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
    model.fit("BMI ~ age_grp + BP + S1", run=False)
    assert model.term_names == ["Intercept", "age_grp", "BP", "S1"]


def test_model_term_names_property_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = Model(crossed_data)
    model.fit("Y ~ threecats*fourcats", run=False)
    assert model.term_names == ["Intercept", "threecats", "fourcats", "threecats:fourcats"]


def test_model_terms_cleaned_levels_interaction(crossed_data):
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3
    model = Model(crossed_data)
    model.fit("Y ~ threecats*fourcats", run=False)
    assert model.terms["threecats:fourcats"].cleaned_levels == [
        "threecats[b]:fourcats[b]",
        "threecats[b]:fourcats[c]",
        "threecats[b]:fourcats[d]",
        "threecats[c]:fourcats[b]",
        "threecats[c]:fourcats[c]",
        "threecats[c]:fourcats[d]",
    ]


def test_model_terms_cleaned_levels():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=50),
            "x": np.random.normal(size=50),
            "z": reduce(add, [[f"Group {x}"] * 10 for x in ["1", "2", "3", "1", "2"]]),
            "time": list(range(1, 11)) * 5,
            "subject": reduce(add, [[f"Subject {x}"] * 10 for x in range(1, 6)]),
        }
    )
    model = Model(data)
    model.fit("y ~ x + z + time + (time|subject)", run=False)
    model.terms["z"].cleaned_levels == ["Group 2", "Group 3"]
    model.terms["1|subject"].cleaned_levels == [f"Subject {x}" for x in range(1, 6)]
    model.terms["time|subject"].cleaned_levels == [f"Subject {x}" for x in range(1, 6)]


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
    model.fit("y ~ x*g + (x|s)", run=False)

    assert isinstance(model.terms["x"], Term)
    assert isinstance(model.terms["g"], Term)
    assert isinstance(model.terms["x:g"], Term)
    assert isinstance(model.terms["1|s"], GroupSpecificTerm)
    assert isinstance(model.terms["x|s"], GroupSpecificTerm)

    # Also check 'categorical' attribute is right
    assert model.terms["g"].categorical


def test_one_shot_formula_fit(diabetes_data):
    model = Model(diabetes_data)
    model.fit("S3 ~ S1 + S2", draws=50)
    named_vars = model.backend.model.named_vars
    targets = ["S3", "S1", "Intercept"]
    assert len(set(named_vars.keys()) & set(targets)) == 3


def test_derived_term_search(diabetes_data):
    model = Model(diabetes_data)
    model.fit("BMI ~ 1 + (age_grp|BP)", categorical=["age_grp"], run=False)
    terms = model._match_derived_terms("age_grp|BP")
    names = set([t.name for t in terms])

    # Since intercept is present, it uses treatment encoding
    lvls = sorted(list(diabetes_data["age_grp"].unique()))[1:]
    assert names == set(["1|BP"] + [f"age_grp[{lvl}]|BP" for lvl in lvls])

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
    model.fit("y ~ x1 + x2 + g1 + (g1|g2) + (x2|g2)", run=False)
    terms = ["x1", "x2", "g1", "1|g2", "g1[b]|g2", "x2|g2"]
    expecteds = [False, False, True, False, True, False]

    for term, expected in zip(terms, expecteds):
        assert model.terms[term].categorical is expected


def test_omit_offsets_false():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = Model(data)
    fitted = model.fit("y ~ x1 + (x1|g1)", omit_offsets=False)
    offsets = [v for v in fitted.posterior.dims if "offset" in v]
    assert offsets == ["1|g1_offset_dim_0", "x1|g1_offset_dim_0"]


def test_omit_offsets_true():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = Model(data)
    fitted = model.fit("y ~ x1 + (x1|g1)", omit_offsets=True)
    offsets = [v for v in fitted.posterior.dims if "offset" in v]
    assert not offsets


def test_hyperprior_on_common_effect():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    sigma = Prior("HalfCauchy", beta=2)
    slope = Prior("Normal", mu=0, sd=sigma)
    priors = {"x1": slope}
    model = Model(data)
    
    with pytest.raises(ValueError):
        model.fit("y ~ x1 + (x1|g1)", priors=priors)

    priors = {"common": slope}
    with pytest.raises(ValueError):
        model.fit("y ~ x1 + (x1|g1)", priors=priors)


def test_set_formula_and_then_fit():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = Model(data)
    model.fit("y ~ x1 + (x1|g1)", run=False)
    model.fit(draws=200)  


def test_formula_overwrite():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "x2": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = Model(data)
    model.fit("y ~ x1 + (x1|g1)", run=False)
    assert "x1" in model.terms
    assert "x1|g1" in model.terms
    model.fit("y ~ x2 + (x2|g1)")
    assert "x1" not in model.terms
    assert "x1|g1" not in model.terms
    assert "x2" in model.terms
    assert "x2|g1" in model.terms


def test_empty_formula_assertion():
    data = pd.DataFrame(
        {
            "y": np.random.normal(size=100),
            "x1": np.random.normal(size=100),
            "g1": ["a"] * 50 + ["b"] * 50,
        }
    )
    model = Model(data)
    # ValueError when attempt to fit a model without having passed a formula
    with pytest.raises(ValueError):
        model.fit()
    # But if you then pass a formula, you can fit.
    model.fit("y ~ x1 + (x1|g1)", run=False)
    model.fit(draws=200)
