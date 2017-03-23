import pytest
from os.path import dirname, join
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from bambi.models import Term, Model


@pytest.fixture(scope="module")
def diabetes_data():
    data_dir = join(dirname(__file__), 'data')
    data = pd.read_csv(join(data_dir, 'diabetes.txt'), sep='\t')
    data['age_grp'] = 0
    data['age_grp'][data['AGE'] > 40] = 1
    data['age_grp'][data['AGE'] > 60] = 2
    return data


@pytest.fixture(scope="module")
def base_model(diabetes_data):
    return Model(diabetes_data)


def test_term_init(diabetes_data):
    # model = Model(diabetes_data)
    # term = Term(model, 'BMI', diabetes_data['BMI'])
    term = Term('BMI', diabetes_data['BMI'])
    # Test that all defaults are properly initialized
    assert term.name == 'BMI'
    assert term.categorical == False
    assert not term.random
    assert term.levels is not None
    assert term.data.shape == (442, 1)


def test_distribute_random_effect_over(diabetes_data):
    # Random slopes
    model = Model(diabetes_data)
    model.add(random='C(age_grp)|BMI')
    assert model.terms['C(age_grp)[T.1]|BMI'].data.shape == (442, 120)
    # Nested or crossed random intercepts
    model.reset()
    model.add(random='0+C(age_grp)|BMI')
    assert model.terms['C(age_grp)[0]|BMI'].data.shape == (442, 83)


def test_model_init_from_filename():
    from os.path import dirname, join
    data_dir = join(dirname(__file__), 'data')
    filename = join(data_dir, 'diabetes.txt')
    model = Model(filename)
    assert isinstance(model.data, pd.DataFrame)
    assert model.data.shape == (442, 11)
    assert 'BMI' in model.data.columns


def test_model_term_names_property(diabetes_data):
    model = Model(diabetes_data)
    model.add('BMI')
    model.add('BP')
    model.add('S1')
    assert model.term_names == ['Intercept' ,'BMI', 'BP', 'S1']


def test_add_to_model(diabetes_data):
    model = Model(diabetes_data)
    model.add('BMI')
    assert isinstance(model.terms['BMI'], Term)
    model.add('age_grp')
    assert set(model.terms.keys()) == {'Intercept' ,'BMI', 'age_grp'}
    # Test that arguments are passed appropriately onto Term initializer
    model.add(random='C(age_grp)|BP')
    assert isinstance(model.terms['C(age_grp)[T.1]|BP'], Term)
    assert 'BP[108.0]' in model.terms['C(age_grp)[T.1]|BP'].levels


def test_reduced_data_representation_for_categoricals(diabetes_data):
    # Test that terms made up entirely of dummy columns are properly re-encoded
    # as 1D arrays of level indices.
    model = Model(diabetes_data)

    model.add('C(BMI)')
    term = model.terms['C(BMI)']
    assert term.data.shape[1] == 162
    assert term._reduced_data is None
    model.add('0 + C(BMI)')
    term = model.terms['C(BMI)']
    assert term.data.shape[1] == 163
    assert term._reduced_data.shape[1] == 1
    assert term._reduced_data.max() == 162


def test_one_shot_formula_fit(diabetes_data):
    model = Model(diabetes_data)
    model.fit('S3 ~ S1 + S2', samples=50, run=False)
    model.build(backend='pymc3')
    nv = model.backend.model.named_vars
    targets = ['S3', 'S1', 'Intercept']
    assert len(set(nv.keys()) & set(targets)) == 3


def test_invalid_chars_in_random_effect(diabetes_data):
    model = Model(diabetes_data)
    with pytest.raises(ValueError):
        model.fit(random=['1+BP|age_grp'])


def test_add_formula_append(diabetes_data):
    model = Model(diabetes_data)
    model.add('S3 ~ 0')
    model.add('S1')
    assert hasattr(model, 'y') and model.y is not None and model.y.name == 'S3'
    assert 'S1' in model.terms
    model.add('S2', append=False)
    assert model.y is None
    assert 'S2' in model.terms
    assert 'S1' not in model.terms


def test_derived_term_search(diabetes_data):
    model = Model(diabetes_data)
    model.add(random='age_grp|BP', categorical=['age_grp'])
    terms = model._match_derived_terms('age_grp|BP')
    names = set([t.name for t in terms])
    assert names == {'1|BP', 'age_grp[T.1]|BP', 'age_grp[T.2]|BP'}

    term = model._match_derived_terms('1|BP')[0]
    assert term.name == '1|BP'

    # All of these should find nothing
    assert model._match_derived_terms('1|ZZZ') is None
    assert model._match_derived_terms('ZZZ|BP') is None
    assert model._match_derived_terms('BP') is None
    assert model._match_derived_terms('BP') is None
