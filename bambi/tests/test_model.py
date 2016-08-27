import pytest
from bambi.models import Term, Model
from bambi.priors import Prior
from os.path import dirname, join
import pandas as pd


@pytest.fixture(scope="module")
def diabetes_data():
    from os.path import dirname, join
    data_dir = join(dirname(__file__), 'data')
    data = pd.read_csv(join(data_dir, 'diabetes.txt'), sep='\t')
    data['age_grp'] = 0
    data['age_grp'][data['AGE'] > 40] = 1
    data['age_grp'][data['AGE'] > 60] = 2
    return data


@pytest.fixture(scope="module")
def crossed_data():
    '''
    Random effects:
    10 subjects, 12 items, 5 sites
    Subjects crossed with items, nested in sites
    Items crossed with sites

    Fixed effects:
    A continuous predictor, a numeric dummy, and a three-level category (levels a,b,c)

    Structure:
    Subjects nested in dummy (e.g., gender), crossed with threecats
    Items crossed with dummy, nested in threecats
    Sites partially crossed with dummy (4/5 see a single dummy, 1/5 sees both dummies)
    Sites crossed with threecats
    '''
    from os.path import dirname, join
    data_dir = join(dirname(__file__), 'data')
    data = pd.read_csv(join(data_dir, 'crossed_random.csv'))
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
    assert term.type_ == 'fixed'
    assert term.levels is not None
    assert term.data.shape == (442, 1)


def test_term_split(diabetes_data):
    # Split a continuous fixed variable
    model = Model(diabetes_data)
    model.add_term('BMI', split_by='age_grp')
    assert model.terms['BMI'].data.shape == (442, 2)
    # Split a categorical fixed variable
    model.reset()
    model.add_term('BMI', split_by='age_grp', categorical=True,
                   drop_first=False)
    assert model.terms['BMI'].data.shape == (442, 489)
    # Split a continuous random variable
    model.reset()
    model.add_term('BMI', split_by='age_grp', categorical=False, random=True,
                   drop_first=True)
    assert model.terms['BMI'].data.shape == (442, 2)
    # Split a categorical random variable
    model.reset()
    model.add_term('BMI', split_by='age_grp', categorical=True, random=True,
                   drop_first=False)
    t = model.terms['BMI'].data
    assert isinstance(t, dict)
    assert t['age_grp[0]'].shape == (442, 83)


def test_model_init_from_filename():
    from os.path import dirname, join
    data_dir = join(dirname(__file__), 'data')
    filename = join(data_dir, 'diabetes.txt')
    model = Model(filename)
    assert isinstance(model.data, pd.DataFrame)
    assert model.data.shape == (442, 11)
    assert 'BMI' in model.data.columns


def test_model_init_and_intercept(diabetes_data):

    model = Model(diabetes_data, intercept=True)
    assert hasattr(model, 'data')
    assert 'Intercept' in model.terms
    assert len(model.terms) == 1
    assert model.y is None
    assert hasattr(model, 'backend')
    model = Model(diabetes_data)
    assert 'Intercept' not in model.terms
    assert not model.terms


def test_model_term_names_property(diabetes_data):
    model = Model(diabetes_data)
    model.add_term('BMI')
    model.add_term('BP')
    model.add_term('S1')
    assert model.term_names == ['BMI', 'BP', 'S1']


def test_add_term_to_model(base_model):

    base_model.add_term('BMI')
    assert isinstance(base_model.terms['BMI'], Term)
    base_model.add_term('age_grp', random=False, categorical=True)
    # Test that arguments are passed appropriately onto Term initializer
    base_model.add_term(
        'BP', random=True, split_by='age_grp', categorical=True)
    assert isinstance(base_model.terms['BP'], Term)


def test_one_shot_formula_fit(base_model):
    base_model.fit('BMI ~ S1 + S2', samples=50)
    nv = base_model.backend.model.named_vars
    targets = ['likelihood', 'b_S1', 'b_Intercept']
    assert len(set(nv.keys()) & set(targets)) == 3
    assert len(base_model.backend.trace) == 50


def test_invalid_chars_in_random_effect(base_model):
    with pytest.raises(ValueError):
        base_model.fit(random=['1+BP|age_grp'])


def test_update_term_priors_after_init(diabetes_data):
    model = Model(diabetes_data)
    model.add_term('BMI')
    model.add_term('S1')
    model.add_term('BP', random=True, split_by='age_grp')

    p1 = Prior('Normal', mu=-10, sd=10)
    p2 = Prior('Beta', alpha=2, beta=2)

    model.set_priors({'BMI': 0.3, 'S1': p2})
    assert model.terms['S1'].prior.args['beta'] == 2
    assert model.terms['BMI'].prior == 0.3

    model.set_priors({('S1', 'BMI'): p1})
    assert model.terms['S1'].prior.args['sd'] == 10
    assert model.terms['BMI'].prior.args['mu'] == -10

    p3 = Prior('Normal', mu=0, sd=Prior('Normal', mu=0, sd=7))
    model.set_priors(fixed=0.4, random=p3)
    assert model.terms['BMI'].prior == 0.4
    assert model.terms['BP'].prior.args['sd'].args['sd'] == 7

# All tests above this point do not build the model.
# All tests below this point build the model, but do not fit it

def test_empty_model(crossed_data):
    # using formula
    model0 = Model(crossed_data)
    model0.add_y('Y')
    model0.build()

    # using add_term
    model1 = Model(crossed_data)
    model1.fit('Y ~ 0', run=False)
    model1.build()


def test_intercept_only_model(crossed_data):
    # using formula
    model0 = Model(crossed_data)
    model0.add_y('Y')
    model0.add_intercept()
    model0.build()

    # using add_term
    model1 = Model(crossed_data)
    model1.fit('Y ~ 1', run=False)
    model1.build()


def test_fixed_only_and_check_agreement(crossed_data):
    # build model using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ continuous + dummy + threecats', run=False)
    model0.build()

    # build model using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    model1.add_intercept()
    model1.add_term('continuous')
    model1.add_term('dummy')
    model1.add_term('threecats')
    model1.build()

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:,lev]) for t in model0.terms.values() for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:,lev]) for t in model1.terms.values() for lev in range(len(t.levels))])
    assert X0 == X1


def test_three_level_categorical_cell_means_parameterization(crossed_data):
    # build model using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ 0 + threecats', run=False)
    model0.build()

    # build model using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    model1.add_term('threecats', drop_first=False)
    model1.build()

    # check that design matries are the same,
    # even if term names / level names / order of columns is different
    X0 = set([tuple(t.data[:,lev]) for t in model0.terms.values() for lev in range(len(t.levels))])
    X1 = set([tuple(t.data[:,lev]) for t in model1.terms.values() for lev in range(len(t.levels))])
    assert X0 == X1


def test_random_intercepts(crossed_data):
    # using formula and 'subj' syntax
    model0 = Model(crossed_data)
    model0.fit('Y ~ continuous', random=['subj','item','site'], run=False)
    model0.build()

    # using formula and '1|' syntax
    model1 = Model(crossed_data)
    model1.fit('Y ~ continuous', random=['1|subj','1|item','1|site'], run=False)
    model1.build()

    # check that they have the same random terms
    assert set(model0.random_terms) == set(model1.random_terms)

    # using add_term
    model2 = Model(crossed_data)
    model2.add_y('Y')
    model2.add_term('continuous')
    model2.add_term('subj', random=True)
    model2.add_term('item', random=True)
    model2.add_term('site', random=True)
    model2.build()

    # check that this has the same random terms as above
    assert set(model0.random_terms) == set(model2.random_terms)


def test_random_and_check_agreement(crossed_data):
    # build model using formula
    model0 = Model(crossed_data)
    model0.fit('Y ~ continuous',
        random=['0+threecats|subj','continuous|item','dummy|item','threecats|site'], run=False)
    model0.build()

    # build model using add_term
    model1 = Model(crossed_data)
    model1.add_y('Y')
    # fixed effects
    model1.add_intercept()
    model1.add_term('continuous')
    # random effects
    model1.add_term('subj', split_by='threecats', drop_first=False, random=True)
    model1.add_term('continuous', split_by='item', random=True)
    model1.add_term('item', split_by='dummy', random=True)
    model1.add_term('site', random=True)
    model1.add_term('site', split_by='threecats', random=True)
    model1.build()

    # check that they have the same random terms
    assert set(model0.random_terms) == set(model1.random_terms)

